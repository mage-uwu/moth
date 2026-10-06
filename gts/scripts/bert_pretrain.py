# Golden Tree Snake (GTS) fork, 2026.
"""BERT-style pretraining of a bidirectional ternary GTS mixed forest on English Wikipedia.

    python scripts/bert_pretrain.py prep  --out /workspace/wiki --train-tokens 1500000000
    python scripts/bert_pretrain.py train --data /workspace/wiki --out /workspace/run --minutes 140

prep:  BERT's uncased WordPiece tokenizer (google-bert/bert-base-uncased, vocabulary 30,522) over the parquet shards of
       wikimedia/wikipedia 20231101.en. Articles are joined with [SEP] into one uint16 stream; the validation tokens come
       from the last shard, the training tokens from the first ones. (BookCorpus, BERT's other source, has no reliable
       public copy any more.)
train: RoBERTa-style masked LM: 512-token windows of the stream starting with [CLS], 15% of the non-special tokens
       chosen, of those 80% [MASK], 10% a random token, 10% unchanged; no next-sentence objective. bf16 autocast,
       torch.compile per block, fused AdamW. The learning-rate schedule (linear warmup, cosine to 10%) is fitted to
       --minutes: after the first steps the step rate is measured and the total set so the run ends in time.
       Writes, to --out: result.json (curve, settings), checkpoint.pt (float: latent weights, AdamW state, step,
       config, curve; also every --ckpt-minutes) and binarized.pt (2-bit ternary codes and scales plus the float
       tensors; mamba_ssm/utils/ternary_pack.py loads it back into an identical model).

``prep --text-file FILE`` tokenises a local text file instead, for smoke tests.
"""

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mamba_ssm.models.gts_encoder import GTSConfig, GTSForMaskedLM  # noqa: E402
from mamba_ssm.utils.ternary_pack import save_binarized  # noqa: E402

TOKENIZER_REPO = "google-bert/bert-base-uncased"


# ----------------------------------------------------------------------------------------------------------- prep


def _tokenizer():
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(hf_hub_download(TOKENIZER_REPO, "tokenizer.json"))
    tok.no_padding()
    tok.no_truncation()
    return tok


def _encode(texts, tok, sep):
    """One uint16 array for a list of documents, each followed by [SEP]."""
    parts = []
    for start in range(0, len(texts), 4096):
        for enc in tok.encode_batch(texts[start : start + 4096], add_special_tokens=False):
            parts.append(np.asarray(enc.ids, dtype=np.uint16))
            parts.append(np.asarray([sep], dtype=np.uint16))
    return np.concatenate(parts) if parts else np.zeros(0, np.uint16)


def _write_stream(texts, tok, f, limit, written, sep):
    ids = _encode(texts, tok, sep)[: limit - written]
    ids.tofile(f)
    return written + len(ids)


def _shard_worker(job):
    """Tokenise one Wikipedia parquet shard (up to ``limit`` tokens) into its own file. Runs in a worker process."""
    name, path_out, limit = job
    os.environ.setdefault("RAYON_NUM_THREADS", "4")
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    tok = _tokenizer()
    sep = tok.token_to_id("[SEP]")
    pf = pq.ParquetFile(hf_hub_download("wikimedia/wikipedia", name, repo_type="dataset"))
    n = 0
    with open(path_out, "wb") as f:
        for g in range(pf.num_row_groups):
            n = _write_stream(pf.read_row_group(g, columns=["text"]).column("text").to_pylist(), tok, f, limit, n, sep)
            if n >= limit:
                break
    return n


def prep(a):
    tok = _tokenizer()
    sep = tok.token_to_id("[SEP]")
    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    if a.text_file:
        text = open(a.text_file, encoding="utf-8", errors="replace").read()
        docs = [d for d in text.split("\n\n") if d.strip()]
        cut = max(1, len(docs) // 20)
        with open(os.path.join(a.out, "val.bin"), "wb") as f:
            n_val = _write_stream(docs[:cut], tok, f, a.val_tokens, 0, sep)
        with open(os.path.join(a.out, "train.bin"), "wb") as f:
            n_train = _write_stream(docs[cut:], tok, f, a.train_tokens, 0, sep)
        source = f"bert-base-uncased WordPiece of {a.text_file}"
    else:
        import multiprocessing as mp
        import shutil

        from huggingface_hub import list_repo_files

        files = sorted(f for f in list_repo_files("wikimedia/wikipedia", repo_type="dataset") if f.startswith("20231101.en/"))
        per_shard = min(160_000_000, a.train_tokens)  # an English shard holds about 150M word pieces
        n_train_shards = min(len(files) - 1, -(-a.train_tokens // per_shard) + 1)
        print(f"{len(files)} parquet shards; tokenising {n_train_shards} for training and the last for validation, "
              f"{a.workers} at a time", flush=True)
        tmp = os.path.join(a.out, "shards")
        os.makedirs(tmp, exist_ok=True)
        jobs = [(files[-1], os.path.join(tmp, "val.bin"), a.val_tokens)]
        jobs += [(files[i], os.path.join(tmp, f"train{i:02d}.bin"), per_shard) for i in range(n_train_shards)]
        with mp.get_context("spawn").Pool(a.workers) as pool:
            counts = pool.map(_shard_worker, jobs, chunksize=1)
        print(f"  tokenised {sum(counts):,} tokens in {time.time() - t0:.0f} s", flush=True)
        shutil.move(jobs[0][1], os.path.join(a.out, "val.bin"))
        n_val, n_train = counts[0], 0
        with open(os.path.join(a.out, "train.bin"), "wb") as out:
            for (_, path, _), n in zip(jobs[1:], counts[1:]):
                take = min(n, a.train_tokens - n_train)
                with open(path, "rb") as f:
                    out.write(f.read(take * 2))
                n_train += take
                os.remove(path)
                if n_train >= a.train_tokens:
                    break
        shutil.rmtree(tmp, ignore_errors=True)
        source = "bert-base-uncased WordPiece of wikimedia/wikipedia 20231101.en"
    meta = {"vocab_size": tok.get_vocab_size(), "source": source, "train_tokens": n_train, "val_tokens": n_val,
            "special": {k: tok.token_to_id(k) for k in ("[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]")}}
    json.dump(meta, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print(f"{source}: {n_train:,} training and {n_val:,} validation tokens in {time.time() - t0:.0f} s", flush=True)


# ---------------------------------------------------------------------------------------------------------- train


def get_batch(data, a, special, gen, device):
    """Windows of the stream starting with [CLS], BERT's masking. Drawn on the CPU from a seeded generator."""
    L, V = a.seq_len, a.vocab
    starts = torch.randint(0, len(data) - L, (a.batch_size,), generator=gen).tolist()
    ids = torch.from_numpy(np.stack([data[s : s + L - 1] for s in starts]).astype(np.int64))
    ids = torch.cat([torch.full((a.batch_size, 1), special["[CLS]"]), ids], 1)
    maskable = (ids != special["[CLS]"]) & (ids != special["[SEP]"]) & (ids != special["[PAD]"])
    chosen = (torch.rand(ids.shape, generator=gen) < a.mask_prob) & maskable
    labels = torch.where(chosen, ids, torch.full_like(ids, -100))
    r = torch.rand(ids.shape, generator=gen)
    inputs = ids.clone()
    inputs[chosen & (r < 0.8)] = special["[MASK]"]
    rand = chosen & (r >= 0.8) & (r < 0.9)
    inputs[rand] = torch.randint(999, V, (int(rand.sum()),), generator=gen)  # 999+: real word pieces, no [unused]
    return inputs.to(device, non_blocking=True), labels.to(device, non_blocking=True)


@torch.no_grad()
def evaluate(model, data, a, special, device):
    model.eval()
    g = torch.Generator().manual_seed(1234)
    loss, correct, total = 0.0, 0, 0
    for _ in range(a.eval_batches):
        x, y = get_batch(data, a, special, g, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.amp):
            out = model(x, labels=y, labelled_only=True)
        sel = y[y != -100]
        loss += out.loss.item() * len(sel)
        correct += (out.logits.argmax(-1) == sel).sum().item()
        total += len(sel)
    model.train()
    return loss / total, correct / total


EXAMPLES = [
    "The capital of France is [MASK].",
    "Water freezes at zero degrees [MASK].",
    "The [MASK] Ocean is the largest ocean on Earth.",
    "He played the [MASK] in the orchestra for twenty years.",
    "The film was directed by Steven [MASK].",
    "She was born in 1950 and died in [MASK].",
]


@torch.no_grad()
def fill_mask_examples(model, a, device):
    """Top predictions at [MASK] for a few sentences, as a sanity check; also written to examples.json."""
    tok = _tokenizer()
    out = []
    for text in EXAMPLES:
        enc = tok.encode(text)  # with [CLS] ... [SEP]
        ids = torch.tensor([enc.ids], device=device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.amp):
            logits = model(ids).logits[0].float()
        pos = enc.ids.index(tok.token_to_id("[MASK]"))
        top = logits[pos].topk(5).indices.tolist()
        out.append({"text": text, "top5": [tok.id_to_token(i) for i in top]})
        print(f"  {text}  ->  {', '.join(out[-1]['top5'])}", flush=True)
    json.dump(out, open(os.path.join(a.out, "examples.json"), "w"), indent=1)


def model_config(a):
    return dict(d_model=a.width, n_layer=a.layers, vocab_size=a.vocab, mixer="mixed", bank_trees=a.bank_trees,
                bank_heads=a.bank_heads, bank_state=a.bank_state, deep_trees=a.deep_trees, deep_depth=a.deep_depth,
                d_conv=3, causal=False, ternary=True, ternary_group=128, act_bits=8, route_ste=a.route_ste, pad_token_id=0)


def train(a):
    t_start = time.time()
    meta = json.load(open(os.path.join(a.data, "meta.json")))
    a.vocab, special = meta["vocab_size"], meta["special"]
    train_data = np.memmap(os.path.join(a.data, "train.bin"), dtype=np.uint16, mode="r")
    val_data = np.memmap(os.path.join(a.data, "val.bin"), dtype=np.uint16, mode="r")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    a.amp = a.amp and device == "cuda"
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(a.seed)
    cfg = model_config(a)
    model = GTSForMaskedLM(GTSConfig(**cfg)).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"GTS masked LM: {n_params / 1e6:.1f}M parameters; {meta['source']}, {meta['train_tokens']:,} training tokens", flush=True)
    if a.compile and device == "cuda":
        layers = model.backbone.layers
        for i in range(len(layers)):
            layers[i] = torch.compile(layers[i])
    decay = [p for p in model.parameters() if p.ndim >= 2 and not getattr(p, "_no_weight_decay", False)]
    rest = [p for p in model.parameters() if not (p.ndim >= 2 and not getattr(p, "_no_weight_decay", False))]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.weight_decay}, {"params": rest, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.98), eps=1e-6, fused=device == "cuda")
    os.makedirs(a.out, exist_ok=True)
    gen = torch.Generator().manual_seed(a.seed)
    tokens_per_step = a.batch_size * a.seq_len
    total_steps = None  # fitted to --minutes once the step rate is known
    curve, run_loss, run_n, last_ckpt, t_rate = [], 0.0, 0, time.time(), None

    def lr_at(step):
        if step < a.warmup:
            return a.lr * (step + 1) / a.warmup
        if total_steps is None:
            return a.lr
        frac = min(1.0, (step - a.warmup) / max(1, total_steps - a.warmup))
        return a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))

    def save_float(step):
        to_save = {"model": model.state_dict() if not a.compile else {k.replace("._orig_mod", ""): v for k, v in model.state_dict().items()},
                   "optimizer": opt.state_dict(), "step": step, "config": cfg, "args": vars(a), "curve": curve,
                   "generator": gen.get_state()}
        tmp = os.path.join(a.out, "checkpoint.pt.tmp")
        torch.save(to_save, tmp)
        os.replace(tmp, os.path.join(a.out, "checkpoint.pt"))

    def record(step):
        vl, acc = evaluate(model, val_data, a, special, device)
        tl = run_loss / run_n if run_n else float("nan")
        elapsed = time.time() - t_start
        curve.append({"step": step, "tokens": step * tokens_per_step, "val_loss": vl, "val_masked_acc": acc, "train_loss": tl,
                      "minutes": elapsed / 60})
        print(f"step {step:6d}  tokens {step * tokens_per_step / 1e6:8.1f}M  val loss {vl:.4f}  masked acc {acc:.4f}  "
              f"train {tl:.4f}  {elapsed / 60:6.1f} min", flush=True)
        json.dump({"params": n_params, "config": cfg, "args": vars(a), "data": meta, "total_steps": total_steps, "curve": curve},
                  open(os.path.join(a.out, "result.json"), "w"), indent=1)

    step = 0
    model.train()
    while True:
        if step % a.eval_every == 0 and step > 0:
            record(step)
            run_loss, run_n = 0.0, 0
        if total_steps is not None and step >= total_steps:
            break
        for group in opt.param_groups:
            group["lr"] = lr_at(step)
        x, y = get_batch(train_data, a, special, gen, device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.amp):
            loss = model(x, labels=y, labelled_only=True).loss
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        if step % a.log_every == 0:
            run_loss += loss.item()
            run_n += 1
        if step == a.rate_from:
            if device == "cuda":
                torch.cuda.synchronize()
            t_rate = time.time()
        if step == a.rate_from + a.rate_steps:
            if device == "cuda":
                torch.cuda.synchronize()
            rate = a.rate_steps / (time.time() - t_rate)
            left = a.minutes * 60 - (time.time() - t_start) - a.reserve_minutes * 60
            # each evaluation is eval_batches forward passes, about a third of a training step each
            share = (a.eval_batches / 3) / (a.eval_every + a.eval_batches / 3)
            total_steps = step + max(0, int(left * (1 - share) * rate))
            print(f"  {rate:.2f} steps/s = {rate * tokens_per_step:,.0f} tokens/s; schedule fitted to {total_steps} steps "
                  f"({total_steps * tokens_per_step / 1e9:.2f}B tokens)", flush=True)
        if step % a.log_every == 0:
            print(f"  step {step:6d}  loss {loss.item():.4f}  lr {lr_at(step):.2e}  {(time.time() - t_start) / 60:.1f} min", flush=True)
        if time.time() - last_ckpt > a.ckpt_minutes * 60:
            save_float(step)
            last_ckpt = time.time()
            print(f"  checkpoint at step {step}", flush=True)

    record(step)
    save_float(step)
    raw = {k.replace("._orig_mod", ""): v for k, v in model.state_dict().items()}
    plain = GTSForMaskedLM(GTSConfig(**cfg))
    plain.load_state_dict(raw)
    save_binarized(plain, cfg, os.path.join(a.out, "binarized.pt"))
    fill_mask_examples(plain.to(device).eval(), a, device)
    sizes = {f: os.path.getsize(os.path.join(a.out, f)) / 2**20 for f in ("checkpoint.pt", "binarized.pt")}
    print(f"saved checkpoint.pt ({sizes['checkpoint.pt']:.0f} MB) and binarized.pt ({sizes['binarized.pt']:.0f} MB) to {a.out}; "
          f"{(time.time() - t_start) / 60:.1f} min in all", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("prep")
    q.add_argument("--out", required=True)
    q.add_argument("--train-tokens", type=int, default=1_500_000_000)
    q.add_argument("--val-tokens", type=int, default=2_000_000)
    q.add_argument("--text-file")
    q.add_argument("--workers", type=int, default=8, help="shards tokenised in parallel")
    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--minutes", type=float, default=140, help="wall-clock budget for this command, all included")
    t.add_argument("--reserve-minutes", type=float, default=4, help="kept back for the final evaluation and saving")
    t.add_argument("--width", type=int, default=768)
    t.add_argument("--layers", type=int, default=14)
    t.add_argument("--bank-trees", type=int, default=32)
    t.add_argument("--bank-heads", type=int, default=8)
    t.add_argument("--bank-state", type=int, default=16)
    t.add_argument("--deep-trees", type=int, default=4)
    t.add_argument("--deep-depth", type=int, default=9)
    t.add_argument("--no-route-ste", dest="route_ste", action="store_false")
    t.add_argument("--batch-size", type=int, default=64)
    t.add_argument("--seq-len", type=int, default=512)
    t.add_argument("--mask-prob", type=float, default=0.15)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--warmup", type=int, default=1000)
    t.add_argument("--weight-decay", type=float, default=0.01)
    t.add_argument("--eval-every", type=int, default=1000)
    t.add_argument("--eval-batches", type=int, default=20)
    t.add_argument("--log-every", type=int, default=100)
    t.add_argument("--ckpt-minutes", type=float, default=30)
    t.add_argument("--rate-from", type=int, default=100, help="first step of the throughput measurement (after compilation)")
    t.add_argument("--rate-steps", type=int, default=100)
    t.add_argument("--no-amp", dest="amp", action="store_false")
    t.add_argument("--no-compile", dest="compile", action="store_false")
    t.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    prep(a) if a.cmd == "prep" else train(a)


if __name__ == "__main__":
    main()

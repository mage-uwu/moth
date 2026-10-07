# Golden Tree Snake (GTS) fork, 2026.
"""MOHAWK distillation of ModernBERT-large into GTS-L (mamba_ssm/models/gts_l.py), the three stages of Bick et al.
(2024, "Transformers to SSMs: Distilling Quadratic Knowledge to Subquadratic Models") adapted to an encoder:

  Stage 1, matrix orientation: every layer's BiSSD mixing matrix M_l(u) against the teacher's attention matrix A_l(u)
           (softmax probabilities, local window or global as the layer has it), u the teacher's own input to that
           attention, all layers in parallel; relative squared Frobenius distance; only the mixers train.
  Stage 2, hidden-state alignment: every student sub-block against its teacher sub-block on the teacher's input: the
           BiSSD (with its output projection) against the attention output, the deep trees against the GeGLU MLP
           output; relative squared L2; mixers and trees train.
  Stage 3, end-to-end knowledge distillation: the whole student on masked text (30% masking, as ModernBERT), loss
           KL(teacher || student) on the masked positions plus 0.1 x cross-entropy with the true tokens; the
           transferred embeddings and head stay frozen; the ternary quantisation ramps from 0 to 1 over the first
           40% of the stage.
Optimizer as MOHAWK: AdamW (0.9, 0.95), weight decay 0.1, clipping 1.0, warmup-stable-decay (10% / 10%); learning
rates 5e-4 / 2e-3 / 3e-4 here (MOHAWK: 5e-4 / 2e-3 / 2e-4..5e-4). Each stage runs for a wall-clock budget; the log
reports tokens/s and the cost per billion tokens at --price-per-hour.

    python scripts/mohawk_distill.py prep --out /root/fwe
    python scripts/mohawk_distill.py train --data /root/fwe --out /root/run --stage1-minutes 15 --stage2-minutes 35 --stage3-minutes 110
"""
import argparse
import json
import math
import os
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mamba_ssm.models.gts_l import GTSLConfig, GTSLForMaskedLM, config_dict  # noqa: E402

TEACHER = os.environ.get("MOHAWK_TEACHER", "answerdotai/ModernBERT-large")
CLS, SEP, PAD, MASK = 50281, 50282, 50283, 50284


# ----------------------------------------------------------------------------------------------------------- data
def _tok_worker(args):
    path, start, stop = args
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(hf_hub_download(TEACHER, "tokenizer.json"))
    tok.no_padding()
    tok.no_truncation()
    texts = pq.read_table(path, columns=["text"]).slice(start, stop - start).column("text").to_pylist()
    out = []
    for enc in tok.encode_batch(texts, add_special_tokens=False):
        out.extend(enc.ids)
        out.append(SEP)
    return np.asarray(out, dtype=np.uint16)


def prep(a):
    """FineWeb-Edu (ODC-By) in ModernBERT's tokenizer: documents joined with [SEP] into one uint16 stream."""
    from multiprocessing import Pool

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    parts = []
    for name in a.files:
        path = hf_hub_download("HuggingFaceFW/fineweb-edu", name, repo_type="dataset")
        n = pq.ParquetFile(path).metadata.num_rows
        step = math.ceil(n / (a.workers * 4))
        with Pool(a.workers) as pool:
            parts += pool.map(_tok_worker, [(path, s, min(n, s + step)) for s in range(0, n, step)])
        print(f"{name}: {n:,} documents, {sum(p.size for p in parts):,} tokens so far ({time.time() - t0:.0f} s)", flush=True)
    stream = np.concatenate(parts)
    val = stream[-a.val_tokens:]
    train = stream[: -a.val_tokens]
    train.tofile(os.path.join(a.out, "train.bin"))
    val.tofile(os.path.join(a.out, "val.bin"))
    json.dump({"tokenizer": TEACHER, "source": "HuggingFaceFW/fineweb-edu " + " ".join(a.files), "train_tokens": int(train.size),
               "val_tokens": int(val.size)}, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
    print(f"train {train.size:,} tokens, val {val.size:,} ({time.time() - t0:.0f} s)", flush=True)


def batch(data, bsz, seq, gen, device):
    """[CLS] + seq - 1 tokens from random offsets of the stream."""
    starts = torch.randint(0, len(data) - seq, (bsz,), generator=gen).tolist()
    x = np.stack([data[s : s + seq - 1] for s in starts]).astype(np.int64)
    x = torch.from_numpy(np.concatenate([np.full((bsz, 1), CLS), x], 1))
    return x.to(device, non_blocking=True)


def mask_tokens(x, prob, gen):
    """30% of the non-special positions: 80% [MASK], 10% a random token, 10% unchanged. Returns (input, labels)."""
    cand = (x != CLS) & (x != SEP) & (x != PAD)
    r = torch.rand(x.shape, generator=gen, device="cpu").to(x.device)
    sel = cand & (r < prob)
    labels = torch.where(sel, x, torch.full_like(x, -100))
    r2 = torch.rand(x.shape, generator=gen, device="cpu").to(x.device)
    inp = torch.where(sel & (r2 < 0.8), torch.full_like(x, MASK), x)
    rnd = torch.randint(0, 50280, x.shape, generator=gen, device="cpu").to(x.device)
    inp = torch.where(sel & (r2 >= 0.8) & (r2 < 0.9), rnd, inp)
    return inp, labels


# -------------------------------------------------------------------------------------------------------- teacher
class Teacher:
    """ModernBERT-large, frozen, with hooks that record every layer's attention input / output / probabilities and
    MLP input / output."""

    def __init__(self, device, dtype=torch.bfloat16):
        from transformers import AutoModelForMaskedLM

        self.m = AutoModelForMaskedLM.from_pretrained(TEACHER, dtype=dtype, attn_implementation="sdpa").to(device).eval()
        for p in self.m.parameters():
            p.requires_grad_(False)
        self.rec, self.want = {}, set()
        for i, layer in enumerate(self.m.model.layers):
            layer.attn.register_forward_hook(lambda mod, args, kw, out, i=i: self._attn(i, args, kw, out), with_kwargs=True)
            layer.mlp.register_forward_hook(lambda mod, args, out, i=i: self._mlp(i, args, out))

    def _attn(self, i, args, kw, out):
        if "attn" in self.want:
            self.rec[("a_in", i)] = args[0] if args else kw["hidden_states"]
            self.rec[("a_out", i)] = out[0]
        if "probs" in self.want:
            self.rec[("probs", i)] = out[1]

    def _mlp(self, i, args, out):
        if "mlp" in self.want:
            self.rec[("m_in", i)] = args[0]
            self.rec[("m_out", i)] = out

    def set_attn(self, impl):
        try:
            self.m.set_attn_implementation(impl)
        except Exception:
            self.m.config._attn_implementation = impl

    @torch.no_grad()
    def run(self, x, want=(), sel=None):
        self.rec, self.want = {}, set(want)
        h = self.m.model(input_ids=x, attention_mask=torch.ones_like(x)).last_hidden_state
        logits = self.m.decoder(self.m.head(h[sel])) if sel is not None else None
        self.want = set()
        return logits


# ------------------------------------------------------------------------------------------------------- training
def rel(a, b):
    """Relative squared error ||a - b||^2 / ||b||^2, in float32."""
    a, b = a.float(), b.float()
    return (a - b).pow(2).sum() / b.pow(2).sum().clamp(min=1e-12)


def wsd(frac, warm=0.1, decay=0.1):
    """Warmup-stable-decay multiplier at a fraction of the stage's budget."""
    if frac < warm:
        return frac / warm
    if frac > 1 - decay:
        return max(0.0, (1 - frac) / decay)
    return 1.0


@torch.no_grad()
def evaluate(student, teacher, val, a, device):
    g = torch.Generator().manual_seed(1234)
    tot = {"s_ce": 0.0, "t_ce": 0.0, "kl": 0.0, "agree": 0.0, "n": 0}
    for _ in range(a.eval_batches):
        x = batch(val, a.batch_size, a.seq_len, g, device)
        inp, lab = mask_tokens(x, a.mask_prob, g)
        sel = lab != -100
        tl = teacher.run(inp, sel=sel).float()
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            sl = student(inp, sel).float()
        y = lab[sel]
        n = y.numel()
        tot["s_ce"] += F.cross_entropy(sl, y, reduction="sum").item()
        tot["t_ce"] += F.cross_entropy(tl, y, reduction="sum").item()
        tot["kl"] += F.kl_div(F.log_softmax(sl, -1), F.log_softmax(tl, -1), log_target=True, reduction="sum").item()
        tot["agree"] += (sl.argmax(-1) == tl.argmax(-1)).sum().item()
        tot["n"] += n
    n = tot.pop("n")
    return {k: v / n for k, v in tot.items()}


def train(a):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(a.seed)
    tr = np.memmap(os.path.join(a.data, "train.bin"), dtype=np.uint16, mode="r")
    val = np.memmap(os.path.join(a.data, "val.bin"), dtype=np.uint16, mode="r")
    os.makedirs(a.out, exist_ok=True)
    teacher = Teacher(device)
    cfg = GTSLConfig(**json.loads(a.config)) if a.config else GTSLConfig()
    student = GTSLForMaskedLM(cfg).to(device)
    if a.resume:
        student.load_state_dict(torch.load(a.resume, map_location="cpu", weights_only=False)["model"])
        print(f"resumed from {a.resume}", flush=True)
    else:
        student.init_from_modernbert(teacher.m)
    if a.compile and device == "cuda":
        for layer in student.layers:
            layer.deep = torch.compile(layer.deep, dynamic=False)
    n_params = sum(p.numel() for p in student.parameters())
    print(f"teacher {TEACHER}; student GTS-L {n_params / 1e6:.1f}M parameters ({cfg}); data {len(tr):,} training tokens", flush=True)
    log = {"config": config_dict(cfg), "args": vars(a), "stages": {}}
    g = torch.Generator().manual_seed(a.seed)
    amp = dict(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda")
    tok_per_step = a.batch_size * a.seq_len

    def report(name, step, t0, n_tok, parts):
        el = time.time() - t0
        rate = n_tok / max(el, 1e-9)
        usd = a.price_per_hour / (rate * 3600) * 1e9 if rate else float("nan")
        print(f"  [{name}] step {step:6d}  {n_tok / 1e6:8.1f}M tokens  {rate:9,.0f} tokens/s  ${usd:.2f} per 1B tokens  "
              + "  ".join(f"{k} {v:.4f}" for k, v in parts.items()) + f"  {el / 60:.1f} min", flush=True)
        return rate, usd

    def run_stage(name, minutes, params, lr, step_fn, bsz):
        if minutes <= 0:
            return
        opt = torch.optim.AdamW(params, lr=lr, betas=(0.9, 0.95), weight_decay=0.1, fused=device == "cuda")
        t0, step, n_tok, rate, usd = time.time(), 0, 0, 0.0, 0.0
        budget = minutes * 60
        curve = []
        while True:
            frac = (time.time() - t0) / budget
            if frac >= 1:
                break
            for gr in opt.param_groups:
                gr["lr"] = lr * wsd(frac)
            loss, parts = step_fn(frac)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            step += 1
            n_tok += bsz * a.seq_len
            if step % a.log_every == 0 or step == 3:
                rate, usd = report(name, step, t0, n_tok, {k: float(v) for k, v in parts.items()})
                curve.append({"step": step, "tokens": n_tok, "minutes": (time.time() - t0) / 60, **{k: float(v) for k, v in parts.items()}})
        rate, usd = report(name, step, t0, n_tok, {k: float(v) for k, v in parts.items()})
        log["stages"][name] = {"steps": step, "tokens": n_tok, "minutes": (time.time() - t0) / 60, "tokens_per_s": rate,
                               "usd_per_1B_tokens": usd, "curve": curve}
        torch.save({"model": student.state_dict(), "config": config_dict(cfg), "stage": name}, os.path.join(a.out, f"{name}.pt"))
        json.dump(log, open(os.path.join(a.out, "log.json"), "w"), indent=1)

    mixers = [p for layer in student.layers for p in layer.mixer.parameters()]
    trees = [p for layer in student.layers for p in layer.deep.parameters()]

    # Stage 1: matrix orientation (teacher attention probabilities need the eager attention path)
    def stage1(frac):
        x = batch(tr, a.s1_batch, a.seq_len, g, device)
        teacher.run(x, want=("attn", "probs"))
        loss, n = 0.0, 0
        for i, layer in enumerate(student.layers):
            with torch.autocast(**amp):
                M = layer.mixer.matrix(teacher.rec[("a_in", i)].detach().float())
            loss = loss + rel(M, teacher.rec[("probs", i)])
            n += 1
        return loss / n, {"matrix_rel": (loss / n).item()}

    if a.stage1_minutes > 0:
        teacher.set_attn("eager")
        print("== Stage 1: matrix orientation", flush=True)
        run_stage("stage1", a.stage1_minutes, mixers, a.lr1, stage1, a.s1_batch)
        teacher.set_attn("sdpa")
        torch.cuda.empty_cache()

    # Stage 2: hidden-state alignment, every sub-block on the teacher's input
    def stage2(frac):
        x = batch(tr, a.batch_size, a.seq_len, g, device)
        teacher.run(x, want=("attn", "mlp"))
        la = lm = 0.0
        for i, layer in enumerate(student.layers):
            with torch.autocast(**amp):
                ya = layer.mixer(teacher.rec[("a_in", i)].detach().float())
                ym = layer.deep(teacher.rec[("m_in", i)].detach().float())
            la = la + rel(ya, teacher.rec[("a_out", i)])
            lm = lm + rel(ym, teacher.rec[("m_out", i)])
        L = len(student.layers)
        return (la + lm) / L, {"attn_rel": (la / L).item(), "mlp_rel": (lm / L).item()}

    if a.stage2_minutes > 0:
        print("== Stage 2: hidden-state alignment", flush=True)
        run_stage("stage2", a.stage2_minutes, mixers + trees, a.lr2, stage2, a.batch_size)

    ev = evaluate(student, teacher, val, a, device)
    print("  eval before stage 3: " + "  ".join(f"{k} {v:.4f}" for k, v in ev.items()), flush=True)
    log["eval_after_stage2"] = ev

    # Stage 3: end-to-end distillation on masked text, ternary ramped in
    frozen = {id(p) for p in student.transferred_parameters()}
    for p in student.transferred_parameters():
        p.requires_grad_(False)
    body = [p for p in student.parameters() if id(p) not in frozen]
    last_eval = [time.time()]

    def stage3(frac):
        student.set_quant(min(1.0, frac / a.quant_ramp) if a.quant_ramp > 0 else 1.0)
        x = batch(tr, a.batch_size, a.seq_len, g, device)
        inp, lab = mask_tokens(x, a.mask_prob, g)
        sel = lab != -100
        tl = teacher.run(inp, sel=sel).float()
        with torch.autocast(**amp):
            sl = student(inp, sel).float()
        kl = F.kl_div(F.log_softmax(sl, -1), F.log_softmax(tl, -1), log_target=True, reduction="batchmean")
        ce = F.cross_entropy(sl, lab[sel])
        if time.time() - last_eval[0] > a.eval_minutes * 60:
            last_eval[0] = time.time()
            student.eval()
            e = evaluate(student, teacher, val, a, device)
            student.train()
            print(f"  eval (quant {min(1.0, frac / a.quant_ramp):.2f}): " + "  ".join(f"{k} {v:.4f}" for k, v in e.items()), flush=True)
            log.setdefault("evals", []).append({"frac": frac, **e})
        return kl + a.ce_weight * ce, {"kl": kl.item(), "ce": ce.item()}

    if a.stage3_minutes > 0:
        print("== Stage 3: end-to-end distillation", flush=True)
        run_stage("stage3", a.stage3_minutes, body, a.lr3, stage3, a.batch_size)
    student.set_quant(1.0)
    student.eval()
    ev = evaluate(student, teacher, val, a, device)
    print("  final eval (ternary): " + "  ".join(f"{k} {v:.4f}" for k, v in ev.items()), flush=True)
    log["final_eval"] = ev
    json.dump(log, open(os.path.join(a.out, "log.json"), "w"), indent=1)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("prep")
    q.add_argument("--out", required=True)
    q.add_argument("--files", nargs="+", default=["sample/10BT/000_00000.parquet"])
    q.add_argument("--val-tokens", type=int, default=2_000_000)
    q.add_argument("--workers", type=int, default=14)
    t = sub.add_parser("train")
    t.add_argument("--data", required=True)
    t.add_argument("--out", required=True)
    t.add_argument("--config", help="GTSLConfig overrides as JSON (tests use a tiny model)")
    t.add_argument("--resume", help="a stage checkpoint to start from instead of the teacher's weights")
    t.add_argument("--stage1-minutes", type=float, default=15)
    t.add_argument("--stage2-minutes", type=float, default=35)
    t.add_argument("--stage3-minutes", type=float, default=110)
    t.add_argument("--lr1", type=float, default=5e-4)
    t.add_argument("--lr2", type=float, default=2e-3)
    t.add_argument("--lr3", type=float, default=3e-4)
    t.add_argument("--batch-size", type=int, default=32)
    t.add_argument("--s1-batch", type=int, default=8, help="Stage 1 batch (it holds every layer's attention matrices)")
    t.add_argument("--seq-len", type=int, default=512)
    t.add_argument("--mask-prob", type=float, default=0.3)
    t.add_argument("--ce-weight", type=float, default=0.1)
    t.add_argument("--quant-ramp", type=float, default=0.4, help="fraction of Stage 3 over which ternary ramps 0 -> 1")
    t.add_argument("--eval-minutes", type=float, default=20)
    t.add_argument("--eval-batches", type=int, default=10)
    t.add_argument("--log-every", type=int, default=50)
    t.add_argument("--price-per-hour", type=float, default=1.59)
    t.add_argument("--no-compile", dest="compile", action="store_false")
    t.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    prep(a) if a.cmd == "prep" else train(a)


if __name__ == "__main__":
    main()

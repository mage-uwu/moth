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
           transferred embeddings and head stay frozen; a decaying layer-by-layer hidden-state term for the first
           --hidden-frac; full precision until --quant-start, then the ternary quantisation ramps in and is held.
Optimizer as MOHAWK: AdamW (0.9, 0.95), weight decay 0.1 (none on gains, biases and the SSM's scalars), clipping
1.0, updates whose gradient norm spikes past --spike-factor x the recent median skipped, warmup-stable-decay; learning rates
5e-4 / 2e-3 / 2e-4 here (MOHAWK: 5e-4 / 2e-3 / 2e-4..5e-4). Each stage runs for a token budget; the log reports tokens/s
and the cost per billion tokens at --price-per-hour. Downstream check (scripts/downstream_check.py: short SST-2, MNLI
and STS-B fine-tunes of a copy of the student) at the middle of Stage 3 and at the end, and once on the teacher for
reference; ~5 minutes each on an A100, not counted in tokens/s.
Student (EVA_CONFIG): a rank-128 ternary linear path beside every layer's trees, set before Stage 2 from the
least-squares affine fit of the teacher MLP so the trees fit only its nonlinear rest; projections with one ternary
scale per row.

    python scripts/mohawk_distill.py prep --out /root/fwe
    python scripts/mohawk_distill.py train --data /root/fwe --out /root/run
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
    """Tokenise row groups [start, stop) of a parquet file: each worker reads only its own row groups (reading the whole
    file per worker and slicing it ran a 117 GB host out of memory)."""
    path, start, stop = args
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(hf_hub_download(TEACHER, "tokenizer.json"))
    tok.no_padding()
    tok.no_truncation()
    pf = pq.ParquetFile(path)
    texts = pf.read_row_groups(list(range(start, stop)), columns=["text"]).column("text").to_pylist()
    out = []
    for enc in tok.encode_batch(texts, add_special_tokens=False):
        out.extend(enc.ids)
        out.append(SEP)
    return np.asarray(out, dtype=np.uint16)


def prep(a):
    """FineWeb-Edu (ODC-By) in ModernBERT's tokenizer: documents joined with [SEP] into one uint16 stream."""
    import multiprocessing as mp

    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download

    os.makedirs(a.out, exist_ok=True)
    t0 = time.time()
    parts = []
    for name in a.files:
        print(f"downloading {name} ...", flush=True)
        path = hf_hub_download("HuggingFaceFW/fineweb-edu", name, repo_type="dataset")
        meta = pq.ParquetFile(path).metadata
        n, groups = meta.num_rows, meta.num_row_groups
        print(f"  {n:,} documents in {groups} row groups; tokenising with {a.workers} workers ({time.time() - t0:.0f} s)", flush=True)
        step = max(1, math.ceil(groups / (a.workers * 4)))
        with mp.get_context("spawn").Pool(a.workers, maxtasksperchild=4) as pool:  # spawn: no forked tokenizer threads
            parts += pool.map(_tok_worker, [(path, g0, min(groups, g0 + step)) for g0 in range(0, groups, step)], chunksize=1)
        print(f"{name}: {n:,} documents, {sum(p.size for p in parts):,} tokens so far ({time.time() - t0:.0f} s)", flush=True)
    stream = np.concatenate(parts)
    val = stream[-a.val_tokens:]
    train = stream[: -a.val_tokens]
    train.tofile(os.path.join(a.out, "train.bin"))
    val.tofile(os.path.join(a.out, "val.bin"))
    # vocab_size (the model's padded embedding rows), special ids and the random-replacement range, for
    # scripts/bert_pretrain.py train (masked-LM / distillation runs on this data)
    json.dump({"tokenizer": TEACHER, "source": "HuggingFaceFW/fineweb-edu " + " ".join(a.files), "train_tokens": int(train.size),
               "val_tokens": int(val.size), "vocab_size": 50368,
               "special": {"[UNK]": 50280, "[CLS]": CLS, "[SEP]": SEP, "[PAD]": PAD, "[MASK]": MASK},
               "random_range": [1000, 50254]}, open(os.path.join(a.out, "meta.json"), "w"), indent=1)
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
            layer.register_forward_hook(lambda mod, args, out, i=i: self._hid(i, out))
            layer.attn.register_forward_hook(lambda mod, args, kw, out, i=i: self._attn(i, args, kw, out), with_kwargs=True)
            layer.mlp.register_forward_hook(lambda mod, args, out, i=i: self._mlp(i, args, out))

    def _attn(self, i, args, kw, out):
        if "attn" in self.want:
            self.rec[("a_in", i)] = args[0] if args else kw["hidden_states"]
            self.rec[("a_out", i)] = out[0]
        if "probs" in self.want:
            self.rec[("probs", i)] = out[1]

    def _hid(self, i, out):
        if "hid" in self.want:
            self.rec[("hid", i)] = out[0] if isinstance(out, tuple) else out

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
# EVA's student: GTSLConfig plus the linear path beside the trees (least-squares init from the teacher's MLPs before
# Stage 2: Stage 2 tree error 0.277 -> 0.191 in results/lin_ab) and one ternary scale per row for the projections.
# --config overrides any field (e.g. '{"linear_rank": 0, "proj_group": 128}' for pilot 1's student).
EVA_CONFIG = {"linear_rank": 128, "proj_group": 0}


def clean_state(sd):
    """A state_dict without torch.compile's ``_orig_mod.`` wrapper prefix (checkpoints saved before the in-place
    compile carry it)."""
    return {k.replace("_orig_mod.", ""): v for k, v in sd.items()}


class SpikeGuard:
    """Skip an update whose gradient norm is non-finite or above ``factor`` x the median of the last ``window`` norms.
    Every finite norm enters the window, skipped ones too, so a lasting rise (the end of a warmup, say) becomes the new
    normal instead of freezing training; and at most ``max_run`` updates in a row are skipped."""

    def __init__(self, factor, window=50, warm=20, max_run=3):
        self.factor, self.window, self.warm, self.max_run = factor, window, warm, max_run
        self.hist, self.run, self.skipped = [], 0, 0

    def __call__(self, gn):
        if not math.isfinite(gn):
            bad = True
        else:
            med = sorted(self.hist)[len(self.hist) // 2] if len(self.hist) >= self.warm else None
            self.hist = (self.hist + [gn])[-self.window:]
            bad = self.factor > 0 and med is not None and gn > self.factor * med and self.run < self.max_run
        self.run = self.run + 1 if bad else 0
        self.skipped += bad
        return bad


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
        tot["s_ce"] += F.cross_entropy(sl, y, reduction="sum").item()
        tot["t_ce"] += F.cross_entropy(tl, y, reduction="sum").item()
        tot["kl"] += F.kl_div(F.log_softmax(sl, -1), F.log_softmax(tl, -1), log_target=True, reduction="sum").item()
        tot["agree"] += (sl.argmax(-1) == tl.argmax(-1)).sum().item()
        tot["n"] += y.numel()
    n = tot.pop("n")
    return {k: v / n for k, v in tot.items()}


def quant_at(frac, a):
    """Stage 3's ternary blend: 0 (full precision) until --quant-start, a linear ramp over --quant-len, then 1."""
    if a.quant_len <= 0:
        return 1.0 if frac >= a.quant_start else 0.0
    return min(1.0, max(0.0, (frac - a.quant_start) / a.quant_len))


def train(a):
    """The three stages with token budgets, resumable: a state file in --out (weights, optimizer, stage, step,
    tokens, sampler, log) is written every --ckpt-minutes and at every stage end, and a restarted job continues
    from it (a spot VM being reclaimed costs at most --ckpt-minutes)."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.manual_seed(a.seed)
    tr = np.memmap(os.path.join(a.data, "train.bin"), dtype=np.uint16, mode="r")
    val = np.memmap(os.path.join(a.data, "val.bin"), dtype=np.uint16, mode="r")
    os.makedirs(a.out, exist_ok=True)
    state_path = os.path.join(a.out, "state.pt")
    teacher = Teacher(device)
    cfg = GTSLConfig(**{**EVA_CONFIG, **json.loads(a.config or "{}")})
    student = GTSLForMaskedLM(cfg).to(device)
    st = torch.load(state_path, map_location="cpu", weights_only=False) if os.path.exists(state_path) else None
    if st is not None:
        student.load_state_dict(clean_state(st["model"]))
        print(f"resuming: stage {st['stage']}, step {st['step']}, {st['tokens'] / 1e6:.1f}M tokens into it; done {st['done']}", flush=True)
    elif a.init:
        student.load_state_dict(clean_state(torch.load(a.init, map_location="cpu", weights_only=False)["model"]))
        print(f"weights from {a.init}", flush=True)
    else:
        student.init_from_modernbert(teacher.m)
    deep_mods = [layer.deep for layer in student.layers]
    if a.compile and device == "cuda":
        for layer in student.layers:
            layer.deep.compile(dynamic=False)  # in place: state_dict keys stay unprefixed
    n_params = sum(p.numel() for p in student.parameters())
    print(f"teacher {TEACHER}; student GTS-L {n_params / 1e6:.1f}M parameters ({cfg}); data {len(tr):,} training tokens; "
          f"budgets {a.stage1_tokens / 1e6:.0f}M / {a.stage2_tokens / 1e6:.0f}M / {a.stage3_tokens / 1e6:.0f}M tokens", flush=True)
    log = st["log"] if st else {"config": config_dict(cfg), "args": vars(a), "stages": {}}
    done = st["done"] if st else []
    g = torch.Generator().manual_seed(a.seed)
    if st:
        g.set_state(st["gen"])
    amp = dict(device_type="cuda", dtype=torch.bfloat16, enabled=device == "cuda")
    publish = a.publish_cmd

    def save_state(stage, step, n_tok, opt, extra=None):
        tmp = state_path + ".tmp"
        torch.save({"model": student.state_dict(), "opt": opt.state_dict() if opt else None, "stage": stage, "step": step,
                    "tokens": n_tok, "gen": g.get_state(), "done": list(done), "log": log, **(extra or {})}, tmp)
        os.replace(tmp, state_path)
        json.dump(log, open(os.path.join(a.out, "log.json"), "w"), indent=1)
        if publish:
            os.system(publish)

    paused = [0.0]  # seconds of the current stage spent in downstream checks, left out of its tokens/s

    def downstream(point):
        """The downstream check on a copy of the student (and the teacher once), into log["downstream"][point]."""
        ds = log.setdefault("downstream", {})
        if not a.downstream or point in ds:
            return
        t = time.time()
        try:
            from downstream_check import student_check, summary, teacher_check

            with torch.random.fork_rng(devices=[torch.cuda.current_device()] if device == "cuda" else []):
                if a.downstream_teacher and "teacher" not in ds:
                    ds["teacher"] = teacher_check(TEACHER, device, a.downstream_teacher_lr)
                    print(f"  downstream, teacher: {summary(ds['teacher'])}", flush=True)
                ds[point] = student_check(student, device, a.downstream_lr)
                print(f"  downstream, student {point} (quant {ds[point]['quant']:.2f}): {summary(ds[point])}", flush=True)
        except Exception as e:  # a failed check (no network, say) must not stop the run
            print(f"  downstream check {point} failed: {e!r}", flush=True)
            ds[point] = {"error": repr(e)}
        if device == "cuda":
            torch.cuda.empty_cache()
        paused[0] += time.time() - t

    def report(name, step, t0, n_tok0, n_tok, budget, parts):
        el = time.time() - t0 - paused[0]
        rate = (n_tok - n_tok0) / max(el, 1e-9)
        usd = a.price_per_hour / (rate * 3600) * 1e9 if rate else float("nan")
        left = (budget - n_tok) / max(rate, 1e-9) / 3600
        print(f"  [{name}] step {step:6d}  {n_tok / 1e6:8.1f}M / {budget / 1e6:.0f}M tokens  {rate:9,.0f} tokens/s  "
              f"${usd:.2f} per 1B tokens  " + "  ".join(f"{k} {v:.4f}" for k, v in parts.items())
              + f"  {el / 60:.1f} min, ~{left:.1f} h left in stage", flush=True)
        return rate, usd

    def run_stage(name, budget, params, lr, step_fn, bsz, warm, decay):
        if budget <= 0 or name in done:
            return
        # no weight decay on gains, biases and the SSM's scalars (A_log, D, dt bias), as in Mamba and MOHAWK
        groups = [{"params": [p for p in params if p.ndim >= 2]},
                  {"params": [p for p in params if p.ndim < 2], "weight_decay": 0.0}]
        opt = torch.optim.AdamW(groups, lr=lr, betas=(0.9, 0.95), weight_decay=0.1, fused=device == "cuda")
        step, n_tok = 0, 0
        if st is not None and st["stage"] == name:
            step, n_tok = st["step"], st["tokens"]
            if st.get("opt") is not None:
                opt.load_state_dict(st["opt"])
        print(f"== {name}: {budget / 1e6:.0f}M tokens, from {n_tok / 1e6:.1f}M", flush=True)
        t0, n_tok0, last_ck = time.time(), n_tok, time.time()
        paused[0] = 0.0
        curve = log["stages"].setdefault(name, {}).setdefault("curve", [])
        parts, rate, usd = {}, 0.0, 0.0
        guard = SpikeGuard(a.spike_factor)
        while n_tok < budget:
            frac = n_tok / budget
            for gr in opt.param_groups:
                gr["lr"] = lr * wsd(frac, warm, decay)
            loss, parts = step_fn(frac)
            opt.zero_grad(set_to_none=True)
            loss.backward()
            gn = torch.nn.utils.clip_grad_norm_(params, 1.0).item()
            if guard(gn):  # a spike: drop this update rather than let it knock the student off course
                print(f"  [{name}] step {step}: skipped update, gradient norm {gn:.3g}", flush=True)
            else:
                opt.step()
            parts = {**parts, "gnorm": gn, "skipped": guard.skipped}
            step += 1
            n_tok += bsz * a.seq_len
            if step % a.log_every == 0 or step == 3:
                rate, usd = report(name, step, t0, n_tok0, n_tok, budget, parts)
                curve.append({"step": step, "tokens": n_tok, **{k: float(v) for k, v in parts.items()}})
            if time.time() - last_ck > a.ckpt_minutes * 60:
                last_ck = time.time()
                save_state(name, step, n_tok, opt)
        rate, usd = report(name, step, t0, n_tok0, n_tok, budget, parts)
        log["stages"][name].update({"steps": step, "tokens": n_tok, "tokens_per_s": rate, "usd_per_1B_tokens": usd})
        done.append(name)
        torch.save({"model": student.state_dict(), "config": config_dict(cfg), "stage": name}, os.path.join(a.out, f"{name}.pt"))
        save_state(name, step, n_tok, None)

    mixers = [p for layer in student.layers for p in layer.mixer.parameters()]
    trees = [p for layer in student.layers for p in layer.ffn_parameters()]  # the trees (and linear paths)

    # Stage 1: matrix orientation (the teacher's attention probabilities need its eager attention path)
    def stage1(frac):
        x = batch(tr, a.s1_batch, a.seq_len, g, device)
        teacher.run(x, want=("attn", "probs"))
        loss = 0.0
        for i, layer in enumerate(student.layers):
            with torch.autocast(**amp):
                M = layer.mixer.matrix(teacher.rec[("a_in", i)].detach().float())
            loss = loss + rel(M, teacher.rec[("probs", i)])
        loss = loss / len(student.layers)
        return loss, {"matrix_rel": loss.item()}

    if a.stage1_tokens > 0 and "stage1" not in done:
        teacher.set_attn("eager")
        run_stage("stage1", a.stage1_tokens, mixers, a.lr1, stage1, a.s1_batch, 0.1, 0.1)
        teacher.set_attn("sdpa")
        torch.cuda.empty_cache()

    # Stage 2: hidden-state alignment, every sub-block on the teacher's input (the trees' term weighted --mlp-weight)
    def stage2(frac):
        x = batch(tr, a.batch_size, a.seq_len, g, device)
        teacher.run(x, want=("attn", "mlp"))
        la = lm = 0.0
        for i, layer in enumerate(student.layers):
            with torch.autocast(**amp):
                ya = layer.mixer(teacher.rec[("a_in", i)].detach().float())
                ym = layer.ffn(teacher.rec[("m_in", i)].detach().float())
            la = la + rel(ya, teacher.rec[("a_out", i)])
            lm = lm + rel(ym, teacher.rec[("m_out", i)])
        L = len(student.layers)
        return (la + a.mlp_weight * lm) / L, {"attn_rel": (la / L).item(), "mlp_rel": (lm / L).item()}

    if cfg.linear_rank and "lin_init" not in done and a.stage2_tokens > 0:
        # the linear paths from the least-squares affine fit of each teacher MLP, before Stage 2 fits the trees
        t = time.time()
        L = len(student.layers)
        xtx = torch.zeros(L, cfg.d_model + 1, cfg.d_model + 1, dtype=torch.float64, device=device)
        xty = torch.zeros(L, cfg.d_model + 1, cfg.d_model, dtype=torch.float64, device=device)
        with torch.no_grad():
            for _ in range(a.lin_init_batches):
                teacher.run(batch(tr, a.batch_size, a.seq_len, g, device), want=("mlp",))
                for i in range(L):
                    x = teacher.rec[("m_in", i)].reshape(-1, cfg.d_model).double()
                    x = torch.cat([x, torch.ones(len(x), 1, dtype=x.dtype, device=device)], 1)
                    xtx[i] += x.T @ x
                    xty[i] += x.T @ teacher.rec[("m_out", i)].reshape(-1, cfg.d_model).double()
            for i, layer in enumerate(student.layers):
                layer.init_linear(xtx[i], xty[i])
        del xtx, xty
        done.append("lin_init")
        print(f"  linear paths (rank {cfg.linear_rank}) from the teacher's affine fits, "
              f"{a.lin_init_batches * a.batch_size * a.seq_len / 1e6:.1f}M tokens ({time.time() - t:.0f} s)", flush=True)

    run_stage("stage2", a.stage2_tokens, mixers + trees, a.lr2, stage2, a.batch_size, 0.1, 0.1)

    if "stage2_layers" not in log and a.stage2_tokens > 0:  # every layer's Stage 2 errors on held-out text
        sums = torch.zeros(len(student.layers), 4, device=device)
        g_val = torch.Generator().manual_seed(1)
        with torch.no_grad():
            for _ in range(a.eval_batches):
                teacher.run(batch(val, a.batch_size, a.seq_len, g_val, device), want=("attn", "mlp"))
                for i, layer in enumerate(student.layers):
                    with torch.autocast(**amp):
                        ya = layer.mixer(teacher.rec[("a_in", i)].float()).float()
                        ym = layer.ffn(teacher.rec[("m_in", i)].float()).float()
                    ta, tm = teacher.rec[("a_out", i)].float(), teacher.rec[("m_out", i)].float()
                    sums[i] += torch.stack([(ya - ta).pow(2).sum(), ta.pow(2).sum(), (ym - tm).pow(2).sum(), tm.pow(2).sum()])
        per = [{"layer": i, "attn_rel": (s[0] / s[1]).item(), "mlp_rel": (s[2] / s[3]).item()} for i, s in enumerate(sums)]
        log["stage2_layers"] = per
        print("  stage 2, per layer (held out): " + "  ".join(f"{p['layer']}:{p['mlp_rel']:.3f}" for p in per), flush=True)
        print(f"  stage 2, held out: attn_rel {sum(p['attn_rel'] for p in per) / len(per):.4f}  "
              f"mlp_rel {sum(p['mlp_rel'] for p in per) / len(per):.4f}", flush=True)

    if "eval_after_stage2" not in log:
        ev = evaluate(student, teacher, val, a, device)
        print("  eval before stage 3: " + "  ".join(f"{k} {v:.4f}" for k, v in ev.items()), flush=True)
        log["eval_after_stage2"] = ev

    # Stage 3: end-to-end distillation on masked text, in full precision until --quant-start, then ternary ramped in;
    # for its first --hidden-frac, a decaying layer-by-layer hidden-state term pulls the chained student back on track
    frozen = {id(p) for p in student.transferred_parameters()}
    for p in student.transferred_parameters():
        p.requires_grad_(False)
    body = [p for p in student.parameters() if id(p) not in frozen]
    last_eval = [time.time()]

    def stage3(frac):
        if frac >= 0.5 and "mid" not in log.get("downstream", {}):
            downstream("mid")
        lam = quant_at(frac, a)
        student.set_quant(lam)
        x = batch(tr, a.batch_size, a.seq_len, g, device)
        inp, lab = mask_tokens(x, a.mask_prob, g)
        sel = lab != -100
        hw = a.hidden_weight * max(0.0, 1.0 - frac / a.hidden_frac) if a.hidden_frac > 0 else 0.0
        tl = teacher.run(inp, sel=sel, want=("hid",) if hw > 0 else ()).float()
        with torch.autocast(**amp):
            if hw > 0:
                hs, hfin = student.hidden(inp, all_layers=True)
                sl = student.logits_at(hfin[sel]).float()
            else:
                sl = student(inp, sel).float()
        kl = F.kl_div(F.log_softmax(sl, -1), F.log_softmax(tl, -1), log_target=True, reduction="batchmean")
        ce = F.cross_entropy(sl, lab[sel])
        loss = kl + a.ce_weight * ce
        parts = {"kl": kl.item(), "ce": ce.item(), "quant": lam}
        if hw > 0:
            hl = sum(rel(hs[i], teacher.rec[("hid", i)]) for i in range(len(student.layers))) / len(student.layers)
            loss = loss + hw * hl
            parts["hid_rel"] = hl.item()
        if time.time() - last_eval[0] > a.eval_minutes * 60:
            last_eval[0] = time.time()
            student.eval()
            e = evaluate(student, teacher, val, a, device)
            student.train()
            print(f"  eval (quant {lam:.2f}): " + "  ".join(f"{k} {v:.4f}" for k, v in e.items()), flush=True)
            log.setdefault("evals", []).append({"frac": frac, "quant": lam, **e})
        return loss, parts

    run_stage("stage3", a.stage3_tokens, body, a.lr3, stage3, a.batch_size, a.warm3, a.decay3)
    student.set_quant(1.0)
    student.eval()
    ev = evaluate(student, teacher, val, a, device)
    print("  final eval (ternary): " + "  ".join(f"{k} {v:.4f}" for k, v in ev.items()), flush=True)
    log["final_eval"] = ev
    downstream("final")
    torch.save({"model": student.state_dict(), "config": config_dict(cfg), "stage": "final"}, os.path.join(a.out, "final.pt"))
    json.dump(log, open(os.path.join(a.out, "log.json"), "w"), indent=1)
    if publish:
        os.system(publish)


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
    t.add_argument("--config", help="GTSLConfig fields as JSON, over EVA_CONFIG (tests use a tiny model)")
    t.add_argument("--init", help="a stage checkpoint (.pt with 'model') to start from instead of the teacher's weights")
    t.add_argument("--stage1-tokens", type=float, default=80e6)
    t.add_argument("--stage2-tokens", type=float, default=300e6)
    t.add_argument("--stage3-tokens", type=float, default=2.62e9)
    t.add_argument("--lr1", type=float, default=5e-4)
    t.add_argument("--lr2", type=float, default=2e-3)
    t.add_argument("--lr3", type=float, default=2e-4)
    t.add_argument("--spike-factor", type=float, default=4.0,
                   help="skip an update whose gradient norm exceeds this x the median of the last 50 norms, at most 3 "
                        "in a row (0: off)")
    t.add_argument("--warm3", type=float, default=0.05, help="Stage 3 warmup fraction")
    t.add_argument("--decay3", type=float, default=0.07, help="Stage 3 final decay fraction (the ternary hold)")
    t.add_argument("--mlp-weight", type=float, default=2.0, help="Stage 2: weight of the trees' alignment term")
    t.add_argument("--quant-start", type=float, default=0.86, help="Stage 3 fraction where the ternary ramp starts")
    t.add_argument("--quant-len", type=float, default=0.07, help="Stage 3 fraction the ramp takes (then held at 1)")
    t.add_argument("--hidden-weight", type=float, default=1.0, help="Stage 3: initial weight of the layer-by-layer term")
    t.add_argument("--hidden-frac", type=float, default=0.05, help="Stage 3 fraction over which that term decays to 0")
    t.add_argument("--lin-init-batches", type=int, default=16, help="batches for the linear paths' least-squares fit")
    t.add_argument("--batch-size", type=int, default=32)
    t.add_argument("--s1-batch", type=int, default=8, help="Stage 1 batch (it holds every layer's attention matrices)")
    t.add_argument("--seq-len", type=int, default=512)
    t.add_argument("--mask-prob", type=float, default=0.3)
    t.add_argument("--ce-weight", type=float, default=0.1)
    t.add_argument("--eval-minutes", type=float, default=30)
    t.add_argument("--eval-batches", type=int, default=10)
    t.add_argument("--log-every", type=int, default=100)
    t.add_argument("--ckpt-minutes", type=float, default=30)
    t.add_argument("--publish-cmd", help="shell command run after every checkpoint (e.g. copy log.json somewhere readable)")
    t.add_argument("--price-per-hour", type=float, default=1.59)
    t.add_argument("--no-downstream", dest="downstream", action="store_false", help="skip the downstream checks")
    t.add_argument("--no-downstream-teacher", dest="downstream_teacher", action="store_false",
                   help="skip the teacher's reference check")
    t.add_argument("--downstream-lr", type=float, default=5e-5)
    t.add_argument("--downstream-teacher-lr", type=float, default=2e-5)
    t.add_argument("--no-compile", dest="compile", action="store_false")
    t.add_argument("--seed", type=int, default=0)
    a = p.parse_args()
    prep(a) if a.cmd == "prep" else train(a)


if __name__ == "__main__":
    main()

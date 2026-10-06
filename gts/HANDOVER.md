# Handover: Golden Tree Snake (GTS)

You are picking up a research codebase from a previous session. Read this file first, then `GTS.md`,
which is the lab notebook with every result and design decision. The owner has a GPU and wants to do
a small-scale run next. Everything so far was developed and measured on one CPU core.

## What this project is

GTS is a sequence mixer that crosses two ideas: Mamba-2's state space layer and the conditional
execution of fast feedforward networks (FFF). Mamba-2's dense inner channels become nodes of binary
trees. A token walks root-to-leaf paths and only the nodes on its paths are computed, written to and
read from; a node's state is decayed lazily when it is next visited. Weights are ternary (BitNet
b1.58 style, grouped absmean with a straight-through estimator). The goal is a language model that is
much cheaper per token on a CPU than Mamba-2 at similar quality.

This repository is a fork of `state-spaces/mamba` at upstream commit `e9594ce`. Upstream files are
untouched except `mamba_ssm/__init__.py`, which guards the GPU-kernel imports so the GTS code imports
anywhere. The GTS code needs only `torch` and `numpy`; do not install or build upstream Mamba unless
you need its fused kernels as an extra baseline.

## First task: a quick 0.5B run on FineWeb, both models

Do this before anything else. The owner's question is whether the speed gap is real or too good to be
true. So far the only evidence at this size is a timing of random weights (0.18 to 0.44 ms per token
for the mixed forest against 31 to 46 ms for Mamba-2, layer stack only). This task replaces that with
trained weights, checked against PyTorch, for both models.

`scripts/lm_run.py` and `scripts/prepare_fineweb.py` were written for this and smoke-tested on a CPU at
toy size only. They have never run at the default size, on a GPU, or against the real FineWeb stream.
Expect to fix things; keep the two runs identical in everything but `--arch`.

```bash
# 0. Environment. The packed and integer kernel paths need AVX-512. If the count is 0, stop and tell
#    the owner: both models would fall back to the float path and the comparison is a different one.
python -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
grep -c avx512f /proc/cpuinfo

# 1. Tests, then tokens (about 8 million are read by a 2,000-step run at batch 8 x 512).
python -m pytest tests/modules/test_gts.py tests/modules/test_gts_ternary.py -q
pip install datasets tiktoken
python scripts/prepare_fineweb.py --out data/fineweb --train-tokens 20000000 --val-tokens 500000

# 2. Shake out device bugs at small size first: a minute or two each.
python scripts/lm_run.py --arch mixed  --data data/fineweb --out runs/smoke_mixed  --width 256 --layers 4 --deep-depth 8 --steps 50 --eval-every 25
python scripts/lm_run.py --arch mamba2 --data data/fineweb --out runs/smoke_mamba2 --width 256 --m2-layers 6 --steps 50 --eval-every 25

# 3. The run: 507.6M and 500.4M parameters at the defaults, 2,000 steps each.
python scripts/lm_run.py --arch mixed  --data data/fineweb --out runs/fw_mixed
python scripts/lm_run.py --arch mamba2 --data data/fineweb --out runs/fw_mamba2

# 4. Kernel: check against PyTorch, then time. Keep the token count small; the output head is slow.
gcc -O3 -march=native -ffast-math -funroll-loops kernel/ar_bench.c -o kernel/ar_bench -lm
kernel/ar_bench runs/fw_mixed/model.bin 2000
kernel/ar_bench runs/fw_mamba2/model.bin 500
```

If a run does not fit in GPU memory, halve `--batch-size` and double `--grad-accum`, then shorten
`--seq-len`, and apply the same change to both models. Gradient checkpointing is already on. My
estimate is 10 to 15 GB at the defaults; that is arithmetic, not a measurement. The learning rate
(1e-3) is a guess. `model.bin` is about 2 GB and the kernel needs about 2.5 GB of RAM to load it.

Report one table: validation loss at each evaluation, training tokens per second, peak GPU memory, the
kernel's check line, and the kernel's time per token both for the mixers alone ("of which mixers") and
in total.

Checks that decide whether the speed is real:

1. **The kernel must be timing the right function.** Its check line has to show the same top-1 as
   PyTorch on nearly all 256 test tokens and a matching loss, for both models, on the trained weights.
   With 8-bit activations a few differing logits are known and unexplained; a different loss is a bug.
2. **Trained against random.** Compare with the random-weight timings above. A trained tree may route
   more or less cache-friendly than a random one. More than a factor of two either way needs explaining.
3. **The Mamba-2 kernel is the weak link in the ratio.** It is a simple packed kernel that pays one
   horizontal sum per 128 weights, not a tuned one. Before quoting a ratio, bound it independently:
   run the same model with `kernel/ar_bench ... float` and time one layer's projections with
   single-threaded BLAS in PyTorch. If either beats the packed path, the packed Mamba-2 number is
   unfair and the kernel should be improved first.
4. **The output head.** Scoring 50,257 tokens at width 1024 costs far more than the mixed forest's whole
   stack. Report the total as well as the mixers-only time, and say which one a claim rests on.
5. **Quality.** 2,000 steps says little about quality. Confirm both models learn (loss well below
   ln 50257 = 10.8 and falling) and report the gap at equal steps without reading much into it. At
   small scale the mixed forest trailed Mamba-2 by 0.10 to 0.25 nats and width did not close it.

## Where things stand

Character-level TinyShakespeare, autoregressive, width 128, about 0.56M parameters, ternary weights,
batch 16 x 128 characters, one seed each. Validation loss in nats per character. Times are from a C
kernel on one CPU core and vary by session; only compare numbers measured back to back.

| Model | Steps, learning rate | Validation loss | Time per token |
|---|---|---|---|
| Original GTS (4 layers, 1 tree of depth 8) | 1,200, 2e-3 | 2.00 | about 5 us |
| Original GTS + routing gradient | 1,200, 2e-3 | 1.95 | about 5 us |
| Wide forest (6 layers, 21 trees of depth 3) | 1,200, 2e-3 | 1.78 | about 21 us |
| Mixed forest | 1,200, 2e-3 | 1.84 | about 8 us |
| Mixed forest + routing gradient | 1,200, 2e-3 | 1.81 | about 8 us |
| **Mixed forest + routing gradient** | **1,200, 4e-3** | **1.73** | **about 8 us** |
| Mamba-2 (5 layers, expand 2) | 1,200, 2e-3 | 1.58 | about 55 us |
| Mixed forest + routing gradient | 3,600, 4e-3 | 1.59 | about 7 us |
| Mamba-2 | 3,600, 4e-3 | 1.49 | about 48 us |

The bold row is the current baseline, which the owner calls the beachhead. At equal training the
mixed forest is about 0.10 nats behind Mamba-2 and about 7 times faster per token. The per-run JSON
files and text samples are in `results/`.

The mixed forest is two GTS mixers summed in each block: a bank of 32 depth-0 trees that every token
visits and that carries all the context, plus 4 trees of depth 6 with no state that hold most of the
parameters. It uses 8-bit activations so the kernel can use integer sums.

## Added after the first handover

- **`GTSMixed`** in `mamba_ssm/modules/gts.py`: the mixed forest as a library module, bidirectional by
  default, selectable in the encoder with `GTSConfig(mixer="mixed", ...)`. Tested against the reference.
- **`scripts/width_test.py`**: speed and quality of the bidirectional mixed forest against a
  bidirectional Mamba-2 at widths 128 and 256 (and speed at 512). The speed ratio doubled with each
  doubling of width (6x, 12x, 24x); the quality gap did not close (0.25 nats at both widths after 600
  steps). Numbers are in `GTS.md` and `results/width_test/`.
- **`kernel/ar_bench synth`**: random packed ternary models built in C, to time sizes PyTorch cannot
  hold on a small machine. At about 0.45B mixer parameters and width 1024, one token through the stack
  took 0.18 to 0.44 ms for the mixed forest and 31 to 46 ms for Mamba-2. Speed only: nothing at that
  size is trained, and the Mamba-2 kernel is not a tuned one.

## Commands

```bash
pip install -r requirements-gts.txt

# 22 CPU tests, about 10 seconds. Run these before and after every change.
python -m pytest tests/modules/test_gts.py tests/modules/test_gts_ternary.py -q

# The baseline. Writes result.json, model.pt, model.bin and sample.txt to --out.
python scripts/shakespeare_ar.py --data data/tinyshakespeare.txt --arch mixed \
  --gts-layers 4 --gts-depth 6 --gts-trees 4 --bank-trees 32 --bank-heads 8 --bank-state 16 \
  --gts-act-bits 8 --gts-route-ste --lr 4e-3 --steps 1200 --eval-every 300 --out runs/mixed

# The Mamba-2 reference under the same wrapper, optimiser, batches and quantiser.
python scripts/shakespeare_ar.py --data data/tinyshakespeare.txt --arch mamba2 \
  --lr 2e-3 --steps 1200 --eval-every 300 --out runs/mamba2

# Inference kernel: checks its logits against PyTorch's, then times token-at-a-time inference.
gcc -O3 -march=native -ffast-math -funroll-loops kernel/ar_bench.c -o kernel/ar_bench -lm
kernel/ar_bench runs/mixed/model.bin 100000
kernel/ar_bench runs/mixed/model.bin 100000 float   # the same weights run as float32
```

`--device` defaults to `cuda` when available. Batches are drawn on the CPU from a seeded generator,
so a given seed sees the same batches on any device.

## File map

| Path | What it holds |
|---|---|
| `GTS.md` | The notebook: design, every experiment, what worked and what did not. |
| `mamba_ssm/modules/gts.py` | The mixer. `forward` is the differentiable training path; `forward_reference` is the token-at-a-time lazy-decay algorithm an inference kernel implements. |
| `mamba_ssm/modules/ternary.py` | Ternary weight and integer activation quantisers, pack and unpack for export. |
| `mamba_ssm/utils/gts_qat.py` | Ternary student from a full-precision teacher, distillation, route distillation. |
| `mamba_ssm/models/gts_encoder.py` | A BERT-style encoder and masked-LM head. It exposes only the early mixer options. |
| `scripts/shakespeare_ar.py` | The autoregressive experiments: `TinyLM` wrapper, `Mamba2Ref` (plain-PyTorch Mamba-2), all model variants, export. |
| `scripts/lm_run.py`, `scripts/prepare_fineweb.py` | The at-scale run: GPT-2 tokens of FineWeb, both models at about 0.5B parameters. Untested at that size. |
| `scripts/width_test.py` | Bidirectional mixed forest against a bidirectional Mamba-2 across widths. |
| `train_gts.py`, `scripts/gts_ternary_demo.py` | Masked-LM training and a toy ternary comparison. Toy scale only. |
| `kernel/ar_bench.c` | C inference for every exported model: float, packed ternary, and integer paths. |
| `mamba_ssm/ops/gts_scan.py` | Chunked Triton scan for depth-0 trees (the mixed forest's bank); `GTS` uses it on CUDA. |
| `scripts/bench_scan.py`, `scripts/diag_scan.py`, `scripts/f64_check.py` | Scan timing and accuracy; float64 check of a trained model's exported logits. |
| `pod/job.sh`, `pod/bench_job.sh` | What the RunPod pods ran: the 0.5B runs, and the scan tests and benchmarks. |
| `results/fineweb_0.5b/`, `results/scan_bench_a100.log` | The 0.5B results and the scan benchmark. |
| `kernel/gts_kernel.c` | The first, bidirectional float kernel. Superseded by `ar_bench.c`; kept for the width-768 timing in `GTS.md`. |
| `checkpoints/bert110m/`, `scripts/bert_pretrain.py`, `kernel/enc_bench.c` | The 110M BERT-style GTS masked LM: checkpoints by phase (README there), its training script (`--resume`, `--teacher`), and its multithreaded CPU inference kernel. |
| `vision/` | Side quest: a ~38M ternary GTS vision backbone and a sidecar into GTS-MLM. `vision/README.md` has the status, data, licensing and next steps; the model is `mamba_ssm/models/gts_vision.py`. |
| `tests/modules/test_gts*.py` | The tests. |
| `results/` | Result JSON and samples for the runs in the table above. |
| `patches/` | Next to this repo in the archive: the same commits as patches, for applying to a full upstream clone. |

## Rules that keep this codebase honest

1. **The two forward paths must agree.** `test_dense_forward_matches_lazy_kernel` runs every mixer
   configuration through `forward` and `forward_reference` and requires equal output. Any new mixer
   option goes into that test's configuration list in the same change.
2. **The C kernel must reproduce PyTorch.** A new option that changes the forward pass also needs an
   export field in `shakespeare_ar.py` and support in `ar_bench.c`. The kernel prints its own check.
3. **Compare like with like.** Same data split, batches, steps and schedule. When one model gets a
   tuned setting the other did not, say so next to the number.
4. **Report what was measured.** Every result so far is one seed. Say when a number is an estimate,
   an operation count, or from a different session.
5. **Pure GTS.** The owner rejected a hybrid with a Mamba-2 block beside the tree (`--arch hybrid`
   still exists in the script as a record). Improvements should stay within tree nodes.

## Known issues and untested ground

- **Scale.** Both 0.5B models have now trained for 2,000 steps on FineWeb on one A100 (results in `GTS.md`).
  Nothing longer or larger has run.
- **GPU.** Runs on CUDA (A100, torch 2.8). Set OMP_NUM_THREADS on pods: PyTorch sized its threads from the
  host and the CPU tests crawled. Mixed precision is still untried.
- **Memory and time of the training path.** Depth-0 trees now use the Triton scan on CUDA (linear in length).
  Deeper trees with context, `Mamba2Ref`, and everything on a CPU are still quadratic in sequence length.
- **Large trees.** The fast training walk (`dense_walk`) is on by default only up to 2,048 nodes per
  mixer, and `route_ste` requires it. `GTSMixed` and the scripts' mixed forest force it on for the deep
  trees, which costs memory proportional to (tokens x nodes) per layer; a bare `GTS` with more nodes
  needs `dense_walk=True` passed explicitly.
- **Kernel with 8-bit activations.** The loss on the test tokens matches PyTorch to three decimals and
  top-1 agrees on all of them, but individual logits differ on a few tokens, by as much as 2.8. The
  suspected cause is a rounding or branch decision flipping on tiny float differences. This is not
  proven. Running the PyTorch model in float64 and comparing all three would settle it.
- **AVX-512.** The packed and integer kernel paths need AVX-512 (plus BW; VNNI is used when present).
  Without it the kernel still builds and runs on the float path, about 1.5 times slower.
- **Parameter count.** The mixed forest has 588,785 parameters, about 6% more than Mamba-2's 556,153.
- **`Mamba2Ref`.** It is my reimplementation of Mamba-2's non-fused path with the scan in quadratic
  form. Its C step reproduces it exactly, but it has never been checked against upstream's `Mamba2`.
- **Kernel export formats.** `model.bin` uses ad hoc format codes 1 to 6, documented only in the code.
- **Git history.** `.git` is a shallow clone of upstream plus the GTS commits. Some hosts refuse
  pushes from a shallow clone; if that happens, clone upstream fully and apply `patches/` with `git am`.

## After the first task

1. Reproduce the two 1,200-step TinyShakespeare runs above on the GPU. Expect about 1.73 for the
   mixed forest and about 1.58 for Mamba-2 (which has never been run for 1,200 steps at 4e-3).
   Differences of a few hundredths from float ordering are normal; a large difference means the GPU
   path has a bug.
2. Add seeds. Three per headline model would show how much of the 0.10 gap is noise.
3. Test the central untested claim: that the tree's advantage grows with width. At width 128 a dense
   block is already cheap. Try width 256 to 512 on a larger corpus of the owner's choosing, against
   Mamba-2 both at equal parameters and at equal kernel time per token.
4. Run the equal-speed control that is still missing: a Mamba-2 shrunk until its kernel time matches
   the mixed forest's.

## Ideas not yet tried

- A separate read path: the token walks a tree a second time with query vectors and reads from where
  that walk lands, so it can fetch from tokens unlike itself. Viable now that routing has a gradient.
- Several decay rates inside each node's state.
- Distillation from the trained Mamba-2 (`gts_qat.py` has the pieces, written for masked LM).
- Deriving keys and queries from the walk's own logits, removing the key projection.
- A temperature sweep or schedule for `route_ste`.

## Tried and set aside

- History-dependent routing (branch on logit plus context): 1.97 against 1.95. Reverted.
- Reading whole node states through a dense projection (`read_state`): 2.24 to 2.20.
- Summing output rows with 16-bit integer coefficients: no faster, less exact. Opt-in only.
- Longer training of the original single tree: still improving at 1,800 steps (1.93) but slowly.

# Golden Tree Snake (GTS)

A fork of `state-spaces/mamba` (upstream commit `e9594ce`) that adds a tree-routed, bidirectional
state space mixer. Nothing upstream is changed except `mamba_ssm/__init__.py`, which now guards the
GPU-kernel imports so the GTS modules import on a CPU-only machine.

## What it is

Mamba-2's dense inner channels become the nodes of a binary tree, as in fast feedforward networks
(FFF). A token walks one root-to-leaf path. Only the nodes on that path are computed, written to
and read from. Tokens that land on the same node share that node's decaying memory, so the tree is
a learned hash and each node is a small key/query memory.

| Mamba-2 | GTS |
|---|---|
| inner channel | tree node |
| `in_proj` row for x | `node_in` row, which also decides the branch |
| `out_proj` column | `node_out` row |
| head (channels sharing a decay) | tree level, one clock per level |
| B, C from `in_proj` | B, C_fwd, C_bwd from `ctx_proj` |
| state per channel, updated every token | state per node, updated only when visited |

Two endpoints are known architectures:

- `use_context=False` is exactly UltraFastBERT's FFF layer.
- `depth=0` puts every token on one shared node: a one-channel linear-attention SSM.

## Files

| File | Contents |
|---|---|
| `mamba_ssm/modules/gts.py` | The mixer. `forward` is the differentiable training path; `forward_reference` is the token-at-a-time lazy-decay kernel. |
| `mamba_ssm/models/gts_encoder.py` | `GTSEncoder` and `GTSForMaskedLM`. |
| `tests/modules/test_gts.py` | CPU tests. |
| `kernel/gts_kernel.c` | Float32 C kernel for the mixer, a Mamba-2 single-token step for comparison, and a benchmark. |
| `mamba_ssm/modules/ternary.py` | Grouped absmean ternary quantiser with STE, activation quantiser, pack/unpack for export. |
| `mamba_ssm/utils/gts_qat.py` | Ternary student from a teacher, distillation loss, route distillation, monitoring. |
| `train_gts.py` | Masked-LM training: full precision, ternary from scratch, or QAT from a teacher. |
| `scripts/gts_ternary_demo.py` | The small comparison reported below. |
| `tests/modules/test_gts_ternary.py` | CPU tests for the ternary path. |
| `scripts/shakespeare_ar.py` | Autoregressive TinyShakespeare run: ternary GTS against a parameter-matched ternary Mamba-2 (a plain-PyTorch `Mamba2Ref`). |
| `kernel/ar_bench.c` | Token-at-a-time C inference for both of those models, checked against PyTorch, with timing. |

## Use

```python
import torch
from mamba_ssm import GTS, GTSConfig, GTSForMaskedLM

mixer = GTS(d_model=768, depth=11, d_state=16)
y = mixer(torch.randn(2, 128, 768))            # (2, 128, 768)
print(mixer.ops_per_token())

model = GTSForMaskedLM(GTSConfig(d_model=768, n_layer=12))
```

Only `torch` is required. Run the tests with `pytest tests/modules/test_gts.py`.

## Kernel

```
gcc -O3 -march=native -ffast-math -funroll-loops kernel/gts_kernel.c -o kernel/gts_kernel -lm
kernel/gts_kernel bench uniform
```

`verify` mode runs one layer on weights exported from PyTorch; the output matched `GTS.forward` to
float32 precision (max difference under 1e-4) at width 768, depth 11.

One run on one core of a 2.1 GHz Xeon with AVX-512, width 768, depth 11, state 16, random weights,
128-token sequences, float32:

| Block | Time per token per layer |
|---|---|
| GTS, both directions | about 12 us |
| Mamba-2, one direction, whole sequence through BLAS | about 170 us |
| Mamba-2, one direction, one token at a time | about 1,380 us |

Random weights visit nearly every node, so this is close to the worst case for the tree's cache
behaviour. That machine has an unusually large L3 cache. Nothing here says anything about quality.

## Ternary training

The recipe follows Ternary Mamba (arXiv:2606.18114), with BitNet b1.58's quantiser underneath:

- **Weights.** Node tables and `ctx_proj` are ternarised in groups of `ternary_group` weights
  (default 128) along each row. The group scale is the mean absolute latent weight, recomputed on
  every forward and never learned; a learnable scale is what that paper found collapses to mostly
  zeros. Gradients pass straight through to the latent weights.
- **Kept in full precision.** `A_log`, `dt_bias`, node biases, the conv, norms, embeddings, the head.
- **From a teacher.** `make_ternary_student` copies a trained full-precision model; `qat_loss` trains
  the copy with `alpha * KL(teacher || student) + (1 - alpha) * CE`, `alpha = 0.5`.
- **Route distillation (GTS-specific).** Wherever the student is at the teacher's node, its branch
  logit is pulled to the teacher's (`beta`). `path_agreement` reports how often that is.
- **Optional.** `act_bits=8` quantises the mixer input per token; `--lambda-warmup` blends
  quantisation in over the first steps.

```
python train_gts.py fp  --data corpus.txt --out fp.pt
python train_gts.py qat --data corpus.txt --teacher fp.pt --out ternary.pt
```

One small run (`scripts/gts_ternary_demo.py`): a byte-level masked LM over 4.7 MB of Python
standard-library source, width 128, 4 layers, depth 6, 64-byte sequences, one seed. The teacher
trained for 1,500 steps and each QAT arm for 500 more. Accuracy is on about 9,000 held-out masked
bytes, so differences under about one point are noise.

| Arm | Masked accuracy | Zero codes | Path agreement with teacher |
|---|---|---|---|
| Full precision, 2,000 steps | 45.5% | | |
| Teacher ternarised, no training | 21.4% | 27% | 0.69 |
| Ternary from scratch, 2,000 steps | 44.8% | 28% | |
| QAT, cross-entropy only | 44.7% | 28% | 0.75 |
| QAT + distillation | 45.1% | 28% | 0.76 |
| QAT + distillation + route distillation | 45.1% | 28% | 0.89 |
| ... + quantisation warmed in | 45.3% | 28% | 0.89 |
| ... + 8-bit activations | 45.0% | 28% | 0.89 |

What this shows: the path trains, ternarising without training breaks the model, the zero share
stays put, and route distillation moves the student's paths back towards the teacher's. What it
does not show: any accuracy benefit from distillation or route distillation, which are within
noise of each other here, or anything about a model of useful size.

## Autoregressive comparison with Mamba-2

`scripts/shakespeare_ar.py`, one run each: character-level TinyShakespeare, width 128, both models
ternary from scratch with the same quantiser, same wrapper, optimiser, batches and 1,200 steps
(batch 16, 128 characters), one seed, one learning rate (2e-3) that was not tuned for either.

| | GTS (causal, 4 layers, depth 8, state 16) | Mamba-2 (5 layers, expand 2, state 32) |
|---|---|---|
| Parameters | 557,445 | 556,153 |
| Validation loss, nats per character | 2.00 | 1.58 |
| Training loss | 1.93 | 1.39 |
| Time per token, C kernel, one core | 11 us | about 120 us |
| Tokens per second | about 89,000 | about 8,000 |
| Training step, PyTorch, one core | 657 ms | 618 ms |

GTS is about 11 times faster per token and clearly worse: Mamba-2 at step 300 (1.80) was already
ahead of GTS at step 1,200. GTS's training loss is high too, so it is underfitting, not overfitting.
Both C kernels reproduce their PyTorch model's logits (max difference under 1e-4, same top-1 on all
256 test tokens) and run the ternary weights as float32; neither is a packed ternary kernel.

## Forest GTS

Follow-up to the comparison above, same data, batches, schedule and ternary training. Switching the
original model's context off at test time moved its loss from 2.00 to 2.34: it used context, but
through 9 scalars per layer. The changes, all still pure GTS:

- **Forest** (`n_trees`). Many shallow trees. Every token visits every root, so the roots of a forest
  are dense channels in the Mamba-2 sense; a forest of depth 0 is close to Mamba-2, and one deep tree
  is the opposite extreme.
- **Heads** (`n_heads`). Trees are split into groups, each with its own clock per level, so nodes at
  one level no longer share a single decay.
- **Split activation** (`act="split"`). The coefficient is `gelu(logit) + ctx`, so context enters
  linearly instead of inside the GELU. `act="linear"` is `logit + ctx`.
- **Training path.** With small node tables the walk computes every node's logit in one matmul, as
  `fff.py` does (`dense_walk`), and the context uses scatter, matmul, gather over node columns
  (`_context_fast`). Inference still touches only the path.

Also added, and of little use in these runs: `read_state` (read whole node states through one dense
projection) and `write_logit=False`.

| | Original GTS | Forest GTS | Mamba-2 |
|---|---|---|---|
| Shape | 4 layers, 1 tree, depth 8 | 6 layers, 21 trees, depth 3, 3 heads, state 32 | 5 layers, expand 2 |
| Parameters | 557,445 | 556,595 | 556,153 |
| Path nodes per token per layer | 9 | 84 | |
| Validation loss, step 300 / 600 / 1,200 | 2.24 / 2.11 / 2.00 | 2.00 / 1.88 / 1.78 | 1.80 / 1.68 / 1.58 |
| Training loss | 1.93 | 1.62 | 1.39 |
| Time per token, C kernel, one core | about 5.5 us | about 40 us | about 105 us |

The forest closes about half of the gap to Mamba-2 (0.42 nats to 0.20) and is about 2.6 times faster
than it, where the original was many times faster. All three timings are from one back-to-back
session of three repetitions; an earlier session timed the original at 11 us and Mamba-2 at 120 us, so
treat the ratios as rough. One seed, one learning rate (2e-3), neither tuned.

Shorter 300-step screens (their own schedule, so compare only with each other): original shape with
`read_state` 2.20, plus `act="linear"` 2.16, plus a doubled learning rate 2.09; forest 2.03; a tree
with a small dense Mamba-2 block beside it (`--arch hybrid` in the script) 1.92.

## Routing with a straight-through gradient

`route_ste=True` gives branch decisions a gradient. The forward pass is unchanged: a branch is still
the hard step `logit > 0`. Going backward the step is replaced by `sigmoid(logit / route_ste_temp)`,
and a node's weight is the product of the branch values above it, so a branch logit receives the
difference between what the model would have output had the token gone right and had it gone left,
each side followed down by the token's own decisions. This needs every node's logit, coefficient and
would-be write, which the dense training path already has; inference and the C kernel are untouched.

Original GTS (4 layers, one tree of depth 8), same data, batches, schedule and ternary training:

| Validation loss at step | 300 | 600 | 900 | 1,200 | Training loss |
|---|---|---|---|---|---|
| Hard routing, no gradient | 2.24 | 2.11 | 2.04 | 2.00 | 1.93 |
| With `route_ste` | 2.20 | 2.05 | 1.99 | 1.95 | 1.85 |

One seed, temperature 1. The kernel runs the STE-trained model at the same speed (about 5 us per
token) and reproduces its PyTorch loss.

### Mixed-depth forest with the routing gradient

The mixed forest (`--arch mixed`: a bank of 32 depth-0 trees carrying the context plus four stateless
trees of depth 6, 4 layers, 8-bit activations, 588,785 parameters) with `route_ste` on its deep trees:

| Validation loss at step | 300 | 600 | 900 | 1,200 | Training loss | Time per token |
|---|---|---|---|---|---|---|
| Mixed forest | 2.06 | 1.94 | 1.89 | 1.84 | | about 8.5 us |
| + `route_ste` | 2.04 | 1.92 | 1.85 | 1.81 | 1.67 | about 8.5 us |
| + `route_ste`, learning rate 4e-3 | | 1.85 | 1.78 | 1.73 | 1.58 | about 8.5 us |
| Wide forest (2e-3) | 2.00 | 1.88 | 1.83 | 1.78 | 1.62 | about 21 us |
| Mamba-2 (2e-3) | 1.80 | 1.68 | | 1.58 | 1.39 | about 59 us |

Timings are three back-to-back repetitions in one session, with the original GTS at about 5 us.
The wide forest and Mamba-2 were not retrained at the doubled learning rate, and the mixed forest
has about 6% more parameters than they do. One seed each.

### Longer training, both models

The mixed forest (with `route_ste`) and Mamba-2 on the same 3,600-step schedule at learning rate 4e-3,
three times the earlier budget. Same data, batches and ternary training; one seed each.

| Validation loss at step | 600 | 1,200 | 1,800 | 2,400 | 3,000 | 3,600 | Training loss | Time per token |
|---|---|---|---|---|---|---|---|---|
| Mixed forest | 1.87 | 1.77 | 1.70 | 1.66 | 1.61 | 1.59 | 1.42 | about 7 us |
| Mamba-2 | 1.71 | 1.63 | 1.57 | 1.53 | 1.51 | 1.49 | 1.26 | about 48 us |
| Gap | 0.16 | 0.14 | 0.13 | 0.13 | 0.11 | 0.10 | | |

Training loss is the mean over the last 600 steps. The gap narrows with training but does not close:
Mamba-2 keeps improving too. Its training loss runs further below its validation loss (0.23 against
0.17), so it is the one closer to overfitting. Timings are three back-to-back repetitions. The kernel
matches PyTorch's top-1 on all 256 test tokens for both; for the mixed forest (8-bit activations) the
loss on them is 1.3796 against 1.3793, with one token's logits off by as much as 2.8.

### Bidirectional mixed forest and the width test

`GTSMixed` (in `modules/gts.py`) is the mixed forest as a library module; it runs in both directions
unless `causal=True`, and `GTSConfig(mixer="mixed", ...)` selects it in the encoder. `scripts/width_test.py`
compares it with a bidirectional Mamba-2 (`BiMamba2Ref`: one set of weights run in both directions,
outputs summed) at equal parameters. As width doubles the mixed forest adds one level to its deep trees.

Speed, causal kernels with random weights (there is no bidirectional kernel for the mixed forest yet):

| Width | Mixed forest | Mamba-2 | Ratio |
|---|---|---|---|
| 128 | 0.59M, 7.7 us per token | 0.56M, 46 us | 6x |
| 256 | 2.23M, 12.8 us | 2.10M, 157 us | 12x |
| 512 | 8.65M, 26 us | 8.18M, 636 us | 24x |

Quality, bidirectional masked-byte modelling on TinyShakespeare (15% masked), 600 steps at 4e-3, both
ternary, same batches, one seed:

| Model | Parameters | Loss at step 300 | Loss at step 600 | Masked accuracy at step 600 |
|---|---|---|---|---|
| Mixed forest, width 128 | 621,874 | 1.88 | 1.65 | 52.0% |
| Mamba-2, width 128 | 581,050 | 1.66 | 1.40 | 59.1% |
| Mixed forest, width 256 | 2,291,890 | 1.84 | 1.55 | 54.6% |
| Mamba-2, width 256 | 2,153,522 | 1.59 | 1.30 | 62.0% |

The speed ratio doubles with width; the quality gap does not close. At step 600 it is 0.25 nats and
7.1 points of masked accuracy at width 128, and 0.25 nats and 7.4 points at width 256.

### Half a billion parameters, speed only

`kernel/ar_bench synth ...` builds random packed ternary models directly in C, for sizes that do not
fit in PyTorch on a small machine, and times one token through the layer stack (no embedding, no
output head). Nothing at this size was trained or checked against PyTorch; at small size the synthetic
models time within about 20% of the trained ones. Width 1024, one core, best of three:

| Model | Mixer parameters | One token through the stack |
|---|---|---|
| Mixed forest, 27 layers, 4 deep trees of depth 10 | 456M | 0.44 ms |
| Mixed forest, 13 layers, 4 deep trees of depth 11 | 438M | 0.18 ms |
| Mamba-2, 68 layers, state 128, float activations | 449M | 46 ms |
| Mamba-2, the same with 8-bit activations | 449M | 31 ms |

Both mixed forests use 8-bit activations and touch 76 to 80 nodes per token per layer out of 8,220 to
16,412. The Mamba-2 kernel here is the same simple packed kernel, not a tuned one: it pays one
horizontal sum per 128 weights. A language model's output head over a 50,000-token vocabulary is not
included and would cost far more than the mixed forest's whole stack.

### Scripts for a first run at scale

`scripts/prepare_fineweb.py` writes a slice of FineWeb as GPT-2 tokens and `scripts/lm_run.py` trains
either model on it with gradient checkpointing; the defaults are 507.6M parameters for the mixed forest
(27 layers, deep trees of depth 10) and 500.4M for Mamba-2 (68 layers) at width 1024, 2,000 steps.
Both were smoke-tested at toy size on a CPU, through export and the kernel check, and have not been
run at the default size. `HANDOVER.md` has the procedure.

Other notes from the same period:

- **Longer training.** On a 3,000-step schedule the hard-routed original reached 2.11, 1.99 and 1.93
  at steps 600, 1,200 and 1,800 (the run was cut there): still improving, with shrinking gains.
- **Mixed-depth forest** (`--arch mixed`): a bank of 32 depth-0 trees that carry the context plus four
  stateless trees of depth 6, 4 layers, 8-bit activations, 588,785 parameters. 1.84 at step 1,200,
  about 9.5 us per token against about 21 for the forest and 5 for the original in the same session.
- **Kernel** (`kernel/ar_bench.c`): packed ternary rows (two bit masks per 16 weights), integer sums
  when the mixer input is int8 (`act_bits=8`), output sums kept in registers. A third argument
  `float` runs the weights as float32. With 8-bit activations a few tokens' logits differ from
  PyTorch's while the loss agrees to three decimals.

### Half a billion parameters, trained: FineWeb, both models

`scripts/lm_run.py` at its defaults on one A100 SXM 80GB (RunPod), 2,000 steps of 8 x 512 GPT-2 tokens of FineWeb
`sample-10BT` (8.2M tokens read), learning rate 1e-3 with 200 warmup steps and cosine decay, TF32 matmuls for
training (switched off for the export), gradient checkpointing, one seed each. Same batches for both.

| Validation loss at step | 0 | 500 | 1,000 | 1,500 | 2,000 | Training tokens/s | Peak GPU memory |
|---|---|---|---|---|---|---|---|
| Mixed forest, 507.6M (27 layers, deep trees of depth 10) | 10.98 | 6.43 | 6.06 | 5.87 | 5.80 | 5,010 | 9.5 GB |
| Mamba-2, 500.4M (68 layers, state 128) | 11.05 | 6.29 | 5.90 | 5.68 | 5.62 | 2,778 | 9.5 GB |
| Gap | | 0.15 | 0.17 | 0.18 | 0.18 | | |

Both learn (ln 50257 = 10.8) and are still falling. The gap at equal steps sits inside the 0.10 to 0.25 nats of the
small runs. 2,000 steps says little about quality beyond that.

Kernel on the trained weights, `kernel/ar_bench`, one core of a 2.1 GHz Xeon (AVX-512 VNNI, 260 MB L3), the two
models timed back to back:

| | Check against PyTorch (256 tokens) | Mixers per token | Total per token |
|---|---|---|---|
| Mixed forest | top-1 248/256, loss 5.9410 vs 5.9418, max logit diff 1.55 | 0.66 to 0.72 ms | 16 to 19 ms |
| Mamba-2, packed (column form) | top-1 256/256, loss 5.8228 vs 5.8228, max logit diff 3e-5 | 33.9 ms | 54 ms |
| Mamba-2, packed, row form (`m2rows`, the earlier kernel) | same | 36.9 ms | 57 ms |
| Mamba-2, weights as float32 (`float`) | same | 184 ms | 205 ms |
| Mamba-2 projections only, float32 BLAS, one thread (PyTorch) | | 122 ms (64 ms with weights in cache) | |

- **The mixed forest's check.** Eight differing top-1s is not a kernel error. PyTorch disagrees with itself as much:
  CPU float32 against the GPU-exported reference has top-1 253/256 and loss 5.9361 vs 5.9418, and CPU float64
  against CPU float32 has max logit difference 0.83 and loss 5.9379 vs 5.9361 (`scripts/f64_check.py`). With hard
  branches and 8-bit activations rounded per token in 54 mixers, float-order differences flip discrete decisions and
  compound through 27 layers. The kernel's loss is closer to the reference than float32 is to float64.
- **Trained against random.** The mixed forest's stack takes 0.66 to 0.72 ms trained against 0.43 to 0.44 ms with
  random weights (1.6x); Mamba-2's takes 33.9 ms trained against 32.5 ms synthetic. Both within 2x.
- **Fairness of the Mamba-2 number.** Neither BLAS nor the float path beats the packed kernel. The kernel was also
  improved first (one horizontal sum per row; column-form projections: 16 registers of 16 rows take masked adds of a
  broadcast input), 40.3 to 32.5 ms on the synthetic model. Its projections now run at about 1.1 masked adds per
  cycle against at most 2, so a perfect kernel of this kind gains at most about 1.8x on them. 130 us per layer is the
  SSM's 1 MB float32 state, read and written per token: a real cost of Mamba-2.
- **Ratios.** Mixers alone: about 47x. Whole model with the 50,257 x 1,024 output head: about 2.8x; the head is about
  18 ms of the mixed forest's total. A claim about the tree rests on the mixer time; a claim about a deployed model
  must count the head.
- **Asymmetries to keep in mind.** The mixed forest uses 8-bit activations and this Mamba-2 does not (`lm_run.py`
  sets `m2_act_bits=None`); Mamba-2's synthetic kernel with 8-bit activations is about 1.5x faster. This CPU's
  260 MB L3 holds either model's packed weights (about 112 MB), which flatters both against a typical desktop.
  Batch-1 decoding only: Mamba-2 would amortise its weight reads over a batch, the tree much less.

## A Triton scan for depth-0 trees

A depth-0 tree is visited by every token, so the mixed forest's bank is Mamba-2's SSD with one key/query group and
the token's own term excluded. `mamba_ssm/ops/gts_scan.py` computes it in chunks instead of the quadratic
(batch, t, s, heads) form; `GTS` uses it for depth-0 trees on CUDA (`scan_kernel=None`, or force with True/False).

- **Kernels.** Three over (chunk, batch x head): each chunk's own (N x P) state, a short sequential pass that turns
  them into the state entering each chunk, and the outputs. The same output kernel gives the forward Y, and in the
  backward pass dX and dB (one transposed pass, run the other way on the exclusive running sum) and dC (reusing the
  forward's states). No global running sum is formed: every exponent is a difference of chunk-local sums or a
  chunk's total. The log-decay gradient is <dY, Y> - <X, dX>, written by the dX pass, then a suffix sum.
- **Correctness.** `tests/modules/test_gts_scan.py`: all four direction and clock variants against the quadratic form,
  every gradient; the GTS layer against its dense path and the token-at-a-time reference, causal and bidirectional,
  with float64 as ground truth. 20 tests pass compiled on an A100 and in Triton's interpreter on a CPU.
- **Accuracy.** On an A100 every output and gradient is within about 2x of float32 PyTorch's error against float64,
  and the log-decay gradient is ten times more accurate at 8K tokens (2e-5 against 2e-4; `scripts/diag_scan.py`).

A100, bank shape (batch 8, 8 heads x 4 trees, state 16), forward + backward, `scripts/bench_scan.py`, log in
`results/scan_bench_a100.log`:

| Length | PyTorch quadratic | gts_scan, GPU kernel time | upstream `mamba_chunk_scan_combined`, GPU kernel time | gts_scan wall clock | upstream wall clock |
|---|---|---|---|---|---|
| 512 | 1.9 ms, 273 MB | 0.10 ms | 0.28 ms | 1.1 ms | 3.9 ms |
| 2,048 | 35 ms, 4.4 GB | 0.20 ms | 0.78 ms | 1.0 ms | 3.7 ms |
| 8,192 | | 0.62 ms | 2.25 ms | 1.1 ms | 4.0 ms |
| 32,768 | | 2.98 ms | 8.67 ms | 3.9 ms | 10.0 ms |

Up to 8K the wall clock is launch overhead on the CPU, for both. Upstream's kernel is run on the same shape and keeps
the diagonal term; it is not tuned for 4 channels per head.

Whole training steps of the 0.5B mixed forest (30 steps, including compilation): 512 x 8 tokens, 4,717 to 5,931
tokens/s (+26%); 2,048 x 2 tokens, 3,323 to 5,879 tokens/s (+77%). With the scan a step costs the same per token at
2,048 as at 512. The 0.5B runs above were trained before the scan existed, with the quadratic form.

What the scan does not touch: the deep trees, which compute every node in training (route_ste needs every node's
logit) and are most of a step's FLOPs, and the rest of the bank layer (projections, conv, quantisation), which is
now most of the bank's time: at 32K tokens 66 ms per layer, of which the scan is 3.

### Route kernels: the deep trees' straight-through gradient without the path weights

Under `route_ste` the training path built the path weights pi level by level (stack, then cat over all nodes) and ran
the activation, the sigmoid and the products over every (token, node), forward and backward, after an 11-level hard
walk whose result went unused. `mamba_ssm/ops/gts_route.py` replaces this for stateless trees on CUDA: pi going
forward is the one-hot of the path, and a path node's routing gradient needs only the path below it and the chain
from its other child that follows the token's own decisions (11 + 55 nodes per depth-10 tree, not 2,047). The
logits and the output are still dense matmuls. Tests: `tests/modules/test_gts_route.py` (values and every gradient
against the dense form, ternary with 8-bit activations, padding), 39 GPU tests passing on an A100.

One training step of the 0.5B mixed forest, 8 x 512 tokens, A100, TF32, `scripts/profile_step.py`, log in
`results/step_profile_a100.log`:

| | ms per step | Tokens/s | Peak memory |
|---|---|---|---|
| As the 0.5B run was trained (quadratic bank, dense path weights) | 795 | 5,151 | 9.5 GB |
| + scan for the bank | 646 | 6,342 | 9.5 GB |
| + route kernels | 370 | 11,063 | 9.5 GB |
| + no gradient checkpointing | 257 | **15,954** | 16.2 GB |
| Mamba-2 (`Mamba2Ref`), as trained | 1,474 | 2,778 | 9.5 GB |

At 2 x 2,048 tokens with both kernels and checkpointing, 11,763 tokens/s: length no longer costs extra. What is
left is mostly the dense matmuls of the deep trees (every node's logit and the output, about 140 ms of GPU time per
step with checkpointing), the output head, AdamW and many small elementwise kernels; bf16 is still untried.

### bf16, fusion and aligned GEMMs

The same step after the route kernels, 8 x 512 tokens on an A100, no gradient checkpointing (`scripts/profile_step.py`):

| | ms per step | Tokens/s |
|---|---|---|
| Route kernels and scan, float32/TF32 | 263 | 15,598 |
| + fused AdamW | 239 | 17,127 |
| + bf16 autocast | 229 | 17,889 |
| + node dimension and head padded to multiples of 64 (aligned GEMMs), fused ternary quantiser, conv as shifted sums | 188 | 21,738 |
| + `torch.compile` of each block | **121** | **33,894** |
| the same with gradient checkpointing (8.6 GB) | 152 | 26,929 |

- **Why bf16 alone gave little.** 4 x 2,047 tree nodes and a 50,257-row head made every GEMM misaligned, and cuBLAS
  fell back to slow kernels. Padding with zero rows the kernels never visit (and slicing the logits back) fixes it.
- **Exactness.** The fused quantiser (`mamba_ssm/ops/ternary_fused.py`) gives the PyTorch codes, scales within one
  float32 ulp; the shifted-sum conv equals `nn.Conv1d` (tested); padding and slicing leave the loss unchanged.
  bf16 autocast rounds activations in float32 before quantising, so the codes match the float32 export.
- **Quality.** 300 steps on FineWeb, validation loss: float32/TF32 6.5943, bf16 6.5979, bf16 + compile 6.5892.
- **What is left.** About 40% of the step is the deep trees' six dense bf16 GEMMs per layer, now running near
  peak; then fused AdamW (9 ms), the head and softmax, and small fused kernels.

`lm_run.py --amp --compile --no-checkpoint` trains this way. The 0.5B results above were trained before all of this,
in float32 with checkpointing.

### Bidirectional (BERT-style) training

The bidirectional mixed forest as a masked LM (`GTSForMaskedLM`, width 1024, 27 layers, deep trees of depth 10, ternary,
8-bit activations, 15% of positions labelled), one A100, no checkpointing, `scripts/profile_step.py --arch bert`:

| | 8 x 512 tokens | 2 x 2,048 tokens |
|---|---|---|
| Before this work (quadratic bank, walk-and-scatter trees, full head, float32) | 8,170 tokens/s | 4,378 tokens/s |
| + scan for the bank | 12,041 | |
| + route kernels (here without route_ste: plain FFF routing) | 16,676 | |
| + head on the labelled positions only | 17,237 | |
| + bf16, fused AdamW | 22,723 | |
| + `torch.compile` | **37,838** | **37,633** |
| the same with route_ste | 37,050 | |

What made the bidirectional path different: `gts_scan_bi` runs the bank's forward and reverse context in the same
launches (a direction grid axis, shared tensors at zero stride); the route kernels gained a mode without the
routing gradient (the encoder's default), replacing the level-loop walk and dense scatter; and
`GTSForMaskedLM(..., labelled_only=True)` scores the about 15% of positions that carry a label instead of all of them,
for the same loss. 64 GPU tests pass.

### A 110M BERT-style run, and CPU inference

`scripts/bert_pretrain.py`: width 768, 14 layers, bank 32 trees (8 heads, state 16) plus 4 deep trees of depth 9 with
route_ste, 112.9M parameters; bert-base-uncased WordPiece on English Wikipedia, RoBERTa-style masking, 512-token
windows, batch 64, one A100 at about 188K tokens/s. Checkpoints (float with optimizer state, and binarized) are in
`checkpoints/bert110m/`.

| | steps | tokens | validation loss | masked accuracy |
|---|---|---|---|---|
| Phase 1 (shards 0-6, peak lr 1.5e-3, 157 min) | 53,753 | 1.76B | 2.815 | 50.5% |
| Phase 2 (resumed on shards 7-19, re-warm to 7.5e-4, 161 min) | 108,829 | 3.57B | 2.713 | 51.7% |
| Phase 3, **GTS3** (resumed on shards 0-39, re-warm to 5e-4, 336 min) | 225,197 | 7.38B | 2.615 | 53.2% |

CPU inference of the trained model with `kernel/enc_bench.c` (packed ternary weights, int8 activations, OpenMP over
tokens and over the bank's two directions), 4-core cloud x86 with VNNI, the encoder without the head, sequence 512:
7,282 tokens/s on 1 thread, 13,574 on 2, 26,662 on 4 (phase 2 weights; phase 1's are within 3%). BERT-base in
float32 PyTorch on the same machine: 618 and 1,975 tokens/s on 1 and 4 threads. The masked-LM head (float32,
30,522 x 768) costs 1.1 ms per scored position on 4 threads. The kernel matches PyTorch's masked-LM loss (4.021 vs
4.010 on the test sequence); individual logits differ by up to about 2, which the model's hard branches and 8-bit
rounding also produce from a 1e-7 nudge of the weights.

## Departures from Mamba-2

- **Bidirectional by default.** Forward and backward context share the key B and use separate queries.
  The self term is excluded from both; the node's own activation carries it. `causal=True` keeps the
  forward context only and makes the conv causal, for autoregressive models.
- **Decay per level, not per head.** A skipped node catches up with one
  `exp(clock_now - clock_at_last_write)`.
- **No gate and no inner norm.** The output coefficient is `gelu(logit + context)`.
- **State is written before the activation.** With FFF's routing rule a left turn has a negative
  logit, and GELU of that is near zero, so writing after the activation would store almost nothing.
- **The conv is centred, sits on the residual stream, and starts as the identity.**
- **Training is the quadratic SSD form.** Cost grows with the square of sequence length. There is
  no chunked or fused kernel.

## Not done

- An integer or ternary kernel for the reference tree layer: `kernel/gts_kernel.c` is float32 and
  single-threaded. (The mixed-forest encoder has one: `kernel/enc_bench.c`, above.)
- Any routing gradient. Branches are hard and learn only through the node's value, as in FFF.
  `GTS.path_stats` reports how many nodes per level are in use.
- Wide trunk nodes. Every node carries a scalar value; `n_trees > 1` is the only way to widen.
- A matched baseline for the 110M encoder (a BERT or Mamba-2 encoder trained on the same tokens), so its 2.713 has
  nothing to compare against yet.

## GTS-Uni: depth recurrence (masked LM and autoregressive)

The whole stack runs `loops` times with shared weights (Universal-Transformer style). Passes after the first add a
learned per-pass embedding and add their update `gate * (stack(x + e) - (x + e))` through a per-channel gate that
starts at zero, so a looped model built from one-pass weights computes exactly what they do (tested for both). Each
training step samples the pass count (default 1/2/3 passes at 10/20/70%), so the same weights run at any depth: one
pass at the one-pass model's speed, more for quality, chosen per query.

- Masked LM (`GTSConfig(loops, latent_tokens)`, `scripts/bert_pretrain.py --loops`): from pass 2, `latent_tokens`
  learned scratch vectors sit after `[CLS]`; the bidirectional scan lets each pass write a summary into them and the
  next pass read it back. `kernel/enc_bench.c` format 8 runs the passes (4 threads, 512 tokens: 27K, 12.9K, 7.2K
  tokens/s at 1, 2, 3 passes).
- Autoregressive (`TinyLM(loops=...)`, `scripts/ar_pretrain.py --loops`): no latent tokens (in a causal model, tokens
  at the start see only the start). `kernel/ar_bench.c` format 7 keeps one recurrent state per pass and layer, sharing
  the weights; checked against PyTorch to 1e-5 on the logits.
- Warm start: `--init-from` a one-pass checkpoint, with `--new-param-lr` for the pass embeddings, gates (and latents).
  A first masked-LM try with one learning rate of 3e-4 for everything, from a fresh optimizer, knocked GTS3 off its
  minimum (validation 2.615 -> 2.78); the run in progress uses 3e-5 for the shared weights and 1e-3 for the new ones.

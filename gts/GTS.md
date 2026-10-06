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

- An integer or ternary kernel. `kernel/gts_kernel.c` is float32 and single-threaded. `pack_ternary` produces
  the codes and scales such a kernel would load.
- Any routing gradient. Branches are hard and learn only through the node's value, as in FFF.
  `GTS.path_stats` reports how many nodes per level are in use.
- Wide trunk nodes. Every node carries a scalar value; `n_trees > 1` is the only way to widen.
- Any run at useful size. The one head-to-head above, at half a million parameters, has GTS well behind Mamba-2.

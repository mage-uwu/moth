# EVA vet: EVA's recipe at pilot 1's budget (14M / 47M / 125M tokens), one A100

Attempt 1 (results/eva_vet/attempt1): forced resume failed (torch.compile's _orig_mod. prefix in state.pt) -> fixed.
Attempt 2 (attempt2): Stage 3 in full precision at lr 3e-4 diverged at 22M tokens (KL 1.86 -> 8.7) -> no weight
decay on gains / biases / SSM scalars, Stage 3 lr 2e-4, gradient-spike guard. A first guard froze training (median
of accepted norms only) -> every norm in a 50-step window, at most 3 skips in a row.
Final run (final/, commit 27e7096; student without the linear path, projections in groups of 128):

| | pilot 1 | EVA vet |
|---|---|---|
| Stage 1 matrix_rel | 0.25 | 0.25 |
| Stage 2 attn_rel / mlp_rel | ~0.18 / 0.273 | 0.155 / 0.276 |
| Stage 3 before ternary (full precision) | - | KL 1.359, agreement 56.9% |
| **final, ternary** | KL 1.68, agreement 51.8%, CE 2.89 | **KL 1.447, agreement 55.6%, CE 2.66** |
| cost of the ternary ramp | ~0.25 KL | ~0.09 KL |
| skipped updates (guard) | - | 22 in Stage 3 |
| resume after a forced kill | not tested | worked (Stage 2, same trajectory) |

Quick downstream check (scripts/downstream_check.py: SST-2 8K / MNLI 16K training subsets, best of 2-3 epochs):

| | SST-2 | MNLI-m | STS-B | mean |
|---|---|---|---|---|
| ModernBERT-large (teacher) | 94.5 | 86.9 | 91.7 | 91.0 |
| student, Stage 3 midpoint (full precision) | 85.7 | 59.3 | 79.2 | 74.7 |
| student, final (ternary) | 85.9 | 61.5 | 77.4 | 74.9 |

At ~186M tokens of training, sentiment transfers early; MNLI (reasoning across two sentences) trails the teacher by
25 points.

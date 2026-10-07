# MOHAWK pilot 1: ModernBERT-large -> GTS-L (405M, ternary, attention-free), one A100, ~$5

`scripts/mohawk_distill.py` (time budgets) via `pod/mohawk_job.sh`, FineWeb-Edu sample-10BT shard 0 in ModernBERT's
tokenizer (748M tokens; 2M held out), 512-token windows, 30% masking for Stage 3 and the evaluations.

| stage | budget | tokens | tokens/s | $ per 1B tokens (A100 $1.59/h) | end state |
|---|---|---|---|---|---|
| 1 matrix orientation | 15 min | 14.2M | 15.8K | 28.0 | mixing-matrix relative error 0.25 |
| 2 hidden-state alignment | 35 min | ~50M | 24.3K | 18.2 | attention-block rel. error ~0.18, MLP -> trees ~0.28 |
| 3 end-to-end KD (ternary ramp over the first 40%) | 100 min | ~125M | 20.9K | 21.1 | see below |

Held-out evaluation (student vs teacher on the same masked text; teacher CE 1.288):

| point | student CE | KL(teacher || student) | top-1 agreement |
|---|---|---|---|
| after Stage 2 | 22.52 | 21.32 | 0.0% |
| Stage 3, ternary 0.5 | 4.15 | 2.94 | 35.5% |
| Stage 3, ternary 1.0 (ramp done) | 4.03 | 2.81 | 37.8% |
| Stage 3, ternary | 3.40 | 2.19 | 44.6% |
| Stage 3, ternary | 3.21 | 1.99 | 47.3% |
| **final (ternary, after the learning-rate decay)** | **2.89** | **1.68** | **51.8%** |

Stage 2 alone leaves the chained student broken (errors compound over 56 sub-blocks); Stage 3 repairs it within
~1M tokens. The ternary ramp cost ~0.25 KL while it ran; after it, learning continued (~0.8 KL in 55M tokens) and the
final decay gave another 0.3. Checkpoints were not kept (the pod's disk). Next: EVA (full precision until late,
longer Stage 2), vetted at this budget by `pod/eva_vet_job.sh`.

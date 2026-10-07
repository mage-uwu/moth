# A/B: the side-branch routing gradient (route_ste) on or off

`pod/ste_ab_job.sh`, one A100: GTS3's float weights (step 225,197) continued twice for 25 minutes each on the same
data (Wikipedia shards 20 to 25, 480M tokens), learning rate 1e-4 with 200 warmup steps, the same validation (40
batches, the held-out shard). Arm A as trained (route_ste: the straight-through gradient through the branches, with
the side-branch chains); arm B path-only (route_ste=False, plain FFF routing: what a fully sparse training step
computes). Arm B runs faster (201K against 190K tokens/s dense), so its time-fitted schedule has more steps.

| step | A: with side-branch gradient | B: path-only |
|---|---|---|
| 0 (GTS3) | 2.6013 | 2.6013 |
| 500 | 2.6453 | 2.6941 |
| 1,000 | 2.6491 | 2.7003 |
| 2,000 | 2.6424 | 2.7090 |
| 3,000 | 2.6324 | 2.7002 |
| 4,000 | 2.6081 | 2.6848 |
| 5,000 | 2.5946 | 2.6735 |
| 6,000 | 2.5766 | 2.6571 |
| 6,500 | 2.5738 | 2.6516 |
| end | **2.5733** (step 6,899), acc 53.77% | **2.6418** (step 7,448), acc 52.87% |

Path-only ends 0.069 worse with 8% more steps, and above where it started. Its trees also lose balance:

| leaf balance (usage perplexity / 512) | dead nodes overall |
|---|---|
| GTS3: 0.28 | 6.0% |
| A: 0.28 | 5.9% |
| B: 0.21 (worst layer 0.13) | 7.5% |

Per level (balance A / B): level 1 0.94 / 0.89, 3 0.77 / 0.68, 5 0.59 / 0.48, 7 0.42 / 0.32, 9 0.28 / 0.21.

Conclusion: the side-branch gradient is doing real work, for both quality and routing balance, at least when
switching a model trained with it. The fully sparse path-only training step (258K tokens/s, 1.36x) is kept as an
option (`route_ste=False` with `sparsify`), not the default. `ab_ste.log` is arm A's training log; arm B's log and
the tree-usage output were read from the pod before it was terminated and are transcribed above.

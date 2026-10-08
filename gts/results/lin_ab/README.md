# Linear path beside the trees: Stage 2 A/B

GTS-L's deep trees with and without a rank-128 ternary linear path (`GTSLConfig.linear_rank`), initialised before
Stage 2 from the least-squares affine fit of each teacher MLP (reduced-rank regression; `GTSLBlock.init_linear`).
Stage 2 alone, 47M tokens (pilot 1's budget), A100, same code otherwise (`pod/lin_ab_job.sh`, commit 27e7096).
Errors are relative squared L2 to the teacher sub-block's output on held-out FineWeb-Edu (Stage 2's metric).

| | trees | trees + linear path |
|---|---|---|
| MLP (tree) error, mean over layers | 0.2766 | **0.1910** (-31%) |
| attention (BiSSD) error | 0.1618 | 0.1614 |
| start of Stage 2 (the least-squares init alone) | ~1.0 | 0.33 |
| CPU cost (kernel/gtsl_bench.c) | | 5-7% of the time |

Per layer:

| layer | trees | + linear | change |
|---|---|---|---|
| 0 | 0.069 | 0.034 | -51% |
| 1 | 0.259 | 0.167 | -36% |
| 2 | 0.351 | 0.246 | -30% |
| 3 | 0.373 | 0.238 | -36% |
| 4 | 0.394 | 0.256 | -35% |
| 5 | 0.325 | 0.230 | -29% |
| 6 | 0.060 | 0.043 | -28% |
| 7 | 0.013 | 0.017 | +33% |
| 8 | 0.208 | 0.151 | -27% |
| 9 | 0.400 | 0.224 | -44% |
| 10 | 0.373 | 0.246 | -34% |
| 11 | 0.471 | 0.338 | -28% |
| 12 | 0.232 | 0.168 | -28% |
| 13 | 0.478 | 0.324 | -32% |
| 14 | 0.012 | 0.013 | +7% |
| 15 | 0.462 | 0.307 | -33% |
| 16 | 0.517 | 0.377 | -27% |
| 17 | 0.519 | 0.364 | -30% |
| 18 | 0.015 | 0.013 | -14% |
| 19 | 0.035 | 0.005 | -85% |
| 20 | 0.409 | 0.284 | -31% |
| 21 | 0.322 | 0.224 | -30% |
| 22 | 0.272 | 0.186 | -31% |
| 23 | 0.235 | 0.170 | -28% |
| 24 | 0.201 | 0.150 | -25% |
| 25 | 0.387 | 0.320 | -17% |
| 26 | 0.208 | 0.158 | -24% |
| 27 | 0.146 | 0.094 | -36% |

Why: an affine map alone fits the teacher's MLPs about as well as the trees did (scripts/mlp_geometry.py: 0.10-0.43,
mean ~0.28), and four trees' always-on roots give only a rank-4 linear part, so the trees were rebuilding it piecewise.
With the linear part given, the trees fit the residual.

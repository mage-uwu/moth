# GLUE, first run: effectively adapter-only (superseded)

GTS3 fine-tuned from its **binarized** checkpoint at learning rate 1e-4 (`pod/glue_job.sh` before the fix). The saved
classifiers show that almost no ternary code changed (RTE: 0 of 22.3M packed bytes; SST-2: 0.04%): latents rebuilt
from 2-bit codes sit on their quantised values, half a step or more from a rounding threshold, and a few hundred
steps at 1e-4 never get there. So only the float parameters trained (head, norms, embeddings, biases, gates).

| SST-2 | MRPC F1 | RTE | QNLI | MNLI-m | CoLA MCC | STS-B Spearman | average |
|---|---|---|---|---|---|---|---|
| 83.4 | 79.3 | 48.0 | 68.4 | 54.7 | 0.19 | 0.19 | 53.2 |

The fix (`--ternary-lr`, float-checkpoint starts) is in `pod/glue_job.sh`; a 50-step local check with it changed 9.3%
of the packed bytes. `eval_partial.log` is this run's per-epoch log, including GTS-Uni (3 passes) until it was stopped.

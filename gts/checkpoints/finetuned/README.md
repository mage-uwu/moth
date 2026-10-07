# Fine-tuned GTS models

Task models built on the GTS masked LMs in `checkpoints/bert110m/`, all binarized (`mamba_ssm/utils/ternary_pack.py`:
2-bit ternary codes plus float heads, norms and biases) and split into 90 MB parts. Rebuild and check:

```bash
cd checkpoints/finetuned/sys1          # or glue_gts3
for f in $(cut -d' ' -f3 SHA256SUMS); do cat $f.part* > $f; done
sha256sum -c SHA256SUMS
```

All were trained from the **float** checkpoints with the ternary weights on their own learning rate (`--ternary-lr
3e-4`, picked on GLUE's small tasks); see `pod/glue_job.sh` for why (from a binarized start no ternary code moves).

## `sys1/`: typed-decision models (Laya's technique)

`scripts/sys1_train.py`: [MASK]-per-option scoring, a 2-layer GTS decision head with an act/escalate head, RLCD
(Gaussian logit exploration, log + spherical (+ RPS) rewards, group-mean REINFORCE) plus soft cross-entropy, and
per-(type, option count) temperatures, which are stored in each file. Load one:

```python
import sys; sys.path.insert(0, "scripts")
from sys1_train import Sys1
model, temperatures = Sys1.load("sys1_gts3.pt")
```

| file | backbone | trained on | test accuracy / KL / Brier / ECE | escalate AUROC |
|---|---|---|---|---|
| `sys1_gts3_general.pt` | GTS3 | 500K general typed decisions | 0.344 / 0.449 / 0.237 / 0.052 (zero-shot) | 0.50 |
| `sys1_gts3.pt` | GTS3 | then the benchmark's train (5 epochs) | **0.578** / 0.256 / 0.143 / 0.076 | 0.66 |
| `sys1_uni_3pass_general.pt` | GTS-Uni, 3 passes | 500K general typed decisions | 0.291 / 0.470 / 0.250 / 0.073 (zero-shot) | 0.46 |
| `sys1_uni_3pass.pt` | GTS-Uni, 3 passes | then the benchmark's train | 0.575 / 0.269 / 0.151 / 0.093 | 0.64 |

Benchmark: LocalLLaMA/typed-decisions test (2,000). The same harness with ModernBERT-base reaches 0.777 / 0.094 /
0.051 / 0.168 fitted (its weights are not kept here); the prior is 0.479 / 0.327 / 0.181 / 0.034. Results with
per-type and per-depth breakdowns: `results/sys1/`.

## `glue_gts3/`: GLUE classifiers on GTS3

`scripts/glue_finetune.py` (3 epochs, batch 32, 128 tokens, lr 1e-4, ternary lr 3e-4; MNLI and QNLI on 100K
examples), each task's best epoch on the dev set. Head on [CLS] and the mean final state; pairs as "[CLS] a [SEP] b
[SEP]". Load one: `GTSClassifier.load("sst2.pt")` (from `scripts/glue_finetune.py`; the file also holds its task,
label count, epoch and dev scores).

| SST-2 | MRPC F1 | RTE | QNLI | MNLI-m | CoLA MCC | STS-B Spearman |
|---|---|---|---|---|---|---|
| 86.4 | 81.5 | 52.7 | 73.5 | 62.9 | 0.17 | 0.58 |

The same harness: BERT-base averages 79.8, DistilBERT 77.9, TinyBERT-4L 70.4, GTS3 61.7 (`results/glue/`). GTS-Uni's
classifiers (61.2 at 3 passes, 61.3 at 1) were not kept.

# GTS-Uni-Sys1

A small, CPU-only decision model. Give it a situation, a question and a set of options; it returns a probability for
each option, plus a confidence that its top pick is right (low confidence means "send this to a human").

It is built on **GTS-Uni**: a 113M-parameter ternary (weights in {-1, 0, +1}), attention-free encoder made of
hard-routed trees and bidirectional scans, which can run its shared layers 1, 2 or 3 times. The decision head follows
Laya's typed-decisions recipe: every option scored at its own [MASK] token, trained with soft cross-entropy plus RLCD
(proper-scoring-rule rewards), with calibrated temperatures. The whole model is a 125 MB file.

> **Research toy, not a product.** It is weak: 57.5% accuracy on the LocalLLaMA typed-decisions test against 77.7%
> for ModernBERT-base trained the same way, and it misses obvious cases (try the phishing example). Don't use it for
> anything that matters, and never for safety, medical or mental-health triage.

## Run it

```bash
./setup.sh                                 # once: fetches the weights (125 MB) and creates .venv (~2 min)
.venv/bin/python decide.py                 # the 8 examples in examples/decisions.json
.venv/bin/python decide.py -i              # interactive: type your own situation, question and options
```

More:

```bash
.venv/bin/python decide.py --passes 1      # 1 pass through the shared layers: ~3x faster, same accuracy
.venv/bin/python decide.py --adaptive 0.7  # start at 1 pass, add passes while confidence < 0.7
.venv/bin/python decide.py --bench         # decisions per second on your Mac
.venv/bin/python decide.py --json mine.json > out.json
```

No network is needed after setup. `setup.sh` gets the weights from the mage-uwu/moth repo with your own git
login (a sparse clone of that one file); to use a copy you already have, put it at `weights/sys1_uni_3pass.pt` first.

## Your own decisions

A JSON list; `descriptions` and `type` are optional. `state` can be text or any JSON object.

```json
[{"state": {"ticket": "Refund asked 40 days after delivery; policy is 30 days", "order_value_usd": 89},
  "question": "How should support resolve this?",
  "options": ["refund", "store_credit", "decline", "escalate"],
  "descriptions": ["Refund in full.", "Offer store credit.", "Decline, citing policy.", "Send to a supervisor."],
  "type": "choice"}]
```

Types: `choice` (pick one), `noul` (true/false statements; options `["false", "true"]`), `score` (an ordinal level;
options `["0", "1", "2", "3"]` with a description per level). Everything must fit in 512 tokens; long states are cut.

From Python:

```python
from gts_sys1 import Decider
d = Decider()                              # loads once (~5-10 s)
r = d.decide(state="Server CPU at 97% for 2 hours, latency normal",
             question="Should we page on-call?", options=["page", "ticket", "ignore"], passes=1)
r["choice"], r["probs"], r["confidence"], r["ms"]
```

## How it scores

LocalLLaMA/typed-decisions test (2,000 decisions), after training on 500K general typed decisions and the benchmark's
train split; same harness for every row:

| | accuracy | KL | Brier | escalate AUROC |
|---|---|---|---|---|
| ModernBERT-base (149M, attention) | 0.777 | 0.094 | 0.051 | 0.76 |
| GTS3 (one-pass GTS) | 0.578 | 0.256 | 0.143 | 0.66 |
| **GTS-Uni-Sys1 (this)** | **0.575** | 0.269 | 0.151 | 0.64 |
| always the most common answer | 0.479 | 0.327 | 0.181 | |

The extra passes don't help (0.575 at 1, 2 and 3), so `--passes 1` is the sensible setting. The confidence signal is
weak (AUROC 0.64): treat "escalate" as a hint.

## What's inside

```
decide.py           command line
gts_sys1.py         Decider: load once, decide many
gts/sys1_model.py   the decision model (encoder + 2-layer GTS head + act/escalate head)
gts/mamba_ssm/      the GTS layers (pure PyTorch; Triton kernels only switch on with an NVIDIA GPU)
weights/            sys1_uni_3pass.pt: ternary weights as 2-bit codes, float heads, fitted temperatures
tokenizer/          bert-base-uncased WordPiece
examples/           sample decisions
assets/             a render of a real GTS3 routing tree (layer 8), drawn from measured token traffic
```

Training code, the GLUE results and the other checkpoints live in the `mage-uwu/moth` repo (`gts/`).

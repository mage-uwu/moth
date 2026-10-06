# GTS BERT-style 110M checkpoints

A bidirectional ternary GTS mixed forest trained as a masked language model on English Wikipedia
(`scripts/bert_pretrain.py`; settings and the whole validation curve are in each phase's `result.json`).

| | |
|---|---|
| Model | `GTSForMaskedLM`, width 768, 14 layers, each a bank of 32 depth-0 trees (8 heads, state 16) plus 4 stateless trees of depth 9 with the routing gradient; ternary weights (group 128), 8-bit activations; 112.9M parameters, 23.4M of them the tied embedding |
| Tokenizer | `google-bert/bert-base-uncased` WordPiece, vocabulary 30,522 |
| Data | `wikimedia/wikipedia` 20231101.en; validation from shard 40 |
| Objective | RoBERTa-style masked LM: 512-token windows from `[CLS]`, 15% of tokens, 80/10/10, no next-sentence prediction |
| Training | AdamW (0.9, 0.98, eps 1e-6, weight decay 0.01), batch 64 x 512 tokens, bf16, one A100 |

## Phase 1

Shards 0 to 6 (902M tokens), 53,753 steps (1.76B tokens, about two passes), peak learning rate 1.5e-3, 1,000
warmup steps, cosine to 1.5e-4; 157 minutes. Validation: loss 2.815, masked-token accuracy 50.5%.

## Phase 2

Resumed from phase 1's `checkpoint.pt` (weights, AdamW state, step) on new data, shards 7 to 19 (1.16B tokens), for the
same budget: 55,076 more steps (1.80B tokens) to step 108,829, 3.57B tokens in all. The learning rate re-warms from
phase 1's last 1.5e-4 to 7.5e-4 over 1,000 steps, then cosine to 7.5e-5; 161 minutes. Validation (same held-out shard
as phase 1) rises to 3.00 during the re-warm, passes below phase 1's 2.815 at step 88,000 (2.803) and ends at loss 2.713, masked-token
accuracy 51.7% (best 2.707 / 51.9% at step 106,000). `result.json` holds both phases' curves and settings.

## Files

Each checkpoint is split into 90 MB parts (GitHub refuses files over 100 MB). Rebuild and check:

```bash
cd checkpoints/bert110m/phase1   # or phase2
cat checkpoint.pt.part* > checkpoint.pt
cat binarized.pt.part* > binarized.pt
sha256sum -c SHA256SUMS
```

- `checkpoint.pt` (float, 1.29 GB): `{"model": latent float32 weights, "optimizer": AdamW state, "step", "config",
  "args", "curve", "generator": the data sampler's state}`. Resume with
  `scripts/bert_pretrain.py train --resume checkpoint.pt ...`.
- `binarized.pt` (114 MB): every ternary weight as 2-bit codes plus its group scales, every other tensor in float32
  (format in `mamba_ssm/utils/ternary_pack.py`). Load into a model:

  ```python
  from mamba_ssm.models.gts_encoder import GTSConfig, GTSForMaskedLM
  from mamba_ssm.utils.ternary_pack import load_binarized
  import torch
  blob = torch.load("binarized.pt")
  model = load_binarized(blob, GTSForMaskedLM(GTSConfig(**blob["config"])))
  ```

  Its ternary codes are identical to the float checkpoint's and its quantised values agree to within 2.5e-7
  (relative). Logits can still differ by a few units on some tokens: the model's hard branches and per-token 8-bit
  rounding flip on last-bit differences, and nudging the float weights by 1e-7 moves the logits as much. The masked-LM
  loss is the same (4.7212 float, 4.7165 binarized on held-out text).
- `examples.json`: top-5 predictions at `[MASK]` for a few sentences, e.g. "The [MASK] Ocean is the largest ocean on
  Earth" -> pacific, atlantic, indian, arctic, open
  (phase 2: ... arctic, southern).

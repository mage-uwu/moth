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

## Phase 3: GTS3

**GTS3** is the name of this checkpoint (`phase3/`): the current best GTS masked LM, and the starting point for
GTS-Uni.


Resumed from phase 2's `checkpoint.pt` for a $10 budget (one A100, 336 minutes): all 40 Wikipedia training shards
(0 to 39, 3.88B tokens; 20 to 39 new, 0 to 19 seen before), 116,368 more steps (3.81B tokens) to step 225,197, 7.38B
tokens in all. The learning rate re-warms from phase 2's last 7.5e-5 to 5e-4 over 2,000 steps, then cosine to 5e-5.
Validation rises to 2.885 in the re-warm, passes below phase 2's 2.713 at step 184,000 and ends at loss **2.615**,
masked-token accuracy **53.2%**. (A distillation leg from bert-base-uncased was tried first and stopped; see
`results/bert110m_kd/`.)

| | steps | tokens | validation loss | masked accuracy |
|---|---|---|---|---|
| Phase 1 | 53,753 | 1.76B | 2.815 | 50.5% |
| Phase 2 | 108,829 | 3.57B | 2.713 | 51.7% |
| Phase 3 (**GTS3**) | 225,197 | 7.38B | 2.615 | 53.2% |

## GTS-Uni

**GTS-Uni** (`uni/`) is GTS3 made depth-recurrent: the same 14 layers run up to 3 times (shared weights), each pass
with its own learned pass embedding and a per-channel gate initialised at zero, plus 16 learned latent tokens appended
to the sequence as a scratchpad (`GTSConfig(loops=3, latent_tokens=16)`; see `GTS.md`). At initialisation it computes
exactly GTS3. Trained from GTS3's float weights (`--init-from`) for $10 (one A100, 359 minutes): 48,700 steps, 1.60B
tokens, the pass count sampled per step (1/2/3 with probability 0.1/0.2/0.7), learning rate 3e-5 for the pretrained
weights and 1e-3 for the new ones (pass embeddings, gates, latents), 500 warmup steps, cosine down.

| GTS-Uni at step 48,700 | validation loss | masked accuracy | CPU tokens/s (4 threads, 512 tokens) |
|---|---|---|---|
| 1 pass | 2.578 | 53.7% | 27K |
| 2 passes | 2.562 | 53.9% | 12.9K |
| 3 passes | **2.561** | **53.9%** | 7.2K |
| GTS3 (start) | 2.618 | 53.1% | 24-27K |

The extra passes help (2.578 to 2.561), but almost all of it comes from the second; the run also improved the one-pass
model by 0.04 on its own. Same file layout as below (`checkpoint.pt` float with AdamW state; `binarized.pt`, whose
ternary codes match the float checkpoint's exactly: 0 mismatches across 89.3M ternary weights). Load with the
snippet below; `model(input_ids, loops=2)` picks the number of passes.

## Files

Each checkpoint is split into 90 MB parts (GitHub refuses files over 100 MB). Rebuild and check:

```bash
cd checkpoints/bert110m/phase3   # or phase1, phase2, uni
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

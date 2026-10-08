# GTS-KD-110: GTS3's architecture distilled from ModernBERT-base

GTS3's architecture (`checkpoints/bert110m/`), retrained from scratch in ModernBERT's tokenizer with logit distillation
from ModernBERT-base (`pod/kd110_job.sh`, `scripts/bert_pretrain.py`; settings and the whole validation curve are in
`result.json`, the log in `results/kd110/train.log`).

| | |
|---|---|
| Model | `GTSForMaskedLM`, width 768, 14 layers, each a bank of 32 depth-0 trees (8 heads, state 16) plus 4 stateless trees of depth 9; ternary weights, 8-bit activations; 128.2M parameters (the larger vocabulary's tied embedding accounts for the difference from GTS3's 112.9M) |
| Tokenizer | `answerdotai/ModernBERT-base`, vocabulary 50,368 (`[CLS]` 50281, `[SEP]` 50282, `[PAD]` 50283, `[MASK]` 50284); token embeddings initialised from the teacher's |
| Data | `HuggingFaceFW/fineweb-edu` sample/10BT files 000 to 009 (7.49B tokens); 2M validation tokens held out |
| Objective | 30% masking (80/10/10). At masked positions: 0.75 x T^2 x KL(teacher \|\| student) at T = 2, plus 0.25 x cross-entropy. The teacher (ModernBERT-base, frozen, bf16) sees the same masked input |
| Training | AdamW, batch 64 x 512 tokens, peak learning rate 1.5e-3, 2,000 warmup steps then cosine; 158,658 steps = 5.20B tokens in 958 minutes on one A100 (91K tokens/s) |

Validation, on this run's own held-out FineWeb-Edu tokens in ModernBERT's vocabulary, so **not comparable with GTS3's
Wikipedia / WordPiece numbers**: final loss **2.766**, masked-token accuracy **50.7%**. The teacher scores 1.251 / 73.5%
on the same tokens. Downstream results (GLUE dev and Sys1 typed decisions, with GTS3's exact settings) are in
`results/kd110/`.

## Files

Each checkpoint is split into 90 MB parts (GitHub refuses files over 100 MB). Rebuild and check:

```bash
cd checkpoints/kd110
cat checkpoint.pt.part* > checkpoint.pt
cat binarized.pt.part* > binarized.pt
sha256sum -c SHA256SUMS
```

- `checkpoint.pt` (float, 1.47 GB): `{"model", "optimizer", "step", "config", "args", "curve", "generator", "phases"}`.
  Resume with `scripts/bert_pretrain.py train --resume checkpoint.pt ...` (for example, a second phase toward GTS3's
  7.4B tokens). `scripts/glue_finetune.py --gts` and `scripts/sys1_train.py --gts` take it directly and pick
  ModernBERT's tokenizer from the vocabulary size.
- `binarized.pt` (172 MB): ternary weights as 2-bit codes plus group scales, other tensors in float32. Load as in
  `checkpoints/bert110m/README.md`.
- `examples.json`: fill-mask samples at the end of training.

#!/bin/bash
# GTS-KD-110: GTS3's architecture (14 x 768 mixed forest: bank of 32 trees / 8 heads / state 16, 4 deep trees of depth
# 9) retrained in ModernBERT's tokenizer and distilled from ModernBERT-base, at GTS3's length (~7.4B tokens).
#   data   FineWeb-Edu sample/10BT shards 000-009 (~7.5B tokens, no repeats), scripts/mohawk_distill.py prep
#   train  scripts/bert_pretrain.py: masked LM at 30% masking (ModernBERT's), loss 0.75 * T^2 * KL(teacher || student)
#          at T = 2 + 0.25 * CE at the masked positions; token embeddings (tied output) start from ModernBERT-base's;
#          peak lr 1.5e-3 (GTS3 phase 1), warmup 2,000 steps, cosine to 10% over --minutes; batch 64 x 512
# Outputs (persistent volume): $O/run/{checkpoint.pt, binarized.pt, result.json, train.log, examples.json}.
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 HF_HOME=/root/hf TOKENIZERS_PARALLELISM=false HF_HUB_ENABLE_HF_TRANSFER=0
set -x
O=/workspace/kd110; R=$O/run; D=/root/kd_data; mkdir -p $R $D
python3 -m pip install -q einops packaging tokenizers huggingface_hub datasets pytest pyarrow "transformers>=4.48" 2>&1 | tail -1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader; df -h /root /workspace | tail -2; free -g | head -2
FILES=$(for i in 0 1 2 3 4 5 6 7 8 9; do printf "sample/10BT/%03d_00000.parquet " $i; done)
W=$(( $(nproc) < 14 ? $(nproc) : 14 ))
for i in 1 2; do [ -f $D/meta.json ] || timeout 120m python3 scripts/mohawk_distill.py prep --out $D --workers $W --files $FILES; done
cat $D/meta.json
python3 scripts/bert_pretrain.py train --data $D --out $R --minutes ${MINUTES:-960} --teacher answerdotai/ModernBERT-base \
  --init-embeddings --mask-prob 0.3 --lr 1.5e-3 --warmup 2000 --ckpt-minutes 30 2>&1 | tee -a $R/train.log
echo "=== DONE $(date -u +%T) ==="

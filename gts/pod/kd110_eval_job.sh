#!/bin/bash
# Evaluate GTS-KD-110 (pod/kd110_job.sh) exactly as GTS3 was: GLUE dev (results/glue/glue_gts3.json's settings) and
# Laya-style typed decisions (results/sys1/sys1_gts3.json's). Usage: bash pod/kd110_eval_job.sh [CHECKPOINT]
export PYTHONUNBUFFERED=1 HF_HOME=/root/hf TOKENIZERS_PARALLELISM=false HF_HUB_ENABLE_HF_TRANSFER=0
set -x
O=/workspace/kd110; CK=${1:-$O/run/checkpoint.pt}; E=$O/eval; mkdir -p $E
python3 -m pip install -q einops packaging tokenizers huggingface_hub datasets pyarrow scipy "transformers>=4.48" 2>&1 | tail -1
python3 scripts/glue_finetune.py --gts $CK --tasks sst2 mrpc rte qnli mnli cola stsb --epochs 3 --batch-size 32 --lr 1e-4 \
  --ternary-lr 3e-4 --max-len 128 --max-train 100000 --out $E/glue_kd110.json 2>&1 | tee -a $E/glue.log
python3 scripts/sys1_train.py --gts $CK --general 500000 --epochs 1 --fit-epochs 5 --lr 1e-4 --ternary-lr 3e-4 \
  --head-layers 2 --out $E/sys1_kd110.json 2>&1 | tee -a $E/sys1.log
echo "=== EVAL DONE $(date -u +%T) ==="

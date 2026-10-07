#!/bin/bash
# EVA: MOHAWK distillation of ModernBERT-large into GTS-L, 3.0B tokens (80M / 300M / 2.62B), full precision until the
# last 14% of Stage 3, then ternary. Usage: bash eva_job.sh OUT_DIR   (OUT_DIR persists: data, checkpoints, logs).
# Rerunning with the same OUT_DIR resumes from the last checkpoint (every 30 minutes), e.g. after a spot preemption.
# A new job with a new OUT_DIR resumes too when given the old job's output folder: bash eva_job.sh OUT_DIR PREVIOUS_OUT.
set -x
OUT=${1:?usage: eva_job.sh OUT_DIR [PREVIOUS_OUT_DIR]}
PREV=${2:-}
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 HF_HOME=${HF_HOME:-/tmp/hf} TOKENIZERS_PARALLELISM=false
mkdir -p "$OUT/data" "$OUT/run"
if [ -n "$PREV" ] && [ ! -f "$OUT/run/state.pt" ]; then   # continue a preempted job: its data and checkpoints
  cp -r "$PREV/data/." "$OUT/data/" 2>/dev/null; cp -r "$PREV/run/." "$OUT/run/" 2>/dev/null
  echo "copied the previous job's output from $PREV"; ls -la "$OUT/run"
fi
python3 -c "import torch, sys; v = tuple(int(x) for x in torch.__version__.split('+')[0].split('.')[:2]); sys.exit(v < (2, 4))" \
  || python3 -m pip install -q "torch==2.8.0" --index-url https://download.pytorch.org/whl/cu126
python3 -m pip install -q einops packaging tokenizers huggingface_hub datasets pytest pyarrow "transformers>=4.48" 2>&1 | tail -1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 -m pytest -q tests/models/test_gts_l.py 2>&1 | tail -3
if [ ! -f "$OUT/data/meta.json" ]; then
  python3 scripts/mohawk_distill.py prep --out "$OUT/data" --workers "$(nproc)" \
    --files sample/10BT/000_00000.parquet sample/10BT/001_00000.parquet sample/10BT/002_00000.parquet \
            sample/10BT/003_00000.parquet sample/10BT/004_00000.parquet
fi
LOCAL=${LOCAL_DATA:-/tmp/eva_data}; mkdir -p "$LOCAL"   # random reads go to local disk, not the mounted output
for f in train.bin val.bin meta.json; do [ -f "$LOCAL/$f" ] || cp "$OUT/data/$f" "$LOCAL/$f"; done
python3 scripts/mohawk_distill.py train --data "$LOCAL" --out "$OUT/run" --price-per-hour "${PRICE:-0.90}" 2>&1 | tee -a "$OUT/run/train.log"
echo "=== DONE $(date -u +%T) ==="

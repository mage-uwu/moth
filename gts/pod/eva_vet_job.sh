# EVA vet on one A100, at pilot 1's budget: EVA's recipe (full precision until the last 14% of Stage 3, then a late
# ternary ramp; trees' Stage 2 term weighted 2x; a decaying layer-by-layer term early in Stage 3) with pilot 1's stage
# sizes (14M / 47M / 125M tokens), so the final ternary evaluation compares directly. Training is killed once 25
# minutes in (Stage 2) and restarted, to exercise resume on the GPU.
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 HF_HOME=/root/hf
set -x
O=/workspace/out; D=/root/fwe; R=$O/eva_vet; mkdir -p $O $R
python3 -m pip install -q einops packaging tokenizers huggingface_hub datasets pytest pyarrow "transformers>=4.48" 2>&1 | tail -1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 -m pytest -q tests/models/test_gts_l.py 2>&1 | tail -3
[ -f $D/meta.json ] || python3 scripts/mohawk_distill.py prep --out $D --workers 14
ARGS="--data $D --out $R --stage1-tokens 14e6 --stage2-tokens 47e6 --stage3-tokens 125e6 --ckpt-minutes 5 --eval-minutes 20 --log-every 50 --price-per-hour 1.59"
timeout 25m python3 scripts/mohawk_distill.py train $ARGS 2>&1 | tee -a $R/train.log
echo "=== killed after 25 minutes; resuming ==="
python3 scripts/mohawk_distill.py train $ARGS 2>&1 | tee -a $R/train.log
echo "=== DONE $(date -u +%T) ==="

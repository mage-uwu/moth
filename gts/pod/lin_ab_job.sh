# Linear-path A/B on one A100: Stage 2 alone (47M tokens, pilot 1's budget) for GTS-L's trees with and without a
# rank-128 ternary linear path initialised from the teacher MLPs' least-squares affine fits. The tree term only sees
# the teacher's MLP inputs, so Stage 1 is skipped; per-layer held-out errors land in each log.json (stage2_layers).
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 HF_HOME=/root/hf TOKENIZERS_PARALLELISM=false HF_HUB_ENABLE_HF_TRANSFER=0
set -x
O=/workspace/out; D=/root/fwe; mkdir -p $O
python3 -m pip install -q einops packaging tokenizers huggingface_hub datasets pytest pyarrow "transformers>=4.48" 2>&1 | tail -1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 -m pytest -q -rf --tb=short tests/models/test_gts_l.py 2>&1 | tail -25
for i in 1 2; do [ -f $D/meta.json ] || timeout 25m python3 scripts/mohawk_distill.py prep --out $D --workers 14; done
COMMON="--data $D --stage1-tokens 0 --stage2-tokens 47e6 --stage3-tokens 0 --no-downstream --ckpt-minutes 60 --eval-minutes 1000 --log-every 50 --price-per-hour 1.59"
python3 scripts/mohawk_distill.py train $COMMON --out $O/ab_linear --config '{"linear_rank": 128}' 2>&1 | tee -a $O/ab_linear.log
python3 scripts/mohawk_distill.py train $COMMON --out $O/ab_trees 2>&1 | tee -a $O/ab_trees.log
echo "=== DONE $(date -u +%T) ==="

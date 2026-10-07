# MOHAWK pilot: ModernBERT-large -> GTS-L (~405M, ternary, attention-free), on one A100. FineWeb-Edu in ModernBERT's
# tokenizer, then Stages 1-3 for wall-clock budgets (MH_S1/MH_S2/MH_S3 minutes). Outputs in /workspace/out/mohawk.
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 HF_HOME=/root/hf
set -x
O=/workspace/out; D=/root/fwe; R=$O/mohawk; mkdir -p $O $R
python3 -m pip install -q einops packaging tokenizers huggingface_hub datasets pytest pyarrow "transformers>=4.48" 2>&1 | tail -1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 -m pytest -q tests/models/test_gts_l.py 2>&1 | tail -3
[ -f $D/meta.json ] || python3 scripts/mohawk_distill.py prep --out $D --workers 14
python3 scripts/mohawk_distill.py train --data $D --out $R --stage1-minutes ${MH_S1:-15} --stage2-minutes ${MH_S2:-35} \
  --stage3-minutes ${MH_S3:-100} --price-per-hour ${MH_PRICE:-1.59} 2>&1 | tee $R/train.log
echo "=== DONE $(date -u +%T) ==="

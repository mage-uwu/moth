# Sparse deep trees on an A100: correctness tests on CUDA, then dense against sparse (one layer, the whole encoder).
export PYTHONUNBUFFERED=1 HF_HOME=/root/hf
set -x
python3 -m pip install -q einops packaging tokenizers huggingface_hub datasets pytest 2>&1 | tail -1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cat checkpoints/bert110m/phase3/binarized.pt.part* > /root/gts3.pt
python3 -m pytest -q tests/modules/test_gts_sparse.py tests/modules/test_gts_route.py 2>&1 | tail -5
python3 scripts/bench_sparse.py --ckpt /root/gts3.pt
echo "=== DONE $(date -u +%T) ==="

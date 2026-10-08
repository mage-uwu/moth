# GPU serving throughput of GTS-L and ModernBERT-large (scripts/bench_gtsl_gpu.py), for the CPU runtime comparison.
export PYTHONUNBUFFERED=1 HF_HOME=/root/hf TOKENIZERS_PARALLELISM=false HF_HUB_ENABLE_HF_TRANSFER=0
set -x
O=/workspace/out; mkdir -p $O
python3 -m pip install -q einops packaging tokenizers huggingface_hub "transformers>=4.48" 2>&1 | tail -1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python3 scripts/bench_gtsl_gpu.py --out $O/gtsl_gpu_bench.json 2>&1 | tee -a $O/gpu_bench.log
echo "=== DONE $(date -u +%T) ==="

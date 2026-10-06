# Kernel benchmarks on a second GPU pod, run the same way as pod/job.sh (clone of this branch, served /workspace/out).
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
set -x
O=/workspace/out; tag=$(git rev-parse --short HEAD)
python3 -m pip install -q einops packaging 2>&1 | tail -1
python3 -c "import torch, triton; print(torch.__version__, triton.__version__, torch.cuda.get_device_name(0))"
python3 -m pytest tests/modules/test_gts_scan.py -q 2>&1 | tail -5 | tee $O/bench_$tag.log
python3 scripts/bench_scan.py --layer --profile 2>&1 | tee -a $O/bench_$tag.log
echo "=== BENCH DONE $(date -u +%T) ===" | tee -a $O/bench_$tag.log

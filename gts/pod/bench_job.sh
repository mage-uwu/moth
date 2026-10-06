# Kernel tests and timing on a GPU pod, run the same way as pod/job.sh (clone of this branch, served /workspace/out).
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
set -x
O=/workspace/out; tag=$(git rev-parse --short HEAD); L=$O/bench_$tag.log
python3 -m pip install -q einops packaging 2>&1 | tail -1
python3 -m pytest tests/modules/test_gts_scan.py tests/modules/test_gts_route.py tests/modules/test_gts.py -q -rf --tb=line 2>&1 | tail -15 | tee $L
for flags in "--no-scan-kernel --no-route-kernel" "--no-route-kernel" "" "--no-checkpoint"; do
  python3 scripts/profile_step.py --arch mixed $flags 2>&1 | grep -v Warning | tee -a $L
done
python3 scripts/profile_step.py --arch mixed --seq-len 2048 --batch-size 2 --top 0 2>&1 | grep tokens/s | tee -a $L
python3 scripts/profile_step.py --arch mamba2 --top 12 2>&1 | grep -v Warning | tee -a $L
echo "=== BENCH DONE $(date -u +%T) ===" | tee -a $L

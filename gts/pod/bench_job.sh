# Kernel tests and timing on a GPU pod, run the same way as pod/job.sh (clone of this branch, served /workspace/out).
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
set -x
O=/workspace/out; tag=$(git rev-parse --short HEAD); L=$O/bench_$tag.log
python3 -m pip install -q einops packaging 2>&1 | tail -1
python3 -m pytest tests/modules/test_gts_scan.py tests/modules/test_gts_route.py tests/modules/test_ternary_fused.py tests/modules/test_gts.py tests/modules/test_gts_ternary.py -q -rf --tb=line 2>&1 | tail -15 | tee $L
P="python3 scripts/profile_step.py --arch bert --no-checkpoint"
# the bidirectional model as it would have trained before this work: quadratic bank, walk-and-scatter trees, full head, fp32
$P --no-scan-kernel --no-route-kernel --full-head --top 0 2>&1 | grep tokens/s | tee -a $L
$P --no-route-kernel --full-head --top 0 2>&1 | grep tokens/s | tee -a $L
$P --full-head --top 0 2>&1 | grep tokens/s | tee -a $L
$P --top 0 2>&1 | grep tokens/s | tee -a $L
$P --fused-adam --amp --top 0 2>&1 | grep tokens/s | tee -a $L
$P --fused-adam --amp --compile --top 25 2>&1 | grep -vi warn | tail -27 | tee -a $L
$P --fused-adam --amp --compile --route-ste --top 0 2>&1 | grep tokens/s | tee -a $L
$P --fused-adam --amp --compile --seq-len 2048 --batch-size 2 --top 0 2>&1 | grep tokens/s | tee -a $L
$P --no-scan-kernel --no-route-kernel --full-head --seq-len 2048 --batch-size 2 --top 0 2>&1 | grep -E "tokens/s|Error" | tail -2 | tee -a $L
echo "=== BENCH DONE $(date -u +%T) ===" | tee -a $L

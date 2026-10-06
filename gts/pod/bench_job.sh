# Kernel tests and timing on a GPU pod, run the same way as pod/job.sh (clone of this branch, served /workspace/out).
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
set -x
O=/workspace/out; tag=$(git rev-parse --short HEAD); L=$O/bench_$tag.log
python3 -m pip install -q einops packaging datasets tiktoken 2>&1 | tail -1
python3 -m pytest tests/modules/test_gts_scan.py tests/modules/test_gts_route.py tests/modules/test_gts.py -q -rf --tb=line 2>&1 | tail -15 | tee $L
P="python3 scripts/profile_step.py --arch mixed --no-checkpoint"
$P --top 0 2>&1 | grep tokens/s | tee -a $L
$P --fused-adam --top 0 2>&1 | grep tokens/s | tee -a $L
$P --fused-adam --amp --top 30 2>&1 | grep -v Warning | tee -a $L
$P --fused-adam --amp --compile --top 30 2>&1 | grep -vi warn | tail -32 | tee -a $L
# quality: 300 steps on FineWeb, float32/TF32 against bf16, everything else equal
D=/workspace/data/fineweb
[ -f $D/meta.json ] || python3 scripts/prepare_fineweb.py --out $D --train-tokens 3000000 --val-tokens 200000 2>&1 | tail -1
for amp in "" "--amp"; do
  echo "--- quality ${amp:-fp32}" | tee -a $L
  python3 scripts/lm_run.py --arch mixed --data $D --out /tmp/q$amp --steps 300 --warmup 50 --eval-every 150 --eval-batches 10 --no-checkpoint --no-export $amp 2>&1 | grep -E "^step|Error" | tee -a $L
done
echo "=== BENCH DONE $(date -u +%T) ===" | tee -a $L

# Kernel tests and timing on a GPU pod, run the same way as pod/job.sh (clone of this branch, served /workspace/out).
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
set -x
O=/workspace/out; tag=$(git rev-parse --short HEAD); L=$O/bench_$tag.log
python3 -m pip install -q einops packaging datasets tiktoken 2>&1 | tail -1
python3 -m pytest tests/modules/test_gts_scan.py tests/modules/test_gts_route.py tests/modules/test_ternary_fused.py tests/modules/test_gts.py -q -rf --tb=line 2>&1 | tail -15 | tee $L
P="python3 scripts/profile_step.py --arch mixed --no-checkpoint --fused-adam --amp"
$P --top 0 2>&1 | grep tokens/s | tee -a $L
$P --compile --no-fused-quant --top 0 2>&1 | grep tokens/s | tee -a $L
$P --compile --top 30 2>&1 | grep -vi warn | tail -32 | tee -a $L
$P --compile --compile-mode max-autotune-no-cudagraphs --top 20 2>&1 | grep -vi warn | grep -E "tokens/s|GPU busy|ms " | head -22 | tee -a $L
python3 scripts/profile_step.py --arch mixed --fused-adam --amp --compile --top 0 2>&1 | grep tokens/s | tee -a $L   # with checkpointing
D=/workspace/data/fineweb
[ -f $D/meta.json ] || python3 scripts/prepare_fineweb.py --out $D --train-tokens 3000000 --val-tokens 200000 2>&1 | tail -1
echo "--- quality amp + compile" | tee -a $L
python3 scripts/lm_run.py --arch mixed --data $D --out /tmp/qc --steps 300 --warmup 50 --eval-every 150 --eval-batches 10 --no-checkpoint --no-export --amp --compile 2>&1 | grep -E "^step|Error" | tee -a $L
echo "=== BENCH DONE $(date -u +%T) ===" | tee -a $L

# Kernel benchmarks on a second GPU pod, run the same way as pod/job.sh (clone of this branch, served /workspace/out).
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8
set -x
O=/workspace/out; tag=$(git rev-parse --short HEAD)
python3 -m pip install -q einops packaging 2>&1 | tail -1
python3 -c "import torch, triton; print(torch.__version__, triton.__version__, torch.cuda.get_device_name(0))"
python3 -m pytest tests/modules/test_gts_scan.py -q -rf --tb=line 2>&1 | tail -25 | tee $O/bench_$tag.log
python3 scripts/bench_scan.py --layer --profile 2>&1 | tee -a $O/bench_$tag.log
echo "=== BENCH DONE $(date -u +%T) ===" | tee -a $O/bench_$tag.log
# End to end: training steps of the 0.5B mixed forest with the scan and with the quadratic context, at equal tokens.
D=/workspace/data/fineweb
[ -f $D/meta.json ] || python3 -m pip install -q datasets tiktoken 2>&1 | tail -1
[ -f $D/meta.json ] || python3 scripts/prepare_fineweb.py --out $D --train-tokens 2000000 --val-tokens 100000
for cfg in "512 8" "2048 2"; do set -- $cfg
  for flag in "" "--no-scan-kernel"; do
    echo "--- mixed forest 0.5B, seq $1 batch $2 ${flag:-scan}" | tee -a $O/bench_$tag.log
    python3 scripts/lm_run.py --arch mixed --data $D --out /tmp/e2e --seq-len $1 --batch-size $2 --steps 30 --warmup 5 --eval-every 1000 --eval-batches 1 --log-every 10 --no-export $flag 2>&1 | grep -E "s/step|peak|Error|error" | tee -a $O/bench_$tag.log
  done
done
echo "=== E2E DONE $(date -u +%T) ===" | tee -a $O/bench_$tag.log

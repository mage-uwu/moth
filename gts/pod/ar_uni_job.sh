# GTS-Uni-AR: the causal ternary GTS mixed forest (110M-class: width 768, 14 layers) with 3 shared-weight passes and
# pause tokens (2 after every 32 real tokens, from pass 2),
# trained on FineWeb GPT-2 tokens by scripts/ar_pretrain.py with the masked-LM path's conveniences. Not run yet.
# Needs END_UTC. With INIT_FROM (a one-pass ar_pretrain checkpoint.pt) it warm-starts from it with split learning
# rates; otherwise it trains from scratch. Logs and outputs in /workspace/out, served read-only on port 8888.
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 HF_HOME=/root/hf
END=$(date -d "${END_UTC:?set END_UTC} UTC" +%s)
set -x
O=/workspace/out; R=/workspace/ar_uni; D=/workspace/fineweb
mkdir -p $O $R
for f in result.json samples.txt checkpoint.pt binarized.pt model.bin; do ln -sf $R/$f $O/ar_$f; done
python3 -m pip install -q einops packaging datasets tiktoken 2>&1 | tail -1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
[ -f $D/meta.json ] || python3 scripts/prepare_fineweb.py --out $D --train-tokens ${TRAIN_TOKENS:-2000000000} --val-tokens 2000000
LEFT=$(( (END - $(date +%s)) / 60 ))
if [ -n "$INIT_FROM" ]; then
  ARGS="--init-from $INIT_FROM --lr 3e-5 --new-param-lr 1e-3 --warmup 500"
else
  ARGS="--lr 1e-3 --warmup 1000"
fi
python3 scripts/ar_pretrain.py --data $D --out $R --minutes $LEFT --loops 3 --loop-probs 0.1,0.2,0.7 $ARGS 2>&1 | tee /root/train.log
if grep -q "InductorError" /root/train.log; then
  LEFT=$(( (END - $(date +%s)) / 60 ))
  python3 scripts/ar_pretrain.py --data $D --out $R --minutes $LEFT --loops 3 --loop-probs 0.1,0.2,0.7 $ARGS --no-compile
fi
gcc -O3 -march=native -ffast-math -funroll-loops -fopenmp kernel/ar_bench.c -o kernel/ar_bench -lm && \
  OMP_NUM_THREADS=4 kernel/ar_bench $R/model.bin 2000
echo "=== ALL DONE $(date -u +%T) ==="

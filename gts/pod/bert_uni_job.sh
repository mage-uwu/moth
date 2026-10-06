# GTS-Uni: the 110M GTS masked LM with depth recurrence. Split learning rates: the shared GTS3 weights at 3e-5 (a
# fresh optimizer at 3e-4 knocked the converged weights off their minimum, +0.16 validation loss, in a first try), the
# new pass embeddings, gates and latents at 1e-3, so the extra passes switch on fast. The 14-layer stack runs 3 times with shared weights (pass
# embeddings, per-channel gates starting at zero, 16 latent scratch tokens from pass 2), started from GTS3 (phase 3's final
# weights), so step 0 computes exactly what GTS3 does. Each step trains with 1, 2 or 3 passes (10/20/70%), so the
# weights also work with fewer passes (full CPU speed at 1). Data: all 40 Wikipedia shards again. No network volume:
# the phase 3 checkpoint comes from this repo's parts. Logs and outputs in /workspace/out on port 8888.
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 HF_HOME=/root/hf
END=$(date -d "${END_UTC:?set END_UTC} UTC" +%s)
set -x
O=/workspace/out; R=/workspace/bert110m_uni; D=/root/wiki_all
mkdir -p $O $R
for f in result.json examples.json checkpoint.pt binarized.pt; do ln -sf $R/$f $O/uni_$f; done
python3 -m pip install -q einops packaging tokenizers huggingface_hub pyarrow 2>&1 | tail -1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
CK=/root/p3/checkpoint.pt; mkdir -p /root/p3
cat checkpoints/bert110m/phase3/checkpoint.pt.part* > $CK
(cd /root/p3 && grep " checkpoint.pt" /root/moth/gts/checkpoints/bert110m/phase3/SHA256SUMS | sha256sum -c) || exit 1
[ -f $D/meta.json ] || python3 scripts/bert_pretrain.py prep --out $D --shard-offset 0 --train-tokens 6400000000 --val-tokens 2000000 --tmp-dir /root/shards --workers 12
LEFT=$(( (END - $(date +%s)) / 60 ))
echo "minutes left for training: $LEFT"
ARGS="--data $D --out $R --init-from $CK --loops 3 --latent-tokens 16 --loop-probs 0.1,0.2,0.7 --lr 3e-5 --new-param-lr 1e-3 --warmup 500 --eval-every 4000"
python3 scripts/bert_pretrain.py train $ARGS --minutes $LEFT 2>&1 | tee /root/train.log
EXTRA=""
grep -q "OutOfMemoryError" /root/train.log && EXTRA="--checkpoint-loops"   # three passes' activations did not fit
grep -q "InductorError" /root/train.log && EXTRA="$EXTRA --no-compile"     # torch.compile failed: run uncompiled
if [ -n "$EXTRA" ]; then
  LEFT=$(( (END - $(date +%s)) / 60 ))
  python3 scripts/bert_pretrain.py train $ARGS --minutes $LEFT $EXTRA 2>&1 | tee /root/train2.log
  if grep -q "InductorError" /root/train2.log; then LEFT=$(( (END - $(date +%s)) / 60 )); python3 scripts/bert_pretrain.py train $ARGS --minutes $LEFT --no-compile; fi
fi
echo "=== ALL DONE $(date -u +%T) ==="

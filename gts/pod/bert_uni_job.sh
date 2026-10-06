# GTS-Uni: the 110M GTS masked LM with depth recurrence. The 14-layer stack runs 3 times with shared weights (pass
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
ARGS="--data $D --out $R --init-from $CK --loops 3 --latent-tokens 16 --loop-probs 0.1,0.2,0.7 --lr 3e-4 --warmup 1000 --eval-every 2000"
python3 scripts/bert_pretrain.py train $ARGS --minutes $LEFT 2>&1 | tee /root/train.log
if grep -q "OutOfMemoryError" /root/train.log; then   # three passes' activations did not fit: recompute passes 2 and 3
  LEFT=$(( (END - $(date +%s)) / 60 ))
  python3 scripts/bert_pretrain.py train $ARGS --minutes $LEFT --checkpoint-loops
fi
echo "=== ALL DONE $(date -u +%T) ==="

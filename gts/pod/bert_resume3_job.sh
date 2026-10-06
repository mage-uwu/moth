# Phase 3 of the ~110M BERT-style run: resume from phase 2's final float checkpoint (weights, AdamW state, step,
# sampler) on Wikipedia shards 20 to 39, which phases 1 and 2 never saw, re-warming to half phase 2's peak (3.75e-4)
# and decaying again, for the same training time as each earlier phase. The volume is nearly full, so this phase's
# tokenised data lives on the pod's disk (and is redone if the pod restarts); checkpoints go to the volume. Logs and
# outputs are linked into /workspace/out, served read-only on port 8888.
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 HF_HOME=/root/hf
END=$(date -d "${END_UTC:?set END_UTC} UTC" +%s)   # hard end of phase 3
set -x
O=/workspace/out; R=/workspace/bert110m_p3; D=/root/wiki3
mkdir -p $O $R
for f in result.json examples.json checkpoint.pt binarized.pt; do ln -sf $R/$f $O/p3_$f; done
python3 -m pip install -q einops packaging tokenizers huggingface_hub pyarrow 2>&1 | tail -1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
df -h /workspace /root | tail -2
[ -f $D/meta.json ] || python3 scripts/bert_pretrain.py prep --out $D --shard-offset 20 --train-tokens 2900000000 --val-tokens 2000000 --tmp-dir /root/shards
cat $D/meta.json
LEFT=$(( (END - $(date +%s)) / 60 ))
echo "minutes left for training: $LEFT"
python3 scripts/bert_pretrain.py train --data $D --out $R --minutes $LEFT --resume /workspace/bert110m_p2/checkpoint.pt \
  --lr 3.75e-4 --rewarm 1000 --eval-every 2000
echo "=== ALL DONE $(date -u +%T) ==="

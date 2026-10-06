# Phase 2 of the ~110M BERT-style run: resume from phase 1's final float checkpoint (weights, AdamW state, step,
# sampler) on Wikipedia shards phase 1 never saw, re-warming to half the original peak and decaying again, for the
# same training time as phase 1. Data and checkpoints on the network volume; logs and outputs linked into
# /workspace/out, served read-only on port 8888.
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 HF_HOME=/root/hf
END=$(date -d "${END_UTC:-2026-10-06 11:45:00} UTC" +%s)   # hard end of phase 2
set -x
O=/workspace/out; R=/workspace/bert110m_p2; D=/workspace/wiki2
mkdir -p $O $R
for f in result.json examples.json checkpoint.pt binarized.pt; do ln -sf $R/$f $O/p2_$f; done
python3 -m pip install -q einops packaging tokenizers huggingface_hub pyarrow 2>&1 | tail -1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
df -h /workspace | tail -1
[ -f $D/meta.json ] || python3 scripts/bert_pretrain.py prep --out $D --shard-offset 7 --train-tokens 1800000000 --val-tokens 2000000 --tmp-dir /root/shards
cat $D/meta.json
LEFT=$(( (END - $(date +%s)) / 60 ))
echo "minutes left for training: $LEFT"
python3 scripts/bert_pretrain.py train --data $D --out $R --minutes $LEFT --resume /workspace/bert110m/checkpoint.pt \
  --lr 7.5e-4 --rewarm 1000 --eval-every 2000
echo "=== ALL DONE $(date -u +%T) ==="

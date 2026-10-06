# Phase 3 of the ~110M BERT-style run: plain masked-LM pretraining resumed from phase 2's final float checkpoint
# (weights, AdamW state, step, sampler) for a $10 budget, about twice phase 2's length. Data: all 40 Wikipedia training
# shards (0 to 39, about 4B tokens, so about one pass, with shards 0 to 19 seen before in phases 1 and 2). The learning
# rate re-warms to 5e-4 over 2,000 steps and decays to 5e-5. (A distillation leg from bert-base-uncased,
# pod/bert_distill_job.sh, was stopped part-way and is not continued; see results/bert110m_kd.)
# No network volume: the phase 2 checkpoint is rebuilt from its parts in this repo and the data tokenised on the pod's
# disk. Logs and outputs are in /workspace/out, served read-only on port 8888.
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 HF_HOME=/root/hf
END=$(date -d "${END_UTC:?set END_UTC} UTC" +%s)   # hard end of phase 3
set -x
O=/workspace/out; R=/workspace/bert110m_p3; D=/root/wiki_all
mkdir -p $O $R
for f in result.json examples.json checkpoint.pt binarized.pt; do ln -sf $R/$f $O/p3_$f; done
python3 -m pip install -q einops packaging tokenizers huggingface_hub pyarrow 2>&1 | tail -1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
df -h /root | tail -1
CK=/root/p2/checkpoint.pt; mkdir -p /root/p2
cat checkpoints/bert110m/phase2/checkpoint.pt.part* > $CK
(cd /root/p2 && grep " checkpoint.pt" /root/moth/gts/checkpoints/bert110m/phase2/SHA256SUMS | sha256sum -c) || exit 1
[ -f $D/meta.json ] || python3 scripts/bert_pretrain.py prep --out $D --shard-offset 0 --train-tokens 6400000000 --val-tokens 2000000 --tmp-dir /root/shards --workers 12
cat $D/meta.json
LEFT=$(( (END - $(date +%s)) / 60 ))
echo "minutes left for training: $LEFT"
python3 scripts/bert_pretrain.py train --data $D --out $R --minutes $LEFT --resume $CK \
  --lr 5e-4 --rewarm 2000 --eval-every 4000
echo "=== ALL DONE $(date -u +%T) ==="

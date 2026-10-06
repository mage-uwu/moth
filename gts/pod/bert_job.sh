# ~110M-parameter BERT-style GTS on English Wikipedia, for a fixed budget. Data and checkpoints on the network
# volume at /workspace; logs and outputs linked into /workspace/out, which the pod serves read-only on port 8888.
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 HF_HOME=/root/hf
TOTAL_MIN=${TOTAL_MIN:-168}   # from job start to the end of saving; the pod has run about 4 minutes by then
T0=$(date +%s)
set -x
O=/workspace/out; R=/workspace/bert110m; D=/workspace/wiki
mkdir -p $O $R
ln -sf $R/result.json $O/result.json; ln -sf $R/examples.json $O/examples.json
ln -sf $R/checkpoint.pt $O/checkpoint.pt; ln -sf $R/binarized.pt $O/binarized.pt
python3 -m pip install -q einops packaging tokenizers huggingface_hub pyarrow 2>&1 | tail -1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
[ -f $D/meta.json ] || python3 scripts/bert_pretrain.py prep --out $D --train-tokens 1000000000 --val-tokens 2000000
cat $D/meta.json
LEFT=$(( TOTAL_MIN - ($(date +%s) - T0) / 60 ))
echo "minutes left for training: $LEFT"
python3 scripts/bert_pretrain.py train --data $D --out $R --minutes $LEFT --lr 1.5e-3 --eval-every 2000
echo "=== ALL DONE $(date -u +%T) ==="

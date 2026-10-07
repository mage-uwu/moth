# GLUE dev evaluation, one harness for GTS and BERT-family baselines (scripts/glue_finetune.py): GTS3, GTS-Uni at
# 1 and 3 passes, BERT-base, DistilBERT and TinyBERT-4L, all fine-tuned with the same data, schedule and budget.
# MNLI, QNLI and QQP are subsampled (--max-train) to keep the job to a few A100 hours. Needs the GTS checkpoints in
# this repo (GTS-Uni's once saved); results in /workspace/out/glue_*.json.
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 HF_HOME=/root/hf
set -x
O=/workspace/out; mkdir -p $O
python3 -m pip install -q einops packaging tokenizers huggingface_hub datasets transformers 2>&1 | tail -1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
mkdir -p /root/ck
cat checkpoints/bert110m/phase3/binarized.pt.part* > /root/ck/gts3.pt
[ -d checkpoints/bert110m/uni ] && cat checkpoints/bert110m/uni/binarized.pt.part* > /root/ck/uni.pt
T="sst2 mrpc rte qnli mnli cola stsb"
COMMON="--tasks $T --epochs 3 --max-train 100000 --batch-size 32 --max-len 128"
python3 scripts/glue_finetune.py --gts /root/ck/gts3.pt $COMMON --lr 1e-4 --out $O/glue_gts3.json
if [ -f /root/ck/uni.pt ]; then
  python3 scripts/glue_finetune.py --gts /root/ck/uni.pt --loops 1 $COMMON --lr 1e-4 --out $O/glue_uni_1pass.json
  python3 scripts/glue_finetune.py --gts /root/ck/uni.pt --loops 3 $COMMON --lr 1e-4 --out $O/glue_uni_3pass.json
fi
for m in google-bert/bert-base-uncased distilbert/distilbert-base-uncased huawei-noah/TinyBERT_General_4L_312D; do
  python3 scripts/glue_finetune.py --hf $m $COMMON --lr 3e-5 --out $O/glue_$(basename $m).json
done
echo "=== ALL DONE $(date -u +%T) ==="

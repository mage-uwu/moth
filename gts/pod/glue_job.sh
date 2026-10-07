# GLUE dev evaluation, one harness for GTS and BERT-family baselines (scripts/glue_finetune.py): GTS3, GTS-Uni at
# 1 and 3 passes, BERT-base, DistilBERT and TinyBERT-4L, all fine-tuned with the same data, schedule and budget.
# MNLI, QNLI and QQP are subsampled (--max-train) to keep the job to a few A100 hours. Results in
# /workspace/out/glue_*.json, every GTS task's best-epoch classifier in /workspace/out/models/.
#
# GTS models start from their float checkpoints and the ternary latents get their own learning rate (--ternary-lr):
# from a binarized checkpoint at 1e-4 no ternary code flips in a GLUE run, which fine-tunes only the floats (the first
# run, results/glue_adapter_only/). The ternary learning rate is picked once, on GTS3 and the four small tasks
# (RTE, MRPC, STS-B, CoLA), from 3e-4 and 1e-3, and then used for every GTS run (TLR overrides the sweep).
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 HF_HOME=/root/hf
set -x
O=/workspace/out; mkdir -p $O /root/ck
python3 -m pip install -q einops packaging tokenizers huggingface_hub datasets transformers 2>&1 | tail -1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
[ -f /root/ck/gts3_float.pt ] || cat checkpoints/bert110m/phase3/checkpoint.pt.part* > /root/ck/gts3_float.pt
[ -d checkpoints/bert110m/uni ] && [ ! -f /root/ck/uni_float.pt ] && cat checkpoints/bert110m/uni/checkpoint.pt.part* > /root/ck/uni_float.pt
T="sst2 mrpc rte qnli mnli cola stsb"
COMMON="--epochs 3 --max-train 100000 --batch-size 32 --max-len 128"
if [ -z "$TLR" ]; then
  for t in 3e-4 1e-3; do
    python3 scripts/glue_finetune.py --gts /root/ck/gts3_float.pt --tasks rte mrpc stsb cola $COMMON --lr 1e-4 \
      --ternary-lr $t --out $O/glue_sweep_gts3_tlr$t.json
  done
  TLR=$(python3 -c "
import json
r = {t: json.load(open('$O/glue_sweep_gts3_tlr%s.json' % t))['average'] for t in ('3e-4', '1e-3')}
print(max(r, key=r.get))")
fi
echo "ternary learning rate: $TLR" | tee $O/ternary_lr.txt
python3 scripts/glue_finetune.py --gts /root/ck/gts3_float.pt --tasks $T $COMMON --lr 1e-4 --ternary-lr $TLR \
  --out $O/glue_gts3.json --save-dir $O/models/glue_gts3
if [ -f /root/ck/uni_float.pt ]; then
  python3 scripts/glue_finetune.py --gts /root/ck/uni_float.pt --loops 3 --tasks $T $COMMON --lr 1e-4 --ternary-lr $TLR \
    --out $O/glue_uni_3pass.json --save-dir $O/models/glue_uni_3pass
  python3 scripts/glue_finetune.py --gts /root/ck/uni_float.pt --loops 1 --tasks $T $COMMON --lr 1e-4 --ternary-lr $TLR \
    --out $O/glue_uni_1pass.json --save-dir $O/models/glue_uni_1pass
fi
for m in google-bert/bert-base-uncased distilbert/distilbert-base-uncased huawei-noah/TinyBERT_General_4L_312D; do
  python3 scripts/glue_finetune.py --hf $m --tasks $T $COMMON --lr 3e-5 --out $O/glue_$(basename $m).json
done
echo "=== GLUE DONE $(date -u +%T) ==="

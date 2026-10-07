# Laya / Jev regime probe on the LocalLLaMA typed-decisions benchmark (scripts/decide_probe.py): each model is trained
# on 500K general typed decisions (tasksource-jev-typed-decisions, benchmark workflows left out) and scored zero-shot,
# then fitted on the benchmark's train and scored again. GTS3, GTS-Uni (3 passes) and ModernBERT-base in one harness.
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 HF_HOME=/root/hf
set -x
O=/workspace/out; mkdir -p $O /root/ck
python3 -m pip install -q einops packaging tokenizers huggingface_hub datasets transformers 2>&1 | tail -1
[ -f /root/ck/gts3.pt ] || cat checkpoints/bert110m/phase3/binarized.pt.part* > /root/ck/gts3.pt
[ -d checkpoints/bert110m/uni ] && [ ! -f /root/ck/uni.pt ] && cat checkpoints/bert110m/uni/binarized.pt.part* > /root/ck/uni.pt
C="--regime fitted --general 500000 --fit-epochs 5 --batch-size 32 --max-len 512"
python3 scripts/decide_probe.py --gts /root/ck/gts3.pt $C --lr 1e-4 --out $O/decide_gts3.json
[ -f /root/ck/uni.pt ] && python3 scripts/decide_probe.py --gts /root/ck/uni.pt --loops 3 $C --lr 1e-4 --out $O/decide_uni_3pass.json
python3 scripts/decide_probe.py --hf answerdotai/ModernBERT-base $C --lr 5e-5 --out $O/decide_modernbert_base.json
echo "=== DECIDE DONE $(date -u +%T) ==="

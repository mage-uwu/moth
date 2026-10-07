# GTS-Uni-Sys1: System One decision models built with Laya's technique (scripts/sys1_train.py) on the LocalLLaMA
# typed-decisions benchmark: [MASK]-per-option scoring, a 2-layer decision head with an act/escalate head, RLCD
# (Gaussian logit exploration, log + spherical (+ RPS) rewards, group-mean REINFORCE) alongside soft cross-entropy,
# per-(type, option count) temperatures. 500K general typed decisions first (scored zero-shot), then the benchmark's
# train (scored fitted). GTS3, GTS-Uni at 3 passes, and ModernBERT-base in the same harness.
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 HF_HOME=/root/hf
set -x
O=/workspace/out; mkdir -p $O /root/ck
python3 -m pip install -q einops packaging tokenizers huggingface_hub datasets transformers 2>&1 | tail -1
[ -f /root/ck/gts3.pt ] || cat checkpoints/bert110m/phase3/binarized.pt.part* > /root/ck/gts3.pt
[ -d checkpoints/bert110m/uni ] && [ ! -f /root/ck/uni.pt ] && cat checkpoints/bert110m/uni/binarized.pt.part* > /root/ck/uni.pt
C="--general 500000 --fit-epochs 5 --batch-size 32 --max-len 512"
[ -f /root/ck/uni.pt ] && python3 scripts/sys1_train.py --gts /root/ck/uni.pt --loops 3 $C --lr 1e-4 --out $O/sys1_uni_3pass.json --save $O/models/sys1_uni_3pass.pt
python3 scripts/sys1_train.py --gts /root/ck/gts3.pt $C --lr 1e-4 --out $O/sys1_gts3.json --save $O/models/sys1_gts3.pt
python3 scripts/sys1_train.py --hf answerdotai/ModernBERT-base $C --lr 5e-5 --out $O/sys1_modernbert_base.json --save $O/models/sys1_modernbert_base.pt
echo "=== DECIDE DONE $(date -u +%T) ==="

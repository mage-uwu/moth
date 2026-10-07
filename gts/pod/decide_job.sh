# GTS-Uni-Sys1: System One decision models built with Laya's technique (scripts/sys1_train.py) on the LocalLLaMA
# typed-decisions benchmark: [MASK]-per-option scoring, a 2-layer decision head with an act/escalate head, RLCD
# (Gaussian logit exploration, log + spherical (+ RPS) rewards, group-mean REINFORCE) alongside soft cross-entropy,
# per-(type, option count) temperatures. 500K general typed decisions first (scored zero-shot), then the benchmark's
# train (scored fitted). GTS3, GTS-Uni at 3 passes, and ModernBERT-base in the same harness.
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 HF_HOME=/root/hf
set -x
O=/workspace/out; mkdir -p $O /root/ck
python3 -m pip install -q einops packaging tokenizers huggingface_hub datasets transformers 2>&1 | tail -1
# GTS models start from their float checkpoints with the ternary learning rate pod/glue_job.sh picked (see there).
[ -f /root/ck/gts3_float.pt ] || cat checkpoints/bert110m/phase3/checkpoint.pt.part* > /root/ck/gts3_float.pt
[ -d checkpoints/bert110m/uni ] && [ ! -f /root/ck/uni_float.pt ] && cat checkpoints/bert110m/uni/checkpoint.pt.part* > /root/ck/uni_float.pt
TLR=${TLR:-$(sed -n 's/ternary learning rate: //p' $O/ternary_lr.txt 2>/dev/null)}; TLR=${TLR:-1e-3}
C="--general 500000 --fit-epochs 5 --batch-size 32 --max-len 512"
[ -f /root/ck/uni_float.pt ] && python3 scripts/sys1_train.py --gts /root/ck/uni_float.pt --loops 3 $C --lr 1e-4 --ternary-lr $TLR --out $O/sys1_uni_3pass.json --save $O/models/sys1_uni_3pass.pt
python3 scripts/sys1_train.py --gts /root/ck/gts3_float.pt $C --lr 1e-4 --ternary-lr $TLR --out $O/sys1_gts3.json --save $O/models/sys1_gts3.pt
python3 scripts/sys1_train.py --hf answerdotai/ModernBERT-base $C --lr 5e-5 --out $O/sys1_modernbert_base.json --save $O/models/sys1_modernbert_base.pt
echo "=== DECIDE DONE $(date -u +%T) ==="

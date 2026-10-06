# Vision-language side quest: a ~38M ternary GTS vision backbone trained on ShareGPT4V-PT (COCO + LCS, 500K) and
# VideoGameBunny (120K) against the frozen GTS masked LM (contrastive + masked caption modelling through a sidecar),
# then an ImageNet-1k linear probe. One RTX 4090, no network volume; data on the pod's disk. Needs END_UTC; HF_TOKEN
# (with the ImageNet-1k terms accepted) for the probe, which is skipped without it. Logs and outputs in /workspace/out,
# served read-only on port 8888.
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 HF_HOME=/root/hf DEBIAN_FRONTEND=noninteractive
END=$(date -d "${END_UTC:?set END_UTC} UTC" +%s)
set -x
O=/workspace/out; R=/workspace/vl_run; D=/root/vl; I=/root/imnet; W=/root/raw
mkdir -p $O $R
for f in result.json backbone.pt binarized.pt checkpoint.pt imagenet_probe.json; do ln -sf $R/$f $O/vl_$f; done
apt-get update -qq && apt-get install -y -qq p7zip-full >/dev/null
python3 -m pip install -q einops packaging tokenizers huggingface_hub pyarrow pillow 2>&1 | tail -1
python3 -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.get_device_name(0))"
nproc; free -g | head -2; df -h /root | tail -1
LM=/root/lm/binarized.pt; mkdir -p /root/lm
cat checkpoints/bert110m/phase2/binarized.pt.part* > $LM
(cd /root/lm && grep " binarized.pt" /root/moth/gts/checkpoints/bert110m/phase2/SHA256SUMS | sha256sum -c) || exit 1
python3 scripts/vl_pretrain.py prep --out $D --work $W || exit 1
if [ -n "$HF_TOKEN" ]; then python3 scripts/vl_pretrain.py prep-imagenet --out $I --work $W/imnet || echo "IMAGENET PREP FAILED"; fi
rm -rf $W
cat $D/*/meta.json
LEFT=$(( (END - $(date +%s)) / 60 - 25 ))   # 25 minutes kept for the probe
echo "minutes left for training: $LEFT"
python3 scripts/vl_pretrain.py train --data $D/sharegpt4v $D/vgb --lm $LM --out $R --minutes $LEFT
[ -f $I/val/meta.json ] && python3 scripts/vl_pretrain.py probe --ckpt $R/backbone.pt --data $I --out $R
echo "=== ALL DONE $(date -u +%T) ==="

# Vision-language side quest: a ~38M ternary GTS vision backbone trained on ShareGPT4V-PT (COCO + LCS, 500K) and
# VideoGameBunny (120K) against the frozen GTS masked LM (contrastive + masked caption modelling through a sidecar).
# No ImageNet probe (its terms are non-commercial). One RTX 4090, no network volume; data on the pod's disk. Needs
# END_UTC. Logs and outputs in /workspace/out, served read-only on port 8888.
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 MKL_NUM_THREADS=8 HF_HOME=/root/hf DEBIAN_FRONTEND=noninteractive
END=$(date -d "${END_UTC:?set END_UTC} UTC" +%s)
set -x
O=/workspace/out; R=/workspace/vl_run; D=/root/vl; W=/root/raw
mkdir -p $O $R
for f in result.json backbone.pt binarized.pt checkpoint.pt; do ln -sf $R/$f $O/vl_$f; done
apt-get update -qq && apt-get install -y -qq p7zip-full >/dev/null
python3 -m pip install -q einops packaging tokenizers huggingface_hub pyarrow pillow 2>&1 | tail -1
for i in $(seq 1 12); do python3 -c "import torch, torchvision; print(torch.__version__, torchvision.__version__, torch.cuda.get_device_name(0))" && break; echo "CUDA not ready (try $i)"; sleep 10; done
python3 -c "import torch; torch.zeros(1).cuda()" || { echo "NO CUDA: stopping"; exit 1; }
nproc; free -g | head -2; df -h /root | tail -1
LM=/root/lm/binarized.pt; mkdir -p /root/lm
cat checkpoints/bert110m/phase2/binarized.pt.part* > $LM
(cd /root/lm && grep " binarized.pt" /root/moth/gts/checkpoints/bert110m/phase2/SHA256SUMS | sha256sum -c) || exit 1
python3 scripts/vl_pretrain.py prep --out $D --work $W || exit 1
rm -rf $W
cat $D/*/meta.json
LEFT=$(( (END - $(date +%s)) / 60 ))
echo "minutes left for training: $LEFT"
python3 scripts/vl_pretrain.py train --data $D/sharegpt4v $D/vgb --lm $LM --out $R --minutes $LEFT
echo "=== ALL DONE $(date -u +%T) ==="

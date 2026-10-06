# The handover's first task on a GPU pod, run from a clone of this branch by the pod's start command:
#   mkdir -p /workspace/out; exec > >(tee -a /workspace/out/boot.log) 2>&1
#   (cd /workspace/out && nohup python3 -m http.server 8888 >/dev/null 2>&1 &)
#   python3 -m pip install -q datasets tiktoken pytest
#   cd /root && rm -rf moth && git clone -q -b claude/determined-cray-yvynh9 https://github.com/mage-uwu/moth.git
#   cd moth/gts && git log --oneline -1 && bash pod/job.sh; sleep infinity
# /workspace/out (served read-only on port 8888) holds the logs and outputs. A run whose model.bin exists is skipped,
# so restarting the pod after a fix picks up the new commit and redoes only what is missing.
export PYTHONUNBUFFERED=1
trap 'echo "=== JOB FAILED at line $LINENO $(date -u +%T) ==="' ERR
set -ex -o pipefail
O=/workspace/out D=/workspace/data/fineweb
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
python3 -c "import torch; print(torch.__version__, torch.cuda.get_device_name(0))"
python3 -m pytest tests/modules/test_gts.py tests/modules/test_gts_ternary.py -q
[ -f $D/meta.json ] || python3 scripts/prepare_fineweb.py --out $D --train-tokens 20000000 --val-tokens 500000
cat $D/meta.json
run() { name=$1; shift; [ -f $O/$name/model.bin ] || python3 scripts/lm_run.py --data $D --out $O/$name "$@" 2>&1 | tee $O/$name.log; }
run smoke_mixed  --arch mixed  --width 256 --layers 4 --deep-depth 8 --steps 50 --eval-every 25
run smoke_mamba2 --arch mamba2 --width 256 --m2-layers 6 --steps 50 --eval-every 25
run fw_mixed     --arch mixed
run fw_mamba2    --arch mamba2
echo "=== ALL DONE $(date -u +%T) ==="

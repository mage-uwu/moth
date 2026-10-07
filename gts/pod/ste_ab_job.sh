# A/B: is the side-branch routing gradient (route_ste) worth keeping? GTS3's float weights continued twice for the same
# wall-clock budget, on the same data order, learning rate and validation: with it (as trained) and without it
# (path-only gradients, which a fully sparse training step can compute). Then the trees' usage balance of both.
export PYTHONUNBUFFERED=1 OMP_NUM_THREADS=8 HF_HOME=/root/hf
set -x
O=/workspace/out; D=/root/wiki_ab; mkdir -p $O
python3 -m pip install -q einops packaging tokenizers huggingface_hub datasets 2>&1 | tail -1
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
cat checkpoints/bert110m/phase3/checkpoint.pt.part* > /root/gts3_float.pt
python3 scripts/bert_pretrain.py prep --out $D --shard-offset 20 --train-tokens 700000000 --val-tokens 2000000 --tmp-dir /root/shards --workers 14
COMMON="--data $D --init-from /root/gts3_float.pt --minutes ${AB_MINUTES:-25} --lr 1e-4 --warmup 200 --eval-every 500 --eval-batches 40 --seed 0"
python3 scripts/bert_pretrain.py train $COMMON --out $O/ab_ste 2>&1 | tee $O/ab_ste.log
python3 scripts/bert_pretrain.py train $COMMON --no-route-ste --out $O/ab_path_only 2>&1 | tee $O/ab_path_only.log
for r in ab_ste ab_path_only; do echo "== tree usage: $r"; python3 scripts/tree_usage.py $O/$r/binarized.pt; done
echo "== tree usage: GTS3 (start)"; python3 scripts/tree_usage.py /root/gts3_float.pt
echo "=== DONE $(date -u +%T) ==="

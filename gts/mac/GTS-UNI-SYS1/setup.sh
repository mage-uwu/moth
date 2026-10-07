#!/bin/bash
# One-time setup on macOS (Apple Silicon or Intel): a local virtual environment with PyTorch and tokenizers, and the
# model weights (125 MB) fetched from the mage-uwu/moth repo with your own git login (only that one file is downloaded).
set -e
cd "$(dirname "$0")"
BRANCH=${WEIGHTS_BRANCH:-claude/determined-cray-yvynh9}
W=weights/sys1_uni_3pass.pt
if [ ! -f "$W" ]; then
  echo "Fetching the weights from github.com/mage-uwu/moth ($BRANCH)..."
  tmp=$(mktemp -d)
  git clone -q --depth 1 --filter=blob:none --sparse -b "$BRANCH" https://github.com/mage-uwu/moth.git "$tmp/moth"
  D=gts/checkpoints/finetuned/sys1
  (cd "$tmp/moth" && git sparse-checkout set --no-cone "/$D/sys1_uni_3pass.pt.part*" "/$D/SHA256SUMS")
  mkdir -p weights
  cat "$tmp/moth/$D"/sys1_uni_3pass.pt.part* > "$W"
  SUM=$(command -v shasum >/dev/null && echo "shasum -a 256" || echo sha256sum)
  (cd weights && grep " sys1_uni_3pass.pt" "$tmp/moth/$D/SHA256SUMS" | $SUM -c -) || { rm -f "$W"; exit 1; }
  rm -rf "$tmp"
fi
python3 -m venv .venv
.venv/bin/pip install -q --upgrade pip
.venv/bin/pip install -q -r requirements.txt
echo "Ready. Try:  .venv/bin/python decide.py"

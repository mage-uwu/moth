# Golden Tree Snake (GTS) fork, 2026.
"""Run a trained lm_run.py model on its kernel test tokens in float64 and compare with the float32 logits that
model.bin stores (and that kernel/ar_bench checks against).

    python scripts/f64_check.py runs/fw_mixed

With 8-bit activations, a per-token absmax rounding can flip on a tiny float difference and change that token's
logits by a lot while the loss barely moves. If float32 and float64 PyTorch disagree on the same few tokens as the
kernel does, the kernel is not the cause.
"""
import argparse, json, os, struct, sys

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "scripts"))
import lm_run  # noqa: E402


def stored_logits(path, vocab):
    """The test ids and float32 reference logits at the end of model.bin."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        for T in (256, 128, 512):
            off = size - 4 * T * vocab - 4 * T - 4
            f.seek(off)
            if struct.unpack("i", f.read(4))[0] == T:
                ids = np.frombuffer(f.read(4 * T), dtype=np.int32)
                return torch.from_numpy(ids.astype(np.int64)), torch.from_numpy(np.frombuffer(f.read(), dtype=np.float32).reshape(T, vocab).copy())
    raise SystemExit("no test block found")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run")
    a = p.parse_args()
    res = json.load(open(os.path.join(a.run, "result.json")))
    args = argparse.Namespace(**res["args"])
    vocab = res["data"]["vocab_size"]
    model = lm_run.build(args, vocab)
    model.load_state_dict(torch.load(os.path.join(a.run, "model.pt"), map_location="cpu"))
    model.eval()
    ids, ref = stored_logits(os.path.join(a.run, "model.bin"), vocab)
    with torch.no_grad():
        l32 = model(ids.unsqueeze(0))[0]
        l64 = model.double()(ids.unsqueeze(0))[0]

    def compare(name, x, y):
        diff = (x.double() - y.double()).abs().amax(-1)
        nll = lambda z: torch.nn.functional.cross_entropy(z[:-1].double(), ids[1:]).item()
        top = (x.argmax(-1) == y.argmax(-1)).sum().item()
        bad = (diff > 1e-2).nonzero().flatten().tolist()
        print(f"{name}: max |diff| {diff.max().item():.2e}, same top-1 on {top}/{len(ids)}, loss {nll(x):.4f} vs {nll(y):.4f}, "
              f"tokens off by more than 1e-2: {len(bad)} {bad[:12]}")

    compare("CPU float32 vs stored (exported) float32", l32, ref)
    compare("CPU float64 vs stored float32           ", l64, ref)
    compare("CPU float64 vs CPU float32              ", l64, l32)


if __name__ == "__main__":
    main()

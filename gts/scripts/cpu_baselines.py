# Golden Tree Snake (GTS) fork, 2026.
"""CPU encoder throughput of BERT-family baselines under their fastest common CPU setups, to compare with
kernel/enc_bench.c on the same machine.

    python scripts/cpu_baselines.py --out results/cpu_baselines.json [--models bert-base distilbert ...] [--lengths 128 512]

For each model: PyTorch float32 (eager), PyTorch dynamic int8 (Linear layers), and ONNX Runtime float32 and dynamic
int8 (graph optimisations on). One sequence at a time (batch 1), encoder only (no head), at each thread count;
tokens/s = sequence length / best-of-N latency. These are the setups a fair "fastest" claim has to beat.
"""
import argparse
import json
import os
import time

import numpy as np
import torch

MODELS = {
    "bert-base": "google-bert/bert-base-uncased",
    "distilbert": "distilbert/distilbert-base-uncased",
    "mobilebert": "google/mobilebert-uncased",
    "tinybert-4l": "huawei-noah/TinyBERT_General_4L_312D",
    "modernbert-base": "answerdotai/ModernBERT-base",
    "modernbert-large": "answerdotai/ModernBERT-large",
}


def best_time(fn, reps):
    fn()
    best = 1e9
    for _ in range(reps):
        t = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - t)
    return best


def bench(name, repo, lengths, threads, reps, work):
    from transformers import AutoModel

    model = AutoModel.from_pretrained(repo).eval()
    n_params = sum(p.numel() for p in model.parameters())
    rows = []
    vocab = model.config.vocab_size
    onnx_path = os.path.join(work, f"{name}.onnx")
    q_path = os.path.join(work, f"{name}.int8.onnx")
    try:  # one export at the longest length, dynamic axes for the others
        L = max(lengths)
        ids = torch.randint(1000, min(vocab, 30000), (1, L))
        class Wrap(torch.nn.Module):  # explicit (ids, mask) -> hidden: the exporter passes inputs by position
            def __init__(self, m):
                super().__init__()
                self.m = m

            def forward(self, input_ids, attention_mask):
                return self.m(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state

        torch.onnx.export(Wrap(model), (ids, torch.ones_like(ids)), onnx_path, input_names=["input_ids", "attention_mask"],
                          output_names=["hidden"], dynamic_axes={"input_ids": {1: "L"}, "attention_mask": {1: "L"}},
                          opset_version=17, dynamo=False)
        from onnxruntime.quantization import QuantType, quantize_dynamic

        quantize_dynamic(onnx_path, q_path, weight_type=QuantType.QInt8)
        have_onnx = True
    except Exception as e:
        print(f"  {name}: ONNX export failed ({type(e).__name__}: {str(e)[:120]}); PyTorch only", flush=True)
        have_onnx = False
    q_model = torch.ao.quantization.quantize_dynamic(model, {torch.nn.Linear}, dtype=torch.qint8)
    for t in threads:
        torch.set_num_threads(t)
        sessions = {}
        if have_onnx:
            import onnxruntime as ort

            for tag, path in (("onnx-fp32", onnx_path), ("onnx-int8", q_path)):
                so = ort.SessionOptions()
                so.intra_op_num_threads, so.inter_op_num_threads = t, 1
                so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                sessions[tag] = ort.InferenceSession(path, so, providers=["CPUExecutionProvider"])
        for L in lengths:
            ids = torch.randint(1000, min(vocab, 30000), (1, L))
            mask = torch.ones_like(ids)
            res = {}
            with torch.inference_mode():
                res["torch-fp32"] = best_time(lambda: model(input_ids=ids, attention_mask=mask), reps)
                res["torch-int8"] = best_time(lambda: q_model(input_ids=ids, attention_mask=mask), reps)
            feed = {"input_ids": ids.numpy(), "attention_mask": mask.numpy()}
            for tag, s in sessions.items():
                names = {i.name for i in s.get_inputs()}
                res[tag] = best_time(lambda: s.run(None, {k: v for k, v in feed.items() if k in names}), reps)
            for tag, sec in res.items():
                rows.append({"model": name, "params_m": round(n_params / 1e6, 1), "setup": tag, "threads": t, "length": L,
                             "tokens_per_s": round(L / sec)})
            best = min(res, key=res.get)
            print(f"  {name:16s} {t} thread(s), length {L}: " + "  ".join(f"{k} {L / v:,.0f}" for k, v in res.items())
                  + f"   (best {best})", flush=True)
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--models", nargs="+", default=["bert-base", "distilbert", "mobilebert", "tinybert-4l", "modernbert-base",
                                                    "modernbert-large"])
    p.add_argument("--lengths", type=int, nargs="+", default=[128, 512])
    p.add_argument("--threads", type=int, nargs="+", default=[1, 4])
    p.add_argument("--reps", type=int, default=5)
    p.add_argument("--work", default="/tmp/cpu_baselines")
    p.add_argument("--out", default="results/cpu_baselines.json")
    a = p.parse_args()
    os.makedirs(a.work, exist_ok=True)
    rows = []
    for name in a.models:
        try:
            rows += bench(name, MODELS[name], a.lengths, a.threads, a.reps, a.work)
        except Exception as e:
            print(f"  {name}: failed ({type(e).__name__}: {str(e)[:160]})", flush=True)
        json.dump({"cpu": open("/proc/cpuinfo").read().split("model name")[1].split("\n")[0].strip(": \t"),
                   "rows": rows}, open(a.out, "w"), indent=1)


if __name__ == "__main__":
    main()

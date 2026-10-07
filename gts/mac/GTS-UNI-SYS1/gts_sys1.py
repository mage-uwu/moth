"""GTS-Uni-Sys1 as a library: load the decision model once, then ask it typed decisions.

    from gts_sys1 import Decider
    d = Decider()                                   # CPU; loads weights/sys1_uni_3pass.pt
    r = d.decide(state={"ticket": "Refund request, order delivered 40 days ago"},
                 question="What should support do?",
                 options=["refund", "partial_refund", "decline", "escalate"])
    print(r["choice"], r["probs"])

A decision is (state, question, options) -> a calibrated probability distribution over the options, plus an
act/escalate confidence (the model's estimate that its top answer is right; low means "send to a human").
"""
import json
import os
import sys
import time

import torch

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "gts"))

from sys1_model import Sys1  # noqa: E402


def _tokenizer():
    """The bert-base-uncased WordPiece tokenizer, bundled in tokenizer/ (no network needed)."""
    from tokenizers import Tokenizer

    tok = Tokenizer.from_file(os.path.join(ROOT, "tokenizer", "tokenizer.json"))
    tok.no_padding()
    tok.no_truncation()
    return tok


DEFAULT_WEIGHTS = os.path.join(ROOT, "weights", "sys1_uni_3pass.pt")
TYPES = ("choice", "noul", "score")  # pick one / true-false / ordinal level (options "0".."k")


class Decider:
    def __init__(self, weights=DEFAULT_WEIGHTS, threads=None, device="cpu", max_len=512):
        if threads:
            torch.set_num_threads(threads)
        self.model, self.temperatures = Sys1.load(weights, _tokenizer())
        self.model.to(device).eval()
        self.device, self.max_len = device, max_len
        self.max_passes = self.model.max_loops

    @staticmethod
    def _state_text(state):
        return state if isinstance(state, str) else json.dumps(state, sort_keys=True)

    @torch.no_grad()
    def _logits(self, ex, passes):
        enc = self.model.encode(ex, self.max_len)
        if enc is None:
            raise ValueError("the decision does not fit in 512 tokens (or has fewer than 2 options)")
        ids, marks = enc
        ids = torch.tensor([ids], device=self.device)
        outs, act = self.model(ids, ids != self.model.pad, [0] * len(marks), marks, [len(marks)], loops=passes)
        return outs[0].float().cpu(), float(torch.sigmoid(act[0]))

    def decide(self, state, question, options, descriptions=None, type="choice", passes=None, adaptive=None):
        """``passes``: how many times the shared layers run (1..3; default all). ``adaptive``: a confidence threshold
        tau; start with 1 pass and add passes while the act/escalate confidence is below tau."""
        options = [str(o) for o in options]
        if descriptions is not None and len(descriptions) != len(options):
            raise ValueError("descriptions must match options one to one")
        ex = {"state": self._state_text(state), "question": question, "options": options,
              "descriptions": descriptions, "type": type}
        t0 = time.perf_counter()
        if adaptive is not None:
            n = 1
            z, conf = self._logits(ex, n)
            while conf < adaptive and n < self.max_passes:
                n += 1
                z, conf = self._logits(ex, n)
        else:
            n = passes or self.max_passes
            z, conf = self._logits(ex, n)
        T = self.temperatures.get(f"{type}/{len(options)}", self.temperatures["*"])
        p = torch.softmax(z / T, -1).tolist()
        order = sorted(range(len(options)), key=lambda i: -p[i])
        return {"choice": options[order[0]], "probs": {options[i]: p[i] for i in order},
                "confidence": conf, "escalate": conf < 0.5, "passes": n, "temperature": T,
                "ms": (time.perf_counter() - t0) * 1000}

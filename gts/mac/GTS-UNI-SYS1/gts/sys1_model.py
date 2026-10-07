"""GTS-Uni-Sys1 for inference: a ternary GTS-Uni encoder plus a 2-layer GTS decision head that scores every option at
its own [MASK] token, and an act/escalate head on [CLS]. (Inference-only extract of the training harness's Sys1.)"""
import torch

from mamba_ssm.models.gts_encoder import GTSBlock, GTSConfig, GTSForMaskedLM, RMSNorm
from mamba_ssm.utils.ternary_pack import load_binarized

CLS, SEP, MASK, PAD = 101, 102, 103, 0


class Sys1(torch.nn.Module):
    def __init__(self, cfg, tokenizer, loops, head_layers=2, head_deep_depth=6):
        super().__init__()
        self.backbone = GTSForMaskedLM(GTSConfig(**cfg)).backbone
        self.loops, self.max_loops, self.pad = loops, cfg.get("loops", 1), PAD
        self.enc = lambda s: tokenizer.encode(s, add_special_tokens=False).ids  # noqa: E731
        hcfg = GTSConfig(**{k: v for k, v in cfg.items() if k not in ("loops", "latent_tokens")})
        hcfg.deep_depth, hcfg.n_layer = head_deep_depth, head_layers
        self.layers = torch.nn.ModuleList([GTSBlock(hcfg, i) for i in range(head_layers)])
        d = cfg["d_model"]
        self.norm = RMSNorm(d)
        self.scorer = torch.nn.Sequential(torch.nn.Linear(d, d), torch.nn.GELU(), torch.nn.Linear(d, 1))
        self.escalate = torch.nn.Sequential(torch.nn.Linear(d, d // 4), torch.nn.GELU(), torch.nn.Linear(d // 4, 1))

    @classmethod
    def load(cls, path, tokenizer):
        """Returns (model, temperatures) from a binarized Sys1 checkpoint."""
        blob = torch.load(path, map_location="cpu", weights_only=False)
        model = cls(blob["config"], tokenizer, blob["loops"], blob["head_layers"])
        return load_binarized(blob, model), blob["temperatures"]

    def encode(self, ex, max_len, head_budget=256):
        """"[CLS] question [SEP] state [SEP] option: description [MASK] ... [SEP]" and each option's [MASK] position."""
        if len(ex["options"]) < 2:
            return None
        q = self.enc(ex["question"])[:64]
        tail, per = [], max(4, head_budget // max(1, len(ex["options"])) - 1)
        for i, o in enumerate(ex["options"]):
            desc = ex["descriptions"][i] if ex.get("descriptions") else None
            tail.append(self.enc(f"{o}: {desc}" if desc else str(o))[:per])
        n_tail = sum(len(t) + 1 for t in tail) + 1
        room = max(8, max_len - 3 - len(q) - n_tail)
        ids = [CLS] + q + [SEP] + self.enc(ex["state"])[:room] + [SEP]
        marks = []
        for t in tail:
            ids += t
            marks.append(len(ids))
            ids.append(MASK)
        ids.append(SEP)
        return None if len(ids) > max_len else (ids, marks)

    def forward(self, ids, mask, rows, cols, n_opts, loops=None):
        h = self.backbone(ids, attention_mask=mask, loops=loops or self.loops)
        for layer in self.layers:
            h = layer(h, attention_mask=mask)
        h = self.norm(h)
        logits = self.scorer(h[rows, cols]).squeeze(-1).float()
        act = self.escalate(h[:, 0]).squeeze(-1).float()
        return torch.split(logits, n_opts), act

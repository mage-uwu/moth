# Golden Tree Snake (GTS) fork, 2026.
"""Small CPU experiment comparing ways of getting a ternary GTS model.

    python scripts/gts_ternary_demo.py --data corpus.txt --arm teacher
    python scripts/gts_ternary_demo.py --data corpus.txt --arm <name>     # see ARMS
    python scripts/gts_ternary_demo.py --report

Results accumulate in ``--workdir``. This is a sanity check of the training path at toy scale
(a byte-level masked LM with a few hundred thousand parameters), not evidence about quality at size.
"""

import argparse
import copy
import json
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import train_gts as T  # noqa: E402
from mamba_ssm.models.gts_encoder import GTSConfig, GTSForMaskedLM  # noqa: E402
from mamba_ssm.utils.gts_qat import make_ternary_student, qat_loss, ternary_stats  # noqa: E402

# name: (description, settings)
ARMS = {
    "teacher": ("full precision", {}),
    "fp_longer": ("full precision, trained for the QAT steps as well", {}),
    "ptq": ("teacher ternarised, no training", {}),
    "scratch": ("ternary from scratch, same total steps", {}),
    "qat_ce": ("QAT from teacher, cross-entropy only", dict(alpha=0.0, beta=0.0)),
    "qat_kd": ("QAT from teacher + distillation", dict(alpha=0.5, beta=0.0)),
    "qat_kd_route": ("QAT + distillation + route distillation", dict(alpha=0.5, beta=1.0)),
    "qat_kd_route_warm": ("... + quantisation warmed in", dict(alpha=0.5, beta=1.0, lambda_warmup=0.25)),
    "qat_kd_route_a8": ("... + 8-bit activations", dict(alpha=0.5, beta=1.0, act_bits=8)),
}


def main():
    p = T.build_parser()
    p.add_argument("--arm", choices=list(ARMS))
    p.add_argument("--report", action="store_true")
    p.add_argument("--workdir", default="demo_runs")
    p.add_argument("--teacher-steps", type=int, default=3000)
    p.add_argument("--qat-steps", type=int, default=1000)
    p.set_defaults(data="", d_model=128, n_layer=4, depth=6, d_state=8, ternary_group=64,
                   batch_size=16, seq_len=96, lr=2e-3, warmup=100, log_every=250)
    for action in p._actions:
        if action.dest == "data":
            action.required = False  # not needed for --report
    args = p.parse_args(["fp", *sys.argv[1:]])  # train_gts's positional mode is unused here
    args.device = "cpu"  # a toy comparison, written for the CPU
    os.makedirs(args.workdir, exist_ok=True)
    results_path = os.path.join(args.workdir, f"results_seed{args.seed}.json")
    results = json.load(open(results_path)) if os.path.exists(results_path) else {}

    if args.report:
        for name, (desc, _) in ARMS.items():
            if name in results:
                r = results[name]
                extra = "".join(f"  {k} {r[k]:.3f}" for k in ("zero_ratio_mean", "path_agreement") if k in r)
                print(f"{name:20s} acc {r['accuracy']:.4f}  loss {r['loss']:.4f}{extra}   {desc}")
        return

    torch.manual_seed(args.seed)
    corpus = T.Corpus(args.data)
    teacher_path = os.path.join(args.workdir, f"teacher_seed{args.seed}.pt")
    settings = ARMS[args.arm][1]

    def fresh(ternary):
        return GTSForMaskedLM(GTSConfig(
            d_model=args.d_model, n_layer=args.n_layer, vocab_size=corpus.vocab_size, depth=args.depth,
            d_state=args.d_state, ternary=ternary, ternary_group=args.ternary_group))

    teacher = None
    if args.arm == "teacher":
        args.steps = args.teacher_steps
        model = T.train(fresh(False), corpus, args)
        T.save(model, teacher_path)
    elif args.arm == "scratch":
        args.steps = args.teacher_steps + args.qat_steps
        model = T.train(fresh(True), corpus, args)
    else:
        teacher = T.load(teacher_path)
        args.steps = args.qat_steps
        args.lr = args.lr / 2  # fine-tuning rate for everything that starts from the teacher
        if args.arm == "fp_longer":
            model = T.train(copy.deepcopy(teacher), corpus, args)
        else:
            model = make_ternary_student(teacher, args.ternary_group, settings.get("act_bits"))
            if args.arm != "ptq":
                args.alpha, args.beta = settings["alpha"], settings["beta"]
                args.lambda_warmup = int(settings.get("lambda_warmup", 0) * args.steps)
                model = T.train(model, corpus, args, teacher=teacher)

    result = T.evaluate(model, corpus, args, batches=60)
    result.update(ternary_stats(model))
    if teacher is not None and model.config.ternary:
        g = torch.Generator().manual_seed(1234)
        model.eval()
        agree = [qat_loss(model, teacher, *corpus.batch("heldout", args.batch_size, args.seq_len, args.mlm_prob, g))["path_agreement"]
                 for _ in range(5)]
        result["path_agreement"] = sum(agree) / len(agree)
    results[args.arm] = result
    json.dump(results, open(results_path, "w"), indent=1)
    print(args.arm, {k: round(v, 4) for k, v in result.items()})


if __name__ == "__main__":
    main()

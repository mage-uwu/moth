#!/usr/bin/env python3
"""Play with GTS-Uni-Sys1 from the terminal.

    python decide.py                          # the bundled examples (examples/decisions.json)
    python decide.py my_decisions.json        # your own: a list of {state, question, options, descriptions?, type?}
    python decide.py -i                       # interactive: type a situation, a question and options
    python decide.py --passes 1               # 1, 2 or 3 passes through the shared layers (default 3)
    python decide.py --adaptive 0.7           # 1 pass, more only while the model's confidence is below 0.7
    python decide.py --bench                  # decisions per second at 1, 2 and 3 passes on this machine
    python decide.py --json                   # machine-readable output
"""
import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.abspath(__file__))
BAR = 28
BLOCK, ARROW, DOT = "█", "▶", "·"


def show(ex, r):
    print(f"\n\033[1m{ex['question']}\033[0m")
    state = ex["state"] if isinstance(ex["state"], str) else json.dumps(ex["state"])
    print(f"\033[2m{state[:220]}{'...' if len(state) > 220 else ''}\033[0m")
    for opt, p in r["probs"].items():
        mark = ARROW if opt == r["choice"] else " "
        bar = BLOCK * round(p * BAR)
        print(f"  {mark} {opt:<18} {bar:<{BAR}} {p:6.1%}")
    verdict = "act" if not r["escalate"] else "escalate to a human"
    print(f"  confidence {r['confidence']:.2f} ({verdict})  {DOT}  {r['passes']} pass{'es' if r['passes'] > 1 else ''}"
          f"  {DOT}  {r['ms']:.0f} ms")


def ask(prompt, default=None):
    v = input(f"{prompt}{f' [{default}]' if default else ''}: ").strip()
    return v or default


def interactive(d, a):
    print("Describe a situation and the decision to make. Empty situation to quit.")
    while True:
        state = ask("\nsituation (text or JSON)")
        if not state:
            return
        try:
            state = json.loads(state)
        except ValueError:
            pass
        question = ask("question", "What should happen next?")
        options = [o.strip() for o in ask("options, comma-separated", "approve, deny, escalate").split(",") if o.strip()]
        kind = ask("type (choice / noul / score)", "choice")
        ex = {"state": state, "question": question, "options": options, "type": kind}
        show(ex, d.decide(**ex, passes=a.passes, adaptive=a.adaptive))


def bench(d, examples):
    print(f"\n{len(examples)} decisions, {os.cpu_count()} CPU cores, torch threads {__import__('torch').get_num_threads()}")
    for n in range(1, d.max_passes + 1):
        d.decide(**examples[0], passes=n)  # warm-up
        t0 = time.perf_counter()
        for ex in examples:
            d.decide(**ex, passes=n)
        dt = time.perf_counter() - t0
        print(f"  {n} pass{'es' if n > 1 else '  '}: {len(examples) / dt:6.1f} decisions/s  ({1000 * dt / len(examples):.0f} ms each)")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("file", nargs="?", default=os.path.join(ROOT, "examples", "decisions.json"))
    p.add_argument("-i", "--interactive", action="store_true")
    p.add_argument("--passes", type=int, choices=[1, 2, 3])
    p.add_argument("--adaptive", type=float, metavar="TAU")
    p.add_argument("--bench", action="store_true")
    p.add_argument("--json", action="store_true")
    p.add_argument("--threads", type=int, help="CPU threads (default: PyTorch's choice)")
    p.add_argument("--weights", help="another Sys1 checkpoint, e.g. the GTS3 one from the repo")
    a = p.parse_args()
    sys.path.insert(0, ROOT)
    from gts_sys1 import DEFAULT_WEIGHTS, Decider

    t0 = time.perf_counter()
    d = Decider(a.weights or DEFAULT_WEIGHTS, threads=a.threads)
    if not a.json:
        print(f"GTS-Uni-Sys1 loaded in {time.perf_counter() - t0:.1f} s (ternary, {d.max_passes} passes max)")
    if a.interactive:
        return interactive(d, a)
    examples = json.load(open(a.file))
    if a.bench:
        return bench(d, examples)
    out = []
    for ex in examples:
        r = d.decide(**ex, passes=a.passes, adaptive=a.adaptive)
        out.append({**ex, **r}) if a.json else show(ex, r)
    if a.json:
        print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()

# Streams exactly-labelled, logic-adjacent training text from kb.json (data/kb_build.py) to stdout:
#   python3 data/kb_tasks.py kb.json --mb 20 | ./moth -
# Every answer is derived from the KB's unanimous, embedding-checked facts, so the stream is validated before
# training. Question frames are the canonical ones or Qwen's judged-equivalent paraphrases (half each), so the
# student has to learn the fact, not the wording. Tasks:
#   fact     is it true that a X <property>?              kind      what kind of thing is X?
#   inherit  most <category> <property>; does X?          which     which of these <property>?
#   odd      odd one out, unique on exactly one property  common    what do X and Y share?
#   differ   how X differs from Y                         both      X <p1> and <not p2>?
#   name     name something that <p1> but <not p2>        logic     syllogisms over made-up words (no KB)
# Held out: 1 fact in 10 (by hash) never appears, stated or implied, in any task. --eval writes a jsonl of
# questions about those held-out facts plus fresh logic items, for measuring what the student generalises.
import argparse, json, random, re, sys, zlib

MASS = {"materials and substances", "drinks"}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kb"); ap.add_argument("--mb", type=float, default=20); ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--eval", default=None); ap.add_argument("--eval-n", type=int, default=2000)
    a = ap.parse_args()
    kb = json.load(open(a.kb))
    P = {p["id"]: p for p in kb["props"]}; C = kb["concepts"]; T = kb["templates"]
    held = lambda c, p: zlib.crc32(f"{c['name']}|{p}".encode()) % 10 == 0
    def val(c, p, train=True):                 # True / False / None (unknown or held out)
        if train and held(c, p): return None
        return True if p in c["yes"] else False if p in c["no"] else None
    def ax(c, cap=False):                      # "a crow", "an owl", "milk"
        s = c["name"] if c["category"] in MASS else ("an " if c["name"][0] in "aeiou" else "a ") + c["name"]
        return s[0].upper() + s[1:] if cap else s
    def frame(kind, rng, **kw):
        t = T[kind][0] if rng.random() < 0.5 or len(T[kind]) == 1 else rng.choice(T[kind][1:])
        t = re.sub(r"\b[aA]n? \{x\}", "{ax}", re.sub(r"\b[aA]n? \{y\}", "{ay}", t))
        return t.format(**kw)
    pred = lambda p, v: P[p]["pred"] if v else P[p]["neg"]
    yn = lambda v: "Yes" if v else "No"
    cats = {}
    for c in C: cats.setdefault(c["category"], []).append(c)

    def t_fact(r, train=True):
        c, p = r.choice(C), r.choice(list(P)); v = val(c, p, train)
        if v is None: return None
        q = frame("fact", r, ax=ax(c), x=c["name"], p=P[p]["pred"])
        return q, f"{yn(v)}. {ax(c, 1)} {pred(p, v)}."
    def t_kind(r):
        c = r.choice(C); k = kb["kinds"][c["category"]]
        return frame("kind", r, ax=ax(c), x=c["name"]), f"{ax(c, 1)} is one of the {c['category']}, a kind of {k}."
    def t_inherit(r):
        cat = r.choice(list(cats)); p = r.choice(list(P)); m = cats[cat]
        vs = [(c, val(c, p)) for c in m]; vs = [(c, v) for c, v in vs if v is not None]
        if len(vs) < 5: return None
        f = sum(v for _, v in vs) / len(vs)
        if 0.2 < f < 0.8: return None
        most = f >= 0.8; c, v = r.choice(vs)
        q = f"Most {cat} {pred(p, most)}. {ax(c, 1)} is one of the {cat}. Is it true that {ax(c)} {P[p]['pred']}?"
        return q, (f"{yn(v)}. {ax(c, 1)} {pred(p, v)}." + (" It is an exception." if v != most else ""))
    def t_which(r):
        p = r.choice(list(P)); k = r.choice([3, 4])
        ys = [c for c in C if val(c, p) is True]; ns = [c for c in C if val(c, p) is False]
        if not ys or len(ns) < k - 1: return None
        pick = [r.choice(ys)] + r.sample(ns, k - 1); r.shuffle(pick); ans = [c for c in pick if val(c, p)][0]
        return frame("which", r, p=P[p]["pred"].replace("is ", "are ", 1) if P[p]["pred"].startswith("is ") else plural_verb(P[p]["pred"]),
                     list=", ".join(c["name"] for c in pick)), f"The {ans['name']}. The others {plural_verb(P[p]['neg'])}."
    def t_odd(r):
        p = r.choice(list(P)); v = r.random() < 0.5
        same = [c for c in C if val(c, p) is v]; other = [c for c in C if val(c, p) is (not v)]
        if len(same) < 3 or not other: return None
        pick = r.sample(same, 3) + [r.choice(other)]
        for q in P:                            # the answer must be the only item that stands alone on any property
            vs = [val(c, q) for c in pick]
            if None in vs: continue
            for i in range(3):
                if vs.count(vs[i]) == 1: return None
        odd = pick[3]; r.shuffle(pick)
        return frame("odd", r, list=", ".join(c["name"] for c in pick)), \
               f"The {odd['name']}. The others {plural_verb(pred(p, v))}; {ax(odd)} {pred(p, not v)}."
    def t_common(r):
        x, y = r.sample(C, 2)
        sh = [p for p in P if val(x, p) is True and val(y, p) is True]
        if not sh: return None
        sh.sort(key=lambda p: sum(p in c["yes"] for c in C)); sh = sh[:2]   # the most telling: rarest first
        return frame("common", r, ax=ax(x), ay=ax(y), x=x["name"], y=y["name"]), \
               "Both " + " and ".join(plural_verb(P[p]["pred"]) for p in sh) + "."
    def t_differ(r):
        x, y = r.sample(C, 2)
        d = [p for p in P if val(x, p) is True and val(y, p) is False]
        if not d: return None
        p = r.choice(d)
        return frame("differ", r, ax=ax(x), ay=ax(y), x=x["name"], y=y["name"]), f"{ax(x, 1)} {P[p]['pred']}, but {ax(y)} {P[p]['neg']}."
    def t_both(r, train=True):
        c = r.choice(C); p1, p2 = r.sample(list(P), 2); v1, v2 = val(c, p1, train), val(c, p2, train)
        if v1 is None or v2 is None: return None
        q = f"Is it true that {ax(c)} {P[p1]['pred']} and {P[p2]['neg']}?"
        if not v1: return q, f"No. {ax(c, 1)} {P[p1]['neg']}."
        if v2: return q, f"No. {ax(c, 1)} {P[p2]['pred']}."
        return q, f"Yes. {ax(c, 1)} {P[p1]['pred']} and {P[p2]['neg']}."
    def t_name(r):
        p1, p2 = r.sample(list(P), 2); ok = [c for c in C if val(c, p1) is True and val(c, p2) is False]
        if not ok: return None
        return f"Name something that {P[p1]['pred']} but {P[p2]['neg']}.", f"{ax(r.choice(ok), 1)}."
    def t_logic(r):
        w = nonce(r, 4); p = r.choice(list(P)); n = r.randint(1, 3)   # chain: w0 is a w1 ... every wn <p or not p>
        v = r.random() < 0.5; facts = [f"Every {w[i]} is a {w[i + 1]}." for i in range(n)]
        facts.append(f"Every {w[n]} {P[p]['pred']}." if v else f"No {w[n]} {P[p]['pred'].replace('can ', 'can ', 1)}.")
        kind = r.random()
        if kind < 0.7: q, ans = f"Is it true that a {w[0]} {P[p]['pred']}?", f"{yn(v)}. A {w[0]} is a {w[n]}, and {'every' if v else 'no'} {w[n]} {P[p]['pred']}."
        else:                                  # a thing outside the chain: nothing follows
            q, ans = f"Is it true that a {w[n + 1] if n + 1 < 4 else nonce(r, 1)[0]} {P[p]['pred']}?", "We cannot tell from what we know."
        r.shuffle(facts)
        return " ".join(facts) + " " + q, ans
    tasks = [(t_fact, 28), (t_kind, 4), (t_inherit, 10), (t_which, 10), (t_odd, 8), (t_common, 8),
             (t_differ, 8), (t_both, 8), (t_name, 6), (t_logic, 10)]
    fns, wts = zip(*tasks)

    if a.eval:                                 # held-out facts (and 'both' built on them) + unseen logic items
        r = random.Random(a.seed + 999); out = []
        hf = [(c, p) for c in C for p in P if held(c, p) and val(c, p, False) is not None]
        for c, p in r.sample(hf, min(len(hf), a.eval_n // 2)):
            v = val(c, p, False)
            out.append({"type": "heldout_fact", "prompt": f"Q: Is it true that {ax(c)} {P[p]['pred']}?\nA:", "answer": yn(v)})
        while len(out) < a.eval_n:
            q, ans = t_logic(r)
            out.append({"type": "logic", "prompt": f"Q: {q}\nA:", "answer": ans.split(".")[0]})
        with open(a.eval, "w") as f: f.write("\n".join(json.dumps(o) for o in out) + "\n")
        print(f"eval: {len(out)} items ({len(hf)} held-out facts available)", file=sys.stderr)

    r, n, target, o = random.Random(a.seed), 0, int(a.mb * 1e6), sys.stdout.buffer
    while n < target:
        got = r.choices(fns, wts)[0](r)
        if got is None: continue
        s = f"Q: {got[0]}\nA: {got[1]}\n\n".encode(); o.write(s); n += len(s)
    o.flush()

def plural_verb(pred):                         # "can fly" -> "can fly", "is made of" -> "are made of", "has" -> "have"
    w = pred.split(" ", 1); v = {"is": "are", "has": "have", "does": "do"}.get(w[0])
    if v: return v + (" " + w[1] if len(w) > 1 else "")
    if w[0] in ("can", "cannot"): return pred
    if w[0].endswith("s") and not w[0].endswith("ss"): return w[0][:-1] + (" " + w[1] if len(w) > 1 else "")
    return pred

def nonce(r, k):                               # pronounceable made-up words, so logic can't lean on facts
    out = set()
    while len(out) < k:
        out.add("".join(r.choice("bdfgklmnprstvz") + r.choice("aeiou") for _ in range(2)) + r.choice("bdgkmnprst"))
    return list(out)

if __name__ == "__main__":
    main()

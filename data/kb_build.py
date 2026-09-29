# Builds a concept knowledge base with Qwen alone, validated before any training, for data/kb_tasks.py.
# The TM-distillation recipe (Gao et al. 2026; Bhattarai et al. 2024) turned into text data: a language
# model's semantic space becomes discrete, checkable clauses, which a generator then teaches through many
# paraphrases. Every step is Qwen (Apache 2.0):
#   1. concepts    Qwen3-4B-Thinking lists concrete members of each category
#   2. properties  it answers a fixed set of yes/no properties per concept, VOTES times; only unanimous
#                  answers become facts, the rest stay unknown and are never taught
#   3. templates   it paraphrases each question frame, then judges each paraphrase for equivalence
#   4. embedding   Qwen3-Embedding-0.6B (CPU): a fact that contradicts 80%+ of the concept's 10 nearest
#                  neighbours, where they agree, is dropped to unknown
# Output: kb.json (the KB, templates, stats) and kb.bin (the guard: per concept, 64-bit masks of known-yes
# and known-no properties, so checking a claim is one bit test).
# usage (GPU): pip install vllm; python3 data/kb_build.py --out kb --per-cat 20
import argparse, json, re, struct, sys, time

CATEGORIES = {  # category: kind of thing (for inheritance tasks)
    "birds": "animal", "mammals": "animal", "fish and sea creatures": "animal", "insects and bugs": "animal",
    "reptiles and amphibians": "animal", "trees and plants": "plant", "fruits and vegetables": "food",
    "foods and dishes": "food", "drinks": "food", "kitchen items": "object", "tools": "object",
    "weapons": "object", "clothing": "object", "furniture": "object", "musical instruments": "object",
    "vehicles": "object", "materials and substances": "material", "toys": "object", "containers": "object",
    "household objects": "object",
}
# id, yes/no question about a typical member, predicate ("a crow ___"), negated predicate
PROPS = [
    ("alive", "Is it a living thing?", "is alive", "is not alive"),
    ("can_fly", "Can it fly by itself?", "can fly", "cannot fly"),
    ("can_swim", "Can it swim by itself?", "can swim", "cannot swim"),
    ("has_legs", "Does it have legs?", "has legs", "has no legs"),
    ("has_fur", "Does it have fur or hair?", "has fur", "has no fur"),
    ("has_feathers", "Does it have feathers?", "has feathers", "has no feathers"),
    ("has_scales", "Does it have scales?", "has scales", "has no scales"),
    ("lays_eggs", "Does it lay eggs?", "lays eggs", "does not lay eggs"),
    ("eats_meat", "Does it eat meat?", "eats meat", "does not eat meat"),
    ("eats_plants", "Does it eat plants?", "eats plants", "does not eat plants"),
    ("metal", "Is it usually made of metal?", "is made of metal", "is not made of metal"),
    ("wood", "Is it usually made of wood?", "is made of wood", "is not made of wood"),
    ("glass", "Is it usually made of glass?", "is made of glass", "is not made of glass"),
    ("cloth", "Is it usually made of cloth?", "is made of cloth", "is not made of cloth"),
    ("stone", "Is it usually made of stone?", "is made of stone", "is not made of stone"),
    ("plastic", "Is it usually made of plastic?", "is made of plastic", "is not made of plastic"),
    ("edible", "Do people usually eat it?", "is eaten by people", "is not eaten by people"),
    ("drink", "Do people drink it?", "is something people drink", "is not something people drink"),
    ("sweet", "Does it usually taste sweet?", "tastes sweet", "does not taste sweet"),
    ("container", "Is it used to hold other things?", "holds other things", "does not hold other things"),
    ("holds_water", "Can it hold water without leaking?", "can hold water", "cannot hold water"),
    ("flammable", "Does it burn easily?", "burns easily", "does not burn easily"),
    ("melts", "Does it melt when heated on a stove?", "melts when heated", "does not melt when heated"),
    ("floats", "Does it float on water?", "floats on water", "does not float on water"),
    ("magnetic", "Does a magnet stick to it?", "sticks to a magnet", "does not stick to a magnet"),
    ("transparent", "Can you see through it?", "is see-through", "is not see-through"),
    ("sharp", "Is it sharp enough to cut you?", "is sharp", "is not sharp"),
    ("soft", "Is it soft to touch?", "is soft", "is not soft"),
    ("heavy", "Is it too heavy for one person to lift?", "is too heavy to lift", "can be lifted by one person"),
    ("fragile", "Does it usually break if dropped on a hard floor?", "breaks if dropped", "does not break if dropped"),
    ("big", "Is it bigger than a person?", "is bigger than a person", "is not bigger than a person"),
    ("small", "Does it fit in a hand?", "fits in a hand", "does not fit in a hand"),
    ("man_made", "Is it made by people?", "is made by people", "is not made by people"),
    ("liquid", "Is it a liquid?", "is a liquid", "is not a liquid"),
    ("kitchen", "Is it usually found in a kitchen?", "is found in a kitchen", "is not found in a kitchen"),
    ("forest", "Is it usually found in a forest?", "is found in a forest", "is not found in a forest"),
    ("sea", "Does it usually live or belong in the sea?", "belongs in the sea", "does not belong in the sea"),
    ("farm", "Is it usually found on a farm?", "is found on a farm", "is not found on a farm"),
    ("cutting", "Is it used for cutting?", "is used for cutting", "is not used for cutting"),
    ("cooking", "Is it used for cooking?", "is used for cooking", "is not used for cooking"),
    ("writing", "Is it used for writing or drawing?", "is used for writing", "is not used for writing"),
    ("worn", "Do people wear it?", "is worn by people", "is not worn by people"),
    ("weapon", "Is it used as a weapon?", "is used as a weapon", "is not used as a weapon"),
    ("travel", "Do people ride in it or on it to travel?", "carries people", "does not carry people"),
    ("music", "Is it used to make music?", "makes music", "does not make music"),
    ("electric", "Does it need electricity to work?", "needs electricity", "does not need electricity"),
    ("light", "Does it give off light?", "gives off light", "does not give off light"),
    ("hot", "Is it usually hot?", "is usually hot", "is not usually hot"),
    ("cold", "Is it usually cold?", "is usually cold", "is not usually cold"),
    ("dangerous", "Can it easily hurt or kill a person?", "is dangerous", "is not dangerous"),
]
assert len(PROPS) <= 64
# question frames to paraphrase: placeholders must survive paraphrasing
FRAMES = {
    "fact": "Is it true that a {x} {p}?",
    "which": "Which of these {p}: {list}?",
    "odd": "Which one is the odd one out: {list}?",
    "common": "What do a {x} and a {y} have in common?",
    "differ": "How is a {x} different from a {y}?",
    "kind": "What kind of thing is a {x}?",
}
NAME = re.compile(r"^[a-z][a-z' -]{1,30}$")

def after_think(o):
    t = o.outputs[0].text
    return t.split("</think>", 1)[1].strip() if "</think>" in t else None   # no </think>: truncated, drop

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="kb"); ap.add_argument("--per-cat", type=int, default=20)
    ap.add_argument("--votes", type=int, default=3); ap.add_argument("--paraphrases", type=int, default=12)
    ap.add_argument("--model", default="Qwen/Qwen3-4B-Thinking-2507")
    ap.add_argument("--embed", default="Qwen/Qwen3-Embedding-0.6B")
    a = ap.parse_args()
    from vllm import LLM, SamplingParams
    llm = LLM(model=a.model, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=0.92, seed=1)
    tok = llm.get_tokenizer()
    chat = lambda p: tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True)
    sp = lambda n=1: SamplingParams(n=n, temperature=0.6, top_p=0.95, top_k=20, max_tokens=6000, seed=1)
    stats, t0 = {}, time.time()
    def log(msg): print(f"[{(time.time() - t0) / 60:5.1f} min] {msg}", flush=True)

    # 1. concepts
    cats = list(CATEGORIES)
    outs = llm.generate([chat(f"List {a.per_cat} common, concrete, well-known {c} that a child would know. "
                              "One per line: a lowercase singular noun of one or two words. No numbering, no other text.")
                         for c in cats], sp())
    concepts, seen = [], set()
    for c, o in zip(cats, outs):
        for line in (after_think(o) or "").splitlines():
            n = line.strip().strip(".-*").strip().lower()
            if NAME.match(n) and n not in seen: seen.add(n); concepts.append({"name": n, "category": c})
    log(f"concepts: {len(concepts)} from {len(cats)} categories")

    # 2. property votes, in groups so each answer stays short
    groups = [PROPS[i:i + 13] for i in range(0, len(PROPS), 13)]
    reqs = [(ci, gi) for ci in range(len(concepts)) for gi in range(len(groups))]
    prompts = []
    for ci, gi in reqs:
        c = concepts[ci]
        qs = "\n".join(f"{pid}: {q}" for pid, q, _, _ in groups[gi])
        prompts.append(chat(f"Think about a typical {c['name']} (one of the {c['category']}). Answer each question "
                            f"about a typical {c['name']} with yes or no.\n\n{qs}\n\nReply with exactly one line per "
                            "question, in the form '<id>: yes' or '<id>: no', and nothing else."))
    votes = {}
    for (ci, gi), o in zip(reqs, llm.generate(prompts, sp(a.votes))):
        for s in o.outputs:
            t = s.text.split("</think>", 1)[1] if "</think>" in s.text else ""
            got = dict(m.groups() for m in re.finditer(r"^\s*([a-z_]+)\s*:\s*(yes|no)\b", t.lower(), re.M))
            for pid, *_ in groups[gi]: votes.setdefault((ci, pid), []).append(got.get(pid))
    for ci, c in enumerate(concepts):
        c["yes"], c["no"] = [], []
        for pid, *_ in PROPS:
            v = votes.get((ci, pid), [])
            if len(v) == a.votes and all(x == "yes" for x in v): c["yes"].append(pid)
            elif len(v) == a.votes and all(x == "no" for x in v): c["no"].append(pid)
    known = sum(len(c["yes"]) + len(c["no"]) for c in concepts)
    stats["facts_unanimous"] = known; stats["facts_asked"] = len(concepts) * len(PROPS)
    log(f"properties: {known}/{len(concepts) * len(PROPS)} unanimous over {a.votes} votes")

    # 3. paraphrased frames, each judged for equivalence
    fr = list(FRAMES)
    outs = llm.generate([chat(f"Write {a.paraphrases} different ways to ask this question. Keep every placeholder in "
                              f"curly braces exactly as it is, use simple words, one question per line, no numbering.\n\n"
                              f"{FRAMES[f]}") for f in fr], sp())
    cands = []
    for f, o in zip(fr, outs):
        need = set(re.findall(r"\{\w+\}", FRAMES[f]))
        for line in (after_think(o) or "").splitlines():
            q = line.strip().lstrip("-*0123456789. ").strip()
            if q.endswith("?") and set(re.findall(r"\{\w+\}", q)) == need and q != FRAMES[f]: cands.append((f, q))
    outs = llm.generate([chat(f"Do these two questions ask for exactly the same thing? Placeholders in curly braces "
                              f"stand for the same values in both.\nA: {FRAMES[f]}\nB: {q}\nAnswer yes or no only.")
                         for f, q in cands], sp())
    templates = {f: [FRAMES[f]] for f in fr}
    for (f, q), o in zip(cands, outs):
        if (after_think(o) or "").lower().startswith("yes"): templates[f].append(q)
    stats["templates"] = {f: len(v) for f, v in templates.items()}
    log(f"templates: {stats['templates']} (from {len(cands)} candidates)")
    del llm

    # 4. embedding check on CPU: drop facts that contradict agreeing nearest neighbours
    import torch
    from transformers import AutoModel, AutoTokenizer
    et = AutoTokenizer.from_pretrained(a.embed, padding_side="left"); em = AutoModel.from_pretrained(a.embed).eval()
    texts = [f"{c['name']}, one of the {c['category']}" for c in concepts]
    vecs = []
    with torch.no_grad():
        for i in range(0, len(texts), 64):
            b = et(texts[i:i + 64], padding=True, return_tensors="pt")
            vecs.append(torch.nn.functional.normalize(em(**b).last_hidden_state[:, -1], dim=-1))   # last-token pooling
    V = torch.cat(vecs); sim = V @ V.T; sim.fill_diagonal_(-1)
    nn = sim.topk(min(10, len(concepts) - 1), dim=1).indices.tolist()
    dropped = 0
    for ci, c in enumerate(concepts):
        c["near"] = [concepts[j]["name"] for j in nn[ci][:5]]
        for pid, *_ in PROPS:
            val = "yes" if pid in c["yes"] else "no" if pid in c["no"] else None
            if val is None: continue
            nv = ["yes" if pid in concepts[j]["yes"] else "no" if pid in concepts[j]["no"] else None for j in nn[ci]]
            nv = [x for x in nv if x]
            if len(nv) >= 8 and sum(x != val for x in nv) >= 0.8 * len(nv):
                c[val].remove(pid); c.setdefault("flagged", []).append(f"{pid}={val}"); dropped += 1
    stats["facts_dropped_by_embedding"] = dropped
    log(f"embedding check: dropped {dropped} facts that contradicted agreeing neighbours")

    kb = {"props": [{"id": p, "question": q, "pred": y, "neg": n} for p, q, y, n in PROPS],
          "kinds": CATEGORIES, "concepts": concepts, "templates": templates, "stats": stats,
          "teacher": a.model, "embedder": a.embed, "votes": a.votes}
    with open(a.out + ".json", "w") as f: json.dump(kb, f, indent=1)
    with open(a.out + ".bin", "wb") as f:   # guard: "MKB1", n, props; per concept: name (len + bytes), yes mask, no mask
        f.write(b"MKB1" + struct.pack("<II", len(concepts), len(PROPS)))
        bit = {p[0]: i for i, p in enumerate(PROPS)}
        for c in concepts:
            nm = c["name"].encode()
            f.write(struct.pack("<B", len(nm)) + nm + struct.pack("<QQ", sum(1 << bit[p] for p in c["yes"]), sum(1 << bit[p] for p in c["no"])))
    log(f"wrote {a.out}.json and {a.out}.bin: {len(concepts)} concepts, {sum(len(c['yes']) + len(c['no']) for c in concepts)} facts")
    print("DONE", flush=True)

if __name__ == "__main__":
    main()

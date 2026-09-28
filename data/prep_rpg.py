# Streams a corpus of RPG-style dialogue to stdout, built from public datasets on the Hugging Face Hub:
#   LIGHT, LIGHT-Wild  in-character fantasy dialogue with a scene setting (Urbanek et al. 2019; Shuster et al. 2021)
#   CRD3               Critical Role D&D session transcripts, CC BY-SA 4.0 (Rameshkumar & Bailey 2020)
# Nothing is written to disk: downloads are held in memory and the text goes straight into moth:
#   python3 data/prep_rpg.py | ./moth -
# Documents are deduplicated and shuffled with a fixed seed, so moth's validation split (the last 10% of the
# stream) is a random sample of all three sources rather than the end of one of them. About 100 MB of text.
# needs: pip install pyarrow
import io, json, random, sys, urllib.request, zipfile
import pyarrow.parquet as pq

HF = "https://huggingface.co/datasets/"

def fetch(url):
    print("downloading", url, file=sys.stderr)
    with urllib.request.urlopen(url) as r: return io.BytesIO(r.read())

def clean(s):
    return " ".join(s.replace("\r", " ").split())

def light(repo):
    docs, seen = [], set()
    for split in ("train", "valid", "test"):
        for r in pq.read_table(fetch(f"{HF}{repo}/resolve/main/data/{split}-00000-of-00001.parquet")).to_pylist():
            turns = [clean(t) for t in (r["dialogue"] or []) if t and clean(t)]
            if len(turns) < 2 or tuple(turns) in seen: continue
            seen.add(tuple(turns))
            ch, st = r["characters"] or {}, r["setting"] or {}
            me, you = clean(ch.get("self_name") or "someone"), clean(ch.get("partner_name") or "someone")
            lines = []
            if st.get("name"): lines.append(f"~ {clean(st['name'])}")
            if st.get("description"): lines.append(clean(st["description"]))
            if ch.get("self_persona"): lines.append(f"({me}) {clean(ch['self_persona'])}")
            lines += [f"{me if i % 2 == 0 else you}: {t}" for i, t in enumerate(turns)]   # LIGHT: even turns are self
            docs.append("\n".join(lines))
    return docs

def crd3():
    z = zipfile.ZipFile(fetch(HF + "microsoft/crd3/resolve/main/data/aligned%20data.zip"))
    eps = {}                                  # episode -> {turn number: line}; the c=2 files hold every turn
    for name in z.namelist():
        if "/c=2/" not in name or not name.endswith(".json"): continue
        ep = name.rsplit("/", 1)[1].split("_")[0]
        for chunk in json.load(io.TextIOWrapper(z.open(name), encoding="utf-8")):
            for t in chunk["TURNS"]:
                text = clean(" ".join(t["UTTERANCES"]))
                if text: eps.setdefault(ep, {})[t["NUMBER"]] = f"{' and '.join(t['NAMES']).title()}: {text}"
    docs = []
    for ep in sorted(eps):                    # split each episode into ~50-turn scenes so the shuffle mixes sources
        turns = [eps[ep][k] for k in sorted(eps[ep])]
        docs += ["\n".join(turns[i:i + 50]) for i in range(0, len(turns), 50)]
    return docs

if __name__ == "__main__":
    parts = {"light": light("dap-exp/light_dialog"), "light_wild": light("dap-exp/light_dialog_wild"), "crd3": crd3()}
    docs = [d for v in parts.values() for d in v]
    random.Random(1234).shuffle(docs)
    for k, v in parts.items(): print(f"{k}: {len(v)} docs, {sum(len(d.encode()) for d in v) / 1e6:.1f} MB", file=sys.stderr)
    out = sys.stdout.buffer
    for i, d in enumerate(docs): out.write(((i and "\n\n" or "") + d).encode())
    out.write(b"\n"); out.flush()

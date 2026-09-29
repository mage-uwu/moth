# Scores a moth checkpoint on the eval set that data/kb_tasks.py --eval writes (held-out facts, unseen logic):
#   python3 data/kb_eval.py ev.jsonl ./moth run.ck
# An answer counts when it starts with the expected one ("Yes", "No", "We cannot tell from what we know").
import collections, json, os, subprocess, sys, tempfile

def main():
    ev, moth, ck = sys.argv[1:4]
    items = [json.loads(l) for l in open(ev) if l.strip()]
    with tempfile.NamedTemporaryFile("w", suffix=".txt", delete=False) as f:
        f.write("\n".join(i["prompt"].replace("\n", "\\n") for i in items) + "\n"); qpath = f.name
    out = subprocess.run([moth, "-g", ck, "-q", qpath], capture_output=True, text=True, errors="replace", check=True)
    os.unlink(qpath)
    answers = out.stdout.split("\n")[:-1][-len(items):]   # the last lines: one per prompt
    right, total = collections.Counter(), collections.Counter()
    for i, got in zip(items, answers):
        total[i["type"]] += 1; right[i["type"]] += got.strip().startswith(i["answer"])
    for t in total: print(f"{t:14s} {right[t] / total[t]:6.1%}  ({right[t]}/{total[t]})")
    base = collections.Counter(i["answer"] for i in items if i["type"] == "heldout_fact")
    if base: print(f"{'(majority)':14s} {max(base.values()) / sum(base.values()):6.1%}  always answering '{base.most_common(1)[0][0]}'")
    print(out.stderr.strip().splitlines()[-1])
    for i, got in list(zip(items, answers))[:4]: print(f"  {i['prompt'].splitlines()[0][3:]} -> {got.strip()!r} (want {i['answer']!r})")

if __name__ == "__main__":
    main()

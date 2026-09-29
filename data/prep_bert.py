# Streams a BERT-style pretraining corpus to stdout: English Wikipedia plus public-domain books, mixed about
# 3:1 as BERT's Wikipedia + BookCorpus was. BookCorpus itself was scraped without its authors' permission, so
# the books are PG-19 instead (Project Gutenberg books published before 1919; DeepMind, Apache 2.0).
#   python3 data/prep_bert.py --wiki-mb 300 --books-mb 100 | ./bmoth - -o bert.ck
# Nothing touches disk. Articles and books are cut into ~8 KB pieces at paragraph breaks, book paragraphs are
# reflowed (Gutenberg wraps lines at ~70 columns), and pieces are shuffled with a fixed seed, so the last 10%
# (the validation split) samples both sources.
import argparse, concurrent.futures, io, random, sys, urllib.request
import pyarrow.parquet as pq

WIKI = "https://huggingface.co/datasets/wikimedia/wikipedia/resolve/main/20231101.en/train-{:05d}-of-00041.parquet"
PG_LIST = "https://huggingface.co/datasets/deepmind/pg19/resolve/main/data/train_files.txt"
PG = "https://storage.googleapis.com/deepmind-gutenberg/"

def get(url):
    with urllib.request.urlopen(url, timeout=120) as r: return r.read()

def pieces(paras, size=8000):                 # group paragraphs into ~size-byte pieces
    out, cur, n = [], [], 0
    for p in paras:
        cur.append(p); n += len(p) + 2
        if n >= size: out.append("\n\n".join(cur)); cur, n = [], 0
    if cur: out.append("\n\n".join(cur))
    return out

def wiki(mb):
    docs, n, shard = [], 0, 0
    while n < mb * 1e6:
        print(f"wikipedia shard {shard}", file=sys.stderr)
        t = pq.read_table(io.BytesIO(get(WIKI.format(shard))), columns=["title", "text"])
        for title, text in zip(t.column("title").to_pylist(), t.column("text").to_pylist()):
            if len(text) < 500: continue       # stubs and redirects
            paras = [title] + [p.strip() for p in text.split("\n") if p.strip()]
            for d in pieces(paras): docs.append(d); n += len(d)
            if n >= mb * 1e6: break
        shard += 1
    return docs

def book(path):
    text = get(PG + path).decode("utf-8", "replace").replace("\r", "")
    paras = [" ".join(p.split()) for p in text.split("\n\n")]
    return pieces([p for p in paras if p])

def books(mb, seed):
    files = get(PG_LIST).decode().split()
    random.Random(seed).shuffle(files)
    docs, n, i = [], 0, 0
    with concurrent.futures.ThreadPoolExecutor(8) as ex:
        while n < mb * 1e6 and i < len(files):
            for ps in ex.map(book, files[i:i + 16]):
                docs += ps; n += sum(len(p) for p in ps)
            i += 16
    print(f"pg-19: {i} books", file=sys.stderr)
    return docs

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--wiki-mb", type=float, default=300); ap.add_argument("--books-mb", type=float, default=100)
    ap.add_argument("--seed", type=int, default=1234)
    a = ap.parse_args()
    w, b = wiki(a.wiki_mb), books(a.books_mb, a.seed)
    docs = w + b; random.Random(a.seed).shuffle(docs)
    print(f"wikipedia {sum(map(len, w)) / 1e6:.1f} MB, books {sum(map(len, b)) / 1e6:.1f} MB, {len(docs)} pieces", file=sys.stderr)
    out = sys.stdout.buffer
    for i, d in enumerate(docs): out.write(((i and "\n\n" or "") + d).encode())
    out.write(b"\n"); out.flush()

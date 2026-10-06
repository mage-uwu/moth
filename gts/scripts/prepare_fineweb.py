# Golden Tree Snake (GTS) fork, 2026.
"""Write a small slice of FineWeb as GPT-2 tokens for scripts/lm_run.py.

    pip install datasets tiktoken
    python scripts/prepare_fineweb.py --out data/fineweb --train-tokens 20000000 --val-tokens 500000

Streams the ``sample-10BT`` subset of ``HuggingFaceFW/fineweb``, so nothing large is downloaded. The first
documents go to ``val.bin`` and the following ones to ``train.bin``, both flat arrays of uint16 token ids, with a
``meta.json`` beside them. A 2,000-step run at batch 8 x 512 tokens reads about 8 million tokens.

Two offline modes exist for smoke tests: ``--text-file FILE`` tokenises a local text file with the same tokenizer,
and ``--bytes FILE`` uses raw bytes as tokens (vocabulary 256, no tokenizer needed).

The FineWeb path has not been run by the author of this script: it was written without network access.
"""

import argparse
import json
import os

import numpy as np


def write_split(chunks, path, limit):
    """Consume token lists from ``chunks`` until ``limit`` tokens are written. Returns the number written."""
    written = 0
    with open(path, "wb") as f:
        for tokens in chunks:
            tokens = tokens[: limit - written]
            np.asarray(tokens, dtype=np.uint16).tofile(f)
            written += len(tokens)
            if written >= limit:
                break
    return written


def fineweb_chunks(subset):
    import tiktoken
    from datasets import load_dataset

    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token  # 50256, written before every document
    for doc in load_dataset("HuggingFaceFW/fineweb", name=subset, split="train", streaming=True):
        yield [eot] + enc.encode_ordinary(doc["text"])


def text_chunks(path, chunk_chars=200_000):
    import tiktoken

    enc = tiktoken.get_encoding("gpt2")
    text = open(path, encoding="utf-8", errors="replace").read()
    for i in range(0, len(text), chunk_chars):
        yield enc.encode_ordinary(text[i : i + chunk_chars])


def byte_chunks(path, chunk=200_000):
    data = open(path, "rb").read()
    for i in range(0, len(data), chunk):
        yield list(data[i : i + chunk])


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--out", default="data/fineweb")
    p.add_argument("--train-tokens", type=int, default=20_000_000)
    p.add_argument("--val-tokens", type=int, default=500_000)
    p.add_argument("--subset", default="sample-10BT")
    p.add_argument("--text-file")
    p.add_argument("--bytes")
    args = p.parse_args()

    os.makedirs(args.out, exist_ok=True)
    if args.bytes:
        chunks, vocab, source = byte_chunks(args.bytes), 256, f"bytes of {args.bytes}"
    elif args.text_file:
        chunks, vocab, source = text_chunks(args.text_file), 50257, f"gpt2 tokens of {args.text_file}"
    else:
        chunks, vocab, source = fineweb_chunks(args.subset), 50257, f"gpt2 tokens of HuggingFaceFW/fineweb {args.subset}"
    n_val = write_split(chunks, os.path.join(args.out, "val.bin"), args.val_tokens)
    n_train = write_split(chunks, os.path.join(args.out, "train.bin"), args.train_tokens)
    json.dump({"vocab_size": vocab, "source": source, "train_tokens": n_train, "val_tokens": n_val},
              open(os.path.join(args.out, "meta.json"), "w"), indent=1)
    print(f"{source}: wrote {n_train:,} training and {n_val:,} validation tokens to {args.out}")
    if n_train < args.train_tokens:
        print("note: the source ran out before --train-tokens was reached")


if __name__ == "__main__":
    main()

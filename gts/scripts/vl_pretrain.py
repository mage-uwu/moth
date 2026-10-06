# Golden Tree Snake (GTS) fork, 2026.
"""Vision-language pretraining of a ternary GTS vision backbone (~38M), with a sidecar into the GTS masked LM.

    python scripts/vl_pretrain.py prep     --out /root/vl --work /root/raw
    python scripts/vl_pretrain.py prep-imagenet --out /root/imnet --work /root/raw     # needs HF_TOKEN (gated set)
    python scripts/vl_pretrain.py train    --data /root/vl --lm binarized.pt --out /root/run --minutes 600
    python scripts/vl_pretrain.py probe    --ckpt /root/run/checkpoint.pt --data /root/imnet --out /root/run

prep: image-caption pairs from
      - ShareGPT4V-PT (Lin-Chen/ShareGPT4V, share-captioner_coco_lcs_sam_1246k): its COCO train2017 and LLaVA
        LCS-558K images (SAM is skipped: terabytes), --sharegpt4v pairs (all of COCO, the rest from LCS);
      - VideoGameBunny (VideoGameBunny/Dataset): one caption per image (a long caption, else a short one, else the
        description inside the image-to-JSON answer; question answering is left out), --vgb images.
      Every image is resized (shorter side --size) and centre-cropped to --size x --size, stored as JPEG. Captions are
      BERT uncased WordPiece (the GTS masked LM's vocabulary), cut at --cap-len tokens. One directory per source with
      images.bin (JPEG bytes back to back), offsets.npy, caps.npy (uint16, zero-padded) and meta.json.
prep-imagenet: an ImageNet-1k subset for linear probes: the first --train-shards parquet shards of the training split
      (about 4,400 images each, in random class order) and the whole validation split, resized to 256 and
      centre-cropped to 224 as JPEG, with labels.
train: the backbone (mamba_ssm/models/gts_vision.py) under two losses, from the same batch:
      - contrastive (CLIP-style, symmetric InfoNCE with a learned temperature) between the mean-pooled image tokens
        and the frozen GTS masked LM's mean-pooled caption states (precomputed once), each through a linear map;
      - masked caption modelling through the frozen LM: the sidecar turns the image tokens into --img-tokens vectors
        placed after [CLS], then the caption with --cap-mask-prob of its tokens masked (80/10/10); the LM's own head
        predicts them. Gradients reach the sidecar and the backbone through the frozen LM.
      Augmentation on the GPU: JPEGs decoded on the GPU, random resized crops (scale --crop-min to 1) to 224. bf16,
      torch.compile per block, fused AdamW; schedule fitted to --minutes as in bert_pretrain.py. Held-out pairs give
      image-to-text and text-to-image recall at 1 among 1,000, and the caption loss with the right image against a
      shuffled one (how much the LM uses the image). Writes checkpoint.pt (float, with optimizer), backbone.pt (the
      backbone and sidecar alone), binarized.pt (the backbone's ternary weights as 2-bit codes) and result.json.
probe: ImageNet-1k linear probe of the frozen backbone: mean-pooled features of the 224 centre crops, standardised,
      and a softmax classifier trained on them with AdamW; top-1 and top-5 on the 50,000 validation images.
"""

import argparse
import gc
import io
import json
import math
import os
import random
import subprocess
import sys
import threading
import time
import zipfile

import numpy as np
import torch
import torch.nn.functional as F

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from mamba_ssm.models.gts_encoder import GTSConfig, GTSForMaskedLM  # noqa: E402
from mamba_ssm.models.gts_vision import GTSVision, GTSVisionConfig, VisionSidecar, config_dict, lm_with_image  # noqa: E402
from mamba_ssm.utils.ternary_pack import load_binarized, save_binarized  # noqa: E402

MEAN = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1) * 255
STD = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1) * 255
CLS, SEP, MASK, PAD = 101, 102, 103, 0


# ----------------------------------------------------------------------------------------------------------- prep

_ZIPS = {}


def _read(src):
    kind, path, member = src
    if kind == "zip":
        z = _ZIPS.get(path)
        if z is None:
            z = _ZIPS[path] = zipfile.ZipFile(path)
        return z.read(member)
    if kind == "bytes":
        return member
    with open(path, "rb") as f:
        data = f.read()
        _drop_cache(f.fileno())
        return data


def _drop_cache(fd):
    """Tell the kernel this file's cached pages are no longer needed: the container's memory limit counts the page
    cache, and reading or writing tens of GB of images otherwise fills it (two prep runs were OOM-killed so)."""
    try:
        os.posix_fadvise(fd, 0, 0, os.POSIX_FADV_DONTNEED)
    except (AttributeError, OSError):
        pass


def _square_jpeg(job):
    """(source, size, quality, short) -> JPEG bytes: shorter side resized to ``short``, centre size x size crop; None if
    the image does not decode."""
    src, size, quality, short = job
    from PIL import Image

    try:
        im = Image.open(io.BytesIO(_read(src)))
        im.draft("RGB", (size, size))
        im = im.convert("RGB")
        w, h = im.size
        s = short / min(w, h)
        im = im.resize((max(short, round(w * s)), max(short, round(h * s))), Image.BICUBIC, reducing_gap=3.0)
        w, h = im.size
        left, top = (w - size) // 2, (h - size) // 2
        im = im.crop((left, top, left + size, top + size))
        out = io.BytesIO()
        im.save(out, "JPEG", quality=quality)
        return out.getvalue()
    except Exception:
        return None


class Writer:
    """Appends resized JPEGs (made by a process pool) and their captions or labels to a prep directory."""

    def __init__(self, out, size, workers, quality=90, short=None):
        import multiprocessing as mp

        os.makedirs(out, exist_ok=True)
        self.out, self.size, self.quality, self.short = out, size, quality, short or size
        self.f = open(os.path.join(out, "images.bin"), "wb")
        # spawned, not forked: a forked worker slowly copies the parent's memory (the caption JSON, ~10 GB of Python
        # objects) as reference counts touch it, and eight of them run out of memory
        self.pool = mp.get_context("spawn").Pool(workers, maxtasksperchild=500)
        self.offsets, self.targets, self.seen, self.t0 = [0], [], 0, time.time()

    def add(self, items, chunk=20000):
        """items: list of (source, caption text or label). Processed ``chunk`` at a time, so at most one chunk of
        results is ever held."""
        for c in range(0, len(items), chunk):
            part = items[c : c + chunk]
            jobs = [(src, self.size, self.quality, self.short) for src, _ in part]
            for (_, target), data in zip(part, self.pool.map(_square_jpeg, jobs, chunksize=32)):
                self.seen += 1
                if data is not None:
                    self.f.write(data)
                    self.offsets.append(self.offsets[-1] + len(data))
                    self.targets.append(target)
            self.f.flush()
            os.fsync(self.f.fileno())
            _drop_cache(self.f.fileno())
            if self.seen % 60000 < chunk:
                print(f"    {self.seen:,} images, {self.seen / (time.time() - self.t0):.0f}/s; {_memory()}", flush=True)

    def close(self, cap_len=None, tok=None, extra=None):
        self.f.close()
        self.pool.close()
        out, n = self.out, len(self.targets)
        np.save(os.path.join(out, "offsets.npy"), np.asarray(self.offsets, dtype=np.int64))
        if tok is not None:
            _save_caps(out, self.targets, cap_len, tok)
        else:
            np.save(os.path.join(out, "labels.npy"), np.asarray(self.targets, dtype=np.int16))
        meta = {"n": n, "dropped": self.seen - n, "size": self.size, "cap_len": cap_len, "bytes": self.offsets[-1], **(extra or {})}
        json.dump(meta, open(os.path.join(out, "meta.json"), "w"), indent=1)
        print(f"  {out}: {n:,} images ({self.seen - n} dropped), {self.offsets[-1] / 2**30:.1f} GB, "
              f"{time.time() - self.t0:.0f} s", flush=True)


def _memory():
    """The container's memory use (cgroup), this process's resident set and the system's available memory, in GB."""
    out = []
    for path in ("/sys/fs/cgroup/memory.current", "/sys/fs/cgroup/memory/memory.usage_in_bytes"):
        if os.path.exists(path):
            out.append(f"cgroup {int(open(path).read()) / 2**30:.1f}")
            break
    for line in open("/proc/self/status"):
        if line.startswith("VmRSS"):
            out.append(f"rss {int(line.split()[1]) / 2**20:.1f}")
    for line in open("/proc/meminfo"):
        if line.startswith("MemAvailable"):
            out.append(f"available {int(line.split()[1]) / 2**20:.0f}")
    return "memory GB: " + ", ".join(out)


def _write_set(out, items, size, workers, cap_len=None, tok=None, extra=None):
    w = Writer(out, size, workers)
    w.add(items)
    w.close(cap_len, tok, extra)


def _save_caps(out, texts, cap_len, tok):
    caps = np.zeros((len(texts), cap_len), dtype=np.uint16)
    for start in range(0, len(texts), 4096):
        for j, enc in enumerate(tok.encode_batch(texts[start : start + 4096], add_special_tokens=False)):
            ids = enc.ids[:cap_len]
            caps[start + j, : len(ids)] = ids
    np.save(os.path.join(out, "caps.npy"), caps)


def _clean(text):
    return text.replace("<image>", "").strip()


def _vgb_caption(conv):
    """(kind, text) for a VideoGameBunny conversation: 'long'/'short' caption, 'json' description, or None (QA)."""
    human, gpt = _clean(conv[0]["value"]), conv[1]["value"].strip()
    if "json" in human.lower():
        try:
            obj = json.loads(json.loads(gpt) if gpt.startswith('"') else gpt)
            desc = obj.get("description") if isinstance(obj, dict) else None
            return ("json", desc) if isinstance(desc, str) and len(desc) > 40 else None
        except Exception:
            return None
    if human.endswith("?") and not human.lower().startswith(("can you", "could you", "what is this", "what's this")):
        return None  # a question about the image
    return ("long" if len(gpt) > 400 else "short", gpt)


def _hf(repo, name, work):
    from huggingface_hub import hf_hub_download

    return hf_hub_download(repo, name, repo_type="dataset", local_dir=work)


def _vgb_select(path, n):
    """Run in a child process (the 3 GB JSON never enters the parent): {image: (kind, caption)} for n images."""
    conv = json.load(open(path))
    best, rank = {}, {"long": 0, "short": 1, "json": 2}
    for e in conv:
        c = _vgb_caption(e["conversations"])
        if c and (e["image"] not in best or rank[c[0]] < rank[best[e["image"]][0]]):
            best[e["image"]] = c
    names = sorted(best)
    random.Random(0).shuffle(names)
    return len(best), {k: best[k] for k in names[:n]}


def _sharegpt4v_select(path, n):
    """Run in a child process: [(image path, caption)], all of COCO first, then LCS to n."""
    data = json.load(open(path))
    coco = [(e["image"], _clean(e["conversations"][1]["value"])) for e in data if e["image"].startswith("coco/")]
    lcs = [(e["image"], _clean(e["conversations"][1]["value"])) for e in data if e["image"].startswith("llava/")]
    random.Random(0).shuffle(lcs)
    return len(coco), len(lcs), coco + lcs[: max(0, n - len(coco))]


def _in_child(fn, *args):
    from concurrent.futures import ProcessPoolExecutor
    import multiprocessing as mp

    with ProcessPoolExecutor(1, mp_context=mp.get_context("spawn")) as ex:
        return ex.submit(fn, *args).result()


def _find_images(root):
    """{file name: path} for every image under root."""
    out = {}
    for d, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith((".jpg", ".jpeg", ".png")):
                out[f] = os.path.join(d, f)
    return out


def prep(a):
    from bert_pretrain import _tokenizer

    tok = _tokenizer()
    os.makedirs(a.work, exist_ok=True)
    t0 = time.time()
    print(_memory(), flush=True)

    # VideoGameBunny: one caption per image, preferring long captions, then short ones, then JSON descriptions
    out = os.path.join(a.out, "vgb")
    if a.vgb and not os.path.exists(os.path.join(out, "meta.json")):
        n_all, best = _in_child(_vgb_select, _hf("VideoGameBunny/Dataset", "conversations.json", a.work), a.vgb)
        kinds = {k: sum(v[0] == k for v in best.values()) for k in ("long", "short", "json")}
        print(f"VideoGameBunny: {n_all:,} images with a caption; using {len(best):,} ({kinds}); {_memory()}", flush=True)
        parts = [_hf("VideoGameBunny/Dataset", f"images.z0{i}", a.work) for i in range(1, 6)]
        parts.append(_hf("VideoGameBunny/Dataset", "images.zip", a.work))
        dest = os.path.join(a.work, "vgb_images")
        subprocess.run(["7z", "x", "-y", "-bd", "-o" + dest, parts[-1]], check=True, stdout=subprocess.DEVNULL)
        for p in parts:
            os.remove(p)
        subprocess.run(["sync"])
        items = [(("file", os.path.join(dest, n.lstrip("./")), None), c[1]) for n, c in best.items()]
        _write_set(out, items, a.size, a.workers, cap_len=a.cap_len, tok=tok, extra={"source": "VideoGameBunny/Dataset", "kinds": kinds})
        subprocess.run(["rm", "-rf", dest])

    # ShareGPT4V-PT: COCO train2017 (all of it) and LCS-558K (the rest), unzipped to disk first
    out = os.path.join(a.out, "sharegpt4v")
    if a.sharegpt4v and not os.path.exists(os.path.join(out, "meta.json")):
        n_coco, n_lcs, chosen = _in_child(_sharegpt4v_select, _hf("Lin-Chen/ShareGPT4V", "share-captioner_coco_lcs_sam_1246k_1107.json", a.work), a.sharegpt4v)
        print(f"ShareGPT4V-PT: {n_coco:,} COCO and {n_lcs:,} LCS captions; using {len(chosen):,}; {_memory()}", flush=True)
        coco_zip = os.path.join(a.work, "train2017.zip")
        if not os.path.exists(coco_zip):
            subprocess.run(["curl", "-fsSL", "--retry", "5", "-o", coco_zip,
                            "http://images.cocodataset.org/zips/train2017.zip"], check=True)
        lcs_zip = _hf("liuhaotian/LLaVA-Pretrain", "images.zip", a.work)
        dest = os.path.join(a.work, "s4v_images")
        for z, sub in ((coco_zip, "coco"), (lcs_zip, "lcs")):
            os.makedirs(os.path.join(dest, sub), exist_ok=True)
            subprocess.run(["unzip", "-q", "-o", z, "-d", os.path.join(dest, sub)], check=True)
            os.remove(z)
            subprocess.run(["sync"])
            print(f"  unzipped {sub}; {_memory()}", flush=True)
        where = _find_images(dest)
        items, missing = [], 0
        for image, caption in chosen:
            path = where.get(os.path.basename(image))
            if path is None:
                missing += 1
                continue
            items.append((("file", path, None), caption))
        del chosen, where
        print(f"  {missing} captions without their image; {_memory()}", flush=True)
        _write_set(out, items, a.size, a.workers, cap_len=a.cap_len, tok=tok,
                   extra={"source": "Lin-Chen/ShareGPT4V share-captioner_coco_lcs_sam_1246k_1107 (COCO + LCS)",
                          "coco": sum("/coco/" in it[0][1] for it in items)})
        subprocess.run(["rm", "-rf", dest])
    print(f"prep done in {(time.time() - t0) / 60:.0f} min", flush=True)


def prep_imagenet(a):
    """Validation: all 50,000 images. Training: shards in turn, keeping up to --per-class images of each class (the
    shards' class order is not relied on), until every class is full or the shards run out."""
    import pyarrow.parquet as pq
    from huggingface_hub import HfApi

    files = sorted(f for f in HfApi().list_repo_files("ILSVRC/imagenet-1k", repo_type="dataset") if f.endswith(".parquet"))
    for split, names, per_class in (("val", [f for f in files if "/validation-" in f], None),
                                    ("train", [f for f in files if "/train-" in f], a.per_class)):
        out = os.path.join(a.out, split)
        if os.path.exists(os.path.join(out, "meta.json")):
            continue
        w, count, t0 = Writer(out, 224, a.workers, quality=95, short=256), np.zeros(1000, dtype=np.int64), time.time()
        for k, n in enumerate(names):
            path = _hf("ILSVRC/imagenet-1k", n, a.work)
            t = pq.read_table(path, columns=["image", "label"])
            items = []
            for img, label in zip(t.column("image").to_pylist(), t.column("label").to_pylist()):
                if per_class is None or count[label] < per_class:
                    count[label] += 1
                    items.append((("bytes", None, img["bytes"]), label))
            del t
            os.remove(path)
            w.add(items)
            print(f"  ImageNet {split}: shard {k + 1}/{len(names)}, {int(count.sum()):,} images, "
                  f"{int((count >= (per_class or 1)).sum())} classes full, {time.time() - t0:.0f} s", flush=True)
            if per_class is not None and (count >= per_class).all():
                break
        w.close(extra={"source": "ILSVRC/imagenet-1k " + split, "per_class": per_class, "shards_read": k + 1})


# ---------------------------------------------------------------------------------------------------------- train


class ImageSet:
    """One prep directory: JPEG bytes by index, captions or labels."""

    def __init__(self, d):
        self.meta = json.load(open(os.path.join(d, "meta.json")))
        self.blob = np.memmap(os.path.join(d, "images.bin"), dtype=np.uint8, mode="r")
        self.off = np.load(os.path.join(d, "offsets.npy"))
        self.caps = np.load(os.path.join(d, "caps.npy")) if os.path.exists(os.path.join(d, "caps.npy")) else None
        self.labels = np.load(os.path.join(d, "labels.npy")) if os.path.exists(os.path.join(d, "labels.npy")) else None
        self.n = len(self.off) - 1

    def jpeg(self, i):
        return torch.from_numpy(np.array(self.blob[self.off[i] : self.off[i + 1]]))


class Pairs:
    """Several ImageSets as one list of pairs; index j -> (set, row)."""

    def __init__(self, dirs):
        self.sets = [ImageSet(d) for d in dirs]
        self.starts = np.cumsum([0] + [s.n for s in self.sets])
        self.n = int(self.starts[-1])
        self.caps = np.concatenate([s.caps for s in self.sets])

    def jpegs(self, idx):
        out = []
        for j in idx:
            k = int(np.searchsorted(self.starts, j, side="right") - 1)
            out.append(self.sets[k].jpeg(int(j - self.starts[k])))
        return out


def decode(jpegs, device):
    """JPEG bytes -> uint8 (B, 3, S, S) on ``device`` (nvJPEG on a GPU)."""
    from torchvision.io import decode_jpeg

    if device == "cuda":
        return torch.stack(decode_jpeg(jpegs, device="cuda"))
    return torch.stack([decode_jpeg(j) for j in jpegs])


def augment(x, out, crop_min, gen):
    """Random resized crops (area crop_min..1, aspect 3/4..4/3) of a uint8 batch, to out x out, normalised."""
    from torchvision.ops import roi_align

    B, _, S, _ = x.shape
    area = torch.empty(B).uniform_(crop_min, 1.0, generator=gen) * S * S
    logr = torch.empty(B).uniform_(math.log(3 / 4), math.log(4 / 3), generator=gen)
    w = torch.sqrt(area * torch.exp(logr)).clamp(max=S)
    h = torch.sqrt(area / torch.exp(logr)).clamp(max=S)
    x0 = torch.rand(B, generator=gen) * (S - w)
    y0 = torch.rand(B, generator=gen) * (S - h)
    boxes = torch.stack([torch.arange(B, dtype=torch.float), x0, y0, x0 + w, y0 + h], 1).to(x.device)
    y = roi_align(x.float(), boxes, output_size=out, spatial_scale=1.0, sampling_ratio=2, aligned=True)
    return (y - MEAN.to(x.device)) / STD.to(x.device)


def normalise(x, out=None):
    x = x.float()
    if out is not None and x.shape[-1] != out:
        x = F.interpolate(x, size=(out, out), mode="bilinear", antialias=True, align_corners=False)
    return (x - MEAN.to(x.device)) / STD.to(x.device)


def load_lm(path, device):
    blob = torch.load(path, map_location="cpu", weights_only=False)
    lm = GTSForMaskedLM(GTSConfig(**blob["config"]))
    if "model" in blob:
        lm.load_state_dict(blob["model"])
    else:
        load_binarized(blob, lm)
    lm = lm.to(device).eval()
    for p in lm.parameters():
        p.requires_grad_(False)
    return lm, blob["config"]


def caption_batch(caps, n_img, mask_prob, gen, vocab):
    """caps (B, C) uint16 -> input ids [CLS] + n_img placeholders + caption + [SEP] (fixed length 2 + n_img + C) and
    labels at the masked caption positions (BERT's 80/10/10)."""
    B, C = caps.shape
    c = torch.from_numpy(caps.astype(np.int64))
    lens = (c != PAD).sum(1)
    ids = torch.zeros(B, 2 + n_img + C, dtype=torch.long)
    ids[:, 0] = CLS
    ids[:, 1 : 1 + n_img] = MASK  # placeholders, replaced by the image vectors
    ids[:, 1 + n_img : 1 + n_img + C] = c
    ids[torch.arange(B), 1 + n_img + lens] = SEP
    pos = torch.zeros_like(ids, dtype=torch.bool)
    pos[:, 1 + n_img : 1 + n_img + C] = c != PAD
    chosen = pos & (torch.rand(ids.shape, generator=gen) < mask_prob)
    labels = torch.where(chosen, ids, torch.full_like(ids, -100))
    r = torch.rand(ids.shape, generator=gen)
    inputs = ids.clone()
    inputs[chosen & (r < 0.8)] = MASK
    rand = chosen & (r >= 0.8) & (r < 0.9)
    inputs[rand] = torch.randint(999, vocab, (int(rand.sum()),), generator=gen)
    return inputs, labels


@torch.no_grad()
def text_embeddings(lm, caps, device, batch=512):
    """Mean of the frozen LM's final states over [CLS] + caption + [SEP], for every caption: (N, d) float16."""
    out = []
    C = caps.shape[1]
    for s in range(0, len(caps), batch):
        c = torch.from_numpy(caps[s : s + batch].astype(np.int64))
        lens = (c != PAD).sum(1)
        ids = torch.zeros(len(c), C + 2, dtype=torch.long)
        ids[:, 0] = CLS
        ids[:, 1 : C + 1] = c
        ids[torch.arange(len(c)), 1 + lens] = SEP
        ids = ids.to(device)
        mask = ids != PAD
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=device == "cuda"):
            h = lm.backbone(ids, attention_mask=mask).float()
        out.append(((h * mask.unsqueeze(-1)).sum(1) / mask.sum(1, keepdim=True)).half())
    return torch.cat(out)


class VLHeads(torch.nn.Module):
    def __init__(self, d_vision, d_lm, d_joint=512, img_tokens=49):
        super().__init__()
        self.img = torch.nn.Linear(d_vision, d_joint, bias=False)
        self.txt = torch.nn.Sequential(torch.nn.LayerNorm(d_lm), torch.nn.Linear(d_lm, d_joint, bias=False))
        self.logit_scale = torch.nn.Parameter(torch.tensor(math.log(1 / 0.07)))
        self.sidecar = VisionSidecar(d_vision, d_lm, img_tokens)


def train(a):
    t_start = time.time()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    a.amp = a.amp and device == "cuda"
    torch.backends.cuda.matmul.allow_tf32 = torch.backends.cudnn.allow_tf32 = True
    torch.manual_seed(a.seed)
    data = Pairs(a.data)
    perm = np.random.default_rng(0).permutation(data.n)
    held, train_idx = perm[: a.held_out], perm[a.held_out :]
    lm, lm_cfg = load_lm(a.lm, device)
    vocab = lm_cfg["vocab_size"]
    print(f"{data.n:,} pairs ({', '.join(s.meta.get('source', '?')[:40] + f': {s.n:,}' for s in data.sets)}); "
          f"{a.held_out} held out", flush=True)
    t0 = time.time()
    temb = text_embeddings(lm, data.caps, device)
    print(f"caption embeddings from the frozen LM: {tuple(temb.shape)} in {time.time() - t0:.0f} s", flush=True)

    vcfg = GTSVisionConfig(image_size=a.image_size, d_model=a.width, n_layer=a.layers, deep_depth=a.deep_depth,
                           deep_trees=a.deep_trees, bank_trees=a.bank_trees)
    model = GTSVision(vcfg).to(device)
    heads = VLHeads(a.width, lm_cfg["d_model"], img_tokens=a.img_tokens).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"GTS vision backbone: {n_params / 1e6:.2f}M parameters; heads and sidecar "
          f"{sum(p.numel() for p in heads.parameters()) / 1e6:.2f}M", flush=True)
    if a.compile and device == "cuda":
        for mods in (model.layers, lm.backbone.layers):
            for i in range(len(mods)):
                mods[i] = torch.compile(mods[i])
    params = list(model.parameters()) + list(heads.parameters())
    decay = [p for p in params if p.ndim >= 2]
    rest = [p for p in params if p.ndim < 2]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": a.weight_decay}, {"params": rest, "weight_decay": 0.0}],
                            lr=a.lr, betas=(0.9, 0.98), eps=1e-6, fused=device == "cuda")
    os.makedirs(a.out, exist_ok=True)
    gen = torch.Generator().manual_seed(a.seed)
    total_steps, curve, last_ckpt, step = None, [], time.time(), 0
    run = {"clip": 0.0, "cap": 0.0, "n": 0}

    def lr_at(s):
        if s < a.warmup:
            return a.lr * (s + 1) / a.warmup
        if total_steps is None:
            return a.lr
        frac = min(1.0, (s - a.warmup) / max(1, total_steps - a.warmup))
        return a.lr * (0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * frac)))

    def losses(idx, x, train_mode=True, shuffle_images=False):
        caps = data.caps[idx]
        inputs, labels = caption_batch(caps, a.img_tokens, a.cap_mask_prob, gen if train_mode else torch.Generator().manual_seed(7), vocab)
        inputs, labels = inputs.to(device), labels.to(device)
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=a.amp):
            tokens = model(x)
            img = F.normalize(heads.img(tokens.mean(1)).float(), dim=-1)
            txt = F.normalize(heads.txt(temb[torch.as_tensor(idx, device=device)].float()).float(), dim=-1)
            vecs = heads.sidecar(tokens)
            if shuffle_images:
                vecs = vecs.roll(1, 0)
            hidden = lm_with_image(lm, vecs, inputs)
            sel = labels != -100
            logits = lm._head(hidden[sel])
        cap = F.cross_entropy(logits.float(), labels[sel])
        scale = heads.logit_scale.exp().clamp(max=100)
        sim = scale * img @ txt.t()
        target = torch.arange(len(idx), device=device)
        clip = 0.5 * (F.cross_entropy(sim, target) + F.cross_entropy(sim.t(), target))
        return clip, cap, img, txt

    @torch.no_grad()
    def evaluate():
        model.eval()
        imgs, txts, cap, cap_shuf, n = [], [], 0.0, 0.0, 0
        for s in range(0, len(held), a.batch_size):
            idx = held[s : s + a.batch_size]
            x = normalise(decode(data.jpegs(idx), device), a.image_size)
            clip, c, img, txt = losses(idx, x, train_mode=False)
            _, c2, _, _ = losses(idx, x, train_mode=False, shuffle_images=True)
            imgs.append(img), txts.append(txt)
            cap, cap_shuf, n = cap + c.item() * len(idx), cap_shuf + c2.item() * len(idx), n + len(idx)
        img, txt = torch.cat(imgs), torch.cat(txts)
        r_it, r_ti = [], []
        k = min(1000, len(img))
        for s in range(0, len(img) - k + 1, k):  # recall at 1 among 1,000
            sim = img[s : s + k] @ txt[s : s + k].t()
            ar = torch.arange(sim.shape[0], device=device)
            r_it.append((sim.argmax(1) == ar).float().mean().item())
            r_ti.append((sim.argmax(0) == ar).float().mean().item())
        model.train()
        return {"i2t_r1": float(np.mean(r_it)), "t2i_r1": float(np.mean(r_ti)), "cap_loss": cap / n, "cap_loss_shuffled": cap_shuf / n}

    def save(step):
        strip = lambda sd: {k.replace("._orig_mod", ""): v for k, v in sd.items()}  # noqa: E731
        state = {"model": strip(model.state_dict()), "heads": heads.state_dict(), "optimizer": opt.state_dict(),
                 "step": step, "vision_config": config_dict(vcfg), "lm_config": lm_cfg, "args": vars(a), "curve": curve}
        torch.save(state, os.path.join(a.out, "checkpoint.pt.tmp"))
        os.replace(os.path.join(a.out, "checkpoint.pt.tmp"), os.path.join(a.out, "checkpoint.pt"))

    def record(step):
        ev = evaluate()
        tl = {k: run[k] / max(1, run["n"]) for k in ("clip", "cap")}
        curve.append({"step": step, "images": step * a.batch_size, "train_clip": tl["clip"], "train_cap": tl["cap"],
                      "minutes": (time.time() - t_start) / 60, **ev})
        print(f"step {step:6d}  epoch {step * a.batch_size / len(train_idx):5.2f}  i2t R@1 {ev['i2t_r1']:.3f}  t2i R@1 "
              f"{ev['t2i_r1']:.3f}  caption loss {ev['cap_loss']:.3f} (shuffled images {ev['cap_loss_shuffled']:.3f})  "
              f"train clip {tl['clip']:.3f} cap {tl['cap']:.3f}  {(time.time() - t_start) / 60:.1f} min", flush=True)
        json.dump({"params": n_params, "vision_config": config_dict(vcfg), "args": vars(a), "total_steps": total_steps,
                   "data": [s.meta for s in data.sets], "curve": curve}, open(os.path.join(a.out, "result.json"), "w"), indent=1)

    # a background thread reads the next batch's JPEG bytes while the GPU works
    def sample():
        idx = train_idx[np.random.randint(0, len(train_idx), a.batch_size)]
        return idx, data.jpegs(idx)

    np.random.seed(a.seed)
    nxt = [sample()]
    reader = None
    t_rate = None
    model.train()
    while True:
        if step % a.eval_every == 0 and step > 0:
            record(step)
            run = {"clip": 0.0, "cap": 0.0, "n": 0}
        if total_steps is not None and step >= total_steps:
            break
        if reader is not None:
            reader.join()
        idx, jpegs = nxt[0]
        reader = threading.Thread(target=lambda: nxt.__setitem__(0, sample()))
        reader.start()
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        x = augment(decode(jpegs, device), a.image_size, a.crop_min, gen)
        clip, cap, _, _ = losses(idx, x)
        loss = clip + a.cap_weight * cap
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        with torch.no_grad():
            heads.logit_scale.clamp_(0, math.log(100))
        step += 1
        if step % a.log_every == 0:
            run["clip"] += clip.item()
            run["cap"] += cap.item()
            run["n"] += 1
            print(f"  step {step:6d}  clip {clip.item():.4f}  cap {cap.item():.4f}  lr {lr_at(step):.2e}  "
                  f"{(time.time() - t_start) / 60:.1f} min", flush=True)
        if step == a.rate_from:
            if device == "cuda":
                torch.cuda.synchronize()
            t_rate = time.time()
        if step == a.rate_from + a.rate_steps:
            if device == "cuda":
                torch.cuda.synchronize()
            rate = a.rate_steps / (time.time() - t_rate)
            left = a.minutes * 60 - (time.time() - t_start) - a.reserve_minutes * 60
            ev_cost = 2 * a.held_out / a.batch_size / 2  # two forward-only passes over the held-out pairs, per evaluation
            share = ev_cost / (a.eval_every + ev_cost)
            total_steps = step + max(0, int(left * (1 - share) * rate))
            print(f"  {rate:.2f} steps/s = {rate * a.batch_size:,.0f} images/s; schedule fitted to {total_steps} steps "
                  f"({total_steps * a.batch_size / len(train_idx):.1f} epochs of {len(train_idx):,} pairs)", flush=True)
        if time.time() - last_ckpt > a.ckpt_minutes * 60:
            save(step)
            last_ckpt = time.time()
            print(f"  checkpoint at step {step}", flush=True)
    reader.join()
    record(step)
    save(step)
    strip = {k.replace("._orig_mod", ""): v for k, v in model.state_dict().items()}
    plain = GTSVision(vcfg)
    plain.load_state_dict(strip)
    torch.save({"vision_config": config_dict(vcfg), "model": strip, "sidecar": heads.sidecar.state_dict(),
                "lm_config": lm_cfg}, os.path.join(a.out, "backbone.pt"))
    save_binarized(plain, config_dict(vcfg), os.path.join(a.out, "binarized.pt"))
    print(f"saved checkpoint.pt, backbone.pt and binarized.pt to {a.out}; {(time.time() - t_start) / 60:.1f} min in all", flush=True)


# ---------------------------------------------------------------------------------------------------------- probe


@torch.no_grad()
def features(model, s, device, batch=256, amp=True):
    out = []
    for i in range(0, s.n, batch):
        x = normalise(decode([s.jpeg(j) for j in range(i, min(s.n, i + batch))], device))
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp and device == "cuda"):
            out.append(model.features(x).float())
    return torch.cat(out)


def probe(a):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    ck = torch.load(a.ckpt, map_location="cpu", weights_only=False)
    model = GTSVision(GTSVisionConfig(**ck["vision_config"]))
    model.load_state_dict(ck["model"])
    model = model.to(device).eval()
    t0 = time.time()
    tr, va = ImageSet(os.path.join(a.data, "train")), ImageSet(os.path.join(a.data, "val"))
    ftr, fva = features(model, tr, device), features(model, va, device)
    ytr = torch.from_numpy(tr.labels.astype(np.int64)).to(device)
    yva = torch.from_numpy(va.labels.astype(np.int64)).to(device)
    print(f"features: {tuple(ftr.shape)} train, {tuple(fva.shape)} val in {time.time() - t0:.0f} s", flush=True)
    mu, sd = ftr.mean(0), ftr.std(0) + 1e-6
    ftr, fva = (ftr - mu) / sd, (fva - mu) / sd
    best = None
    for wd in a.probe_wd:
        clf = torch.nn.Linear(ftr.shape[1], 1000).to(device)
        opt = torch.optim.AdamW(clf.parameters(), lr=1e-3, weight_decay=wd)
        bs = min(1024, len(ftr))
        steps = a.probe_epochs * math.ceil(len(ftr) / bs)
        sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
        g = torch.Generator(device=device).manual_seed(0)
        for _ in range(steps):
            i = torch.randint(0, len(ftr), (bs,), device=device, generator=g)
            loss = F.cross_entropy(clf(ftr[i]), ytr[i])
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
        with torch.no_grad():
            logits = clf(fva)
            top1 = (logits.argmax(1) == yva).float().mean().item()
            top5 = (logits.topk(5, 1).indices == yva[:, None]).any(1).float().mean().item()
        print(f"  weight decay {wd}: top-1 {top1:.4f}  top-5 {top5:.4f}", flush=True)
        if best is None or top1 > best["top1"]:
            best = {"weight_decay": wd, "top1": top1, "top5": top5}
    res = {"train_images": len(ftr), "val_images": len(fva), "feature_dim": ftr.shape[1], "best": best,
           "note": "linear probe on mean-pooled final tokens, 224 centre crops; the weight decay is chosen on the validation set"}
    json.dump(res, open(os.path.join(a.out, "imagenet_probe.json"), "w"), indent=1)
    print(f"ImageNet-1k linear probe ({len(ftr):,} training images): top-1 {best['top1']:.4f}, top-5 {best['top5']:.4f}", flush=True)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    q = sub.add_parser("prep")
    q.add_argument("--out", required=True)
    q.add_argument("--work", required=True, help="downloads and extracted archives (removed when done)")
    q.add_argument("--sharegpt4v", type=int, default=500_000)
    q.add_argument("--vgb", type=int, default=120_000)
    q.add_argument("--size", type=int, default=256)
    q.add_argument("--cap-len", type=int, default=128)
    q.add_argument("--workers", type=int, default=os.cpu_count())
    q = sub.add_parser("prep-imagenet")
    q.add_argument("--out", required=True)
    q.add_argument("--work", required=True)
    q.add_argument("--per-class", type=int, default=100, help="training images per class for the probe")
    q.add_argument("--workers", type=int, default=os.cpu_count())
    t = sub.add_parser("train")
    t.add_argument("--data", nargs="+", required=True, help="prep directories (e.g. /root/vl/sharegpt4v /root/vl/vgb)")
    t.add_argument("--lm", required=True, help="the GTS masked LM: float checkpoint.pt or binarized.pt")
    t.add_argument("--out", required=True)
    t.add_argument("--minutes", type=float, default=600)
    t.add_argument("--reserve-minutes", type=float, default=6)
    t.add_argument("--image-size", type=int, default=224)
    t.add_argument("--width", type=int, default=512)
    t.add_argument("--layers", type=int, default=17)
    t.add_argument("--bank-trees", type=int, default=32)
    t.add_argument("--deep-trees", type=int, default=4)
    t.add_argument("--deep-depth", type=int, default=8)
    t.add_argument("--img-tokens", type=int, default=49, help="sidecar vectors per image (a square)")
    t.add_argument("--batch-size", type=int, default=256)
    t.add_argument("--lr", type=float, default=1e-3)
    t.add_argument("--warmup", type=int, default=2000)
    t.add_argument("--weight-decay", type=float, default=0.05)
    t.add_argument("--crop-min", type=float, default=0.4)
    t.add_argument("--cap-mask-prob", type=float, default=0.4)
    t.add_argument("--cap-weight", type=float, default=1.0)
    t.add_argument("--held-out", type=int, default=2048, help="a multiple of --batch-size; recall is among 1,000")
    t.add_argument("--eval-every", type=int, default=2000)
    t.add_argument("--log-every", type=int, default=100)
    t.add_argument("--ckpt-minutes", type=float, default=30)
    t.add_argument("--rate-from", type=int, default=100)
    t.add_argument("--rate-steps", type=int, default=100)
    t.add_argument("--no-amp", dest="amp", action="store_false")
    t.add_argument("--no-compile", dest="compile", action="store_false")
    t.add_argument("--seed", type=int, default=0)
    r = sub.add_parser("probe")
    r.add_argument("--ckpt", required=True, help="backbone.pt or checkpoint.pt")
    r.add_argument("--data", required=True, help="the prep-imagenet directory")
    r.add_argument("--out", required=True)
    r.add_argument("--probe-epochs", type=int, default=30)
    r.add_argument("--probe-wd", type=float, nargs="+", default=[1e-4, 1e-2, 1e-1])
    a = p.parse_args()
    {"prep": prep, "prep-imagenet": prep_imagenet, "train": train, "probe": probe}[a.cmd](a)


if __name__ == "__main__":
    main()

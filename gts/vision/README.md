# GTS vision: a ternary GTS vision backbone and a sidecar into GTS-MLM

A side quest: can the GTS mixer (hard-routed trees plus a bidirectional Mamba-2-style scan, ternary weights, 8-bit
activations) work as a vision backbone at about MobileNetV4's size (~38M), and can a small sidecar give the GTS masked
LM (`checkpoints/bert110m/`) vision? Only GTS models are trained here; there are no non-GTS baselines.

## Where it stands (7 October 2026)

- **Run 1 is done** (`runs/run1/`): 38,095 steps, 15.8 epochs of 617,952 pairs, 678 minutes on a secure A40
  (pod `f4umyld24ayokv`, CA-MTL-1, $0.49/hr, about $6.40, now terminated) at 241 images/s, against the **phase 2**
  GTS-MLM (`checkpoints/bert110m/phase2/binarized.pt`, frozen). 2,048 pairs are held out for evaluation.
- **GTS can see.** On the held-out pairs, retrieval at 1 among 1,000 is 55% (chance: 0.1%), and the frozen LM's
  caption loss is 0.36 nats lower with the right image than with a shuffled one, a gap that grew all run.
- It was still improving at the end, but slowly, and the training contrastive loss (0.65) had pulled well below
  what held-out retrieval implies: at 16 epochs the backbone is memorising this data. More (and permissive) pairs
  should help a next run more than more steps.
- **Resume from** `/workspace/vl_run/checkpoint.pt` on RunPod network volume `aiazdht0py` (CA-MTL-1, with the
  prepared data in `/workspace/vl` and the caption-embedding cache): attach it to a pod in CA-MTL-1 and set
  `VL_RESUME=/workspace/vl_run/checkpoint.pt VL_OUT=/workspace/vl_run2`. The float checkpoint (476 MB, with optimizer
  state) is only there; delete the volume when no resume is planned.

| Run 1 | |
|---|---|
| Images/s, epochs | 241 (A40, bf16, compiled), 15.8 epochs, 38,095 steps of 256 |
| Held-out image-to-text / text-to-image R@1 among 1,000 | **55.3% / 57.1%** (step 26,000: 48.5% / 51.7%; 10,000: 35.2% / 38.1%) |
| Held-out caption loss: right image / shuffled image | 3.206 / 3.566 |
| Backbone size | 38.0M parameters; 15.8 MB binarized (2-bit ternary codes, float norms and biases) |

Files in `runs/run1/`: `vl_binarized.pt` (the backbone, `mamba_ssm.utils.ternary_pack` format, config inside),
`vl_backbone.pt.part*` (float backbone plus the sidecar into the LM and their configs: `cat vl_backbone.pt.part* >
vl_backbone.pt`, then `sha256sum -c SHA256SUMS`), `vl_result.json` (settings and the whole curve), `vl.log`.

## Files

| | |
|---|---|
| `mamba_ssm/models/gts_vision.py` | `GTSVision` (the backbone), `VisionSidecar`, `lm_with_image`; tests in `tests/modules/test_gts_vision.py` |
| `vision/vl_pretrain.py` | `prep` (data), `train`, `probe` and `prep-imagenet` (an ImageNet-1k linear probe: built and smoke-tested, **not used**, see licensing) |
| `vision/job.sh` | the pod job: data prep, then training until `END_UTC` |
| `pod/vl_job.sh` | one line, runs `vision/job.sh` (the running pod's start command calls it) |

## The model

- **Backbone, 37.97M parameters**: a float conv stem (3 -> 64 -> 128 channels, stride 4, then a 4x4 patch conv to
  width 512), so a 224 image becomes a 14x14 grid of tokens; a learned position embedding; 17 bidirectional mixed-forest
  GTS blocks, the same blocks as GTS-MLM (bank of 32 depth-0 trees with 8 heads and state 16, plus 4 stateless trees
  of depth 8 with route_ste), ternary weights in groups of 128 and 8-bit activations; a final RMSNorm. A GTS block
  mixes along a sequence, so even layers read the grid in raster order and odd layers in column order: any two layers
  reach every token in both directions. Other resolutions interpolate the position grid. `features()` is the mean of
  the final tokens.
- **Sidecar** (0.99M): the token grid average-pooled to 7x7, then LayerNorm and a two-layer MLP into GTS-MLM's
  embedding width (768). The 49 vectors go right after `[CLS]`, where word embeddings would go, and the frozen LM
  reads them like words (`lm_with_image`).
- **Training**, both losses from every batch of 256:
  - contrastive (CLIP-style, symmetric InfoNCE, learned temperature): mean-pooled image tokens against the frozen
    LM's mean-pooled caption states (computed once at the start), each through a linear map to 512;
  - masked caption modelling: `[CLS]` + the 49 sidecar vectors + the caption with 40% of its tokens masked (80/10/10);
    the frozen LM's own head predicts them, and the gradient reaches the sidecar and the backbone through the frozen
    LM. This is what trains the glue.
  - AdamW (lr 1e-3, betas 0.9/0.98, weight decay 0.05), 2,000 warmup steps, cosine to 10%, fitted to the time left;
    bf16, `torch.compile` per block. Augmentation on the GPU: nvJPEG decoding, random resized crops (area 0.4 to 1)
    to 224; no flips (captions say left and right).
- **Evaluation** every 2,000 steps on the held-out pairs: image-to-text and text-to-image recall at 1 among 1,000, and
  the caption loss with the right image against a shuffled one (the gap is how much the LM uses what it sees).

## Data

`vision/vl_pretrain.py prep` resizes every image (shorter side 256, centre 256x256 crop, JPEG quality 90) and
tokenises captions with the BERT uncased WordPiece GTS-MLM uses, cut at 128 tokens.

| Source | Pairs | Notes | License |
|---|---|---|---|
| ShareGPT4V-PT (`Lin-Chen/ShareGPT4V`, share-captioner 1246k) | 500,000: all 118,287 COCO train2017, 381,713 of LCS-558K | SAM images skipped (terabytes). COCO from images.cocodataset.org, LCS from `liuhaotian/LLaVA-Pretrain` | captions **CC BY-NC 4.0**; COCO images Flickr licenses; LCS "other" (LAION/CC/SBU) |
| VideoGameBunny (`VideoGameBunny/Dataset`) | 120,000 images, one caption each: 52,954 long, 12,715 short, 54,331 the description inside an image-to-JSON answer | QA pairs left out; 178,582 images had a caption | MIT, but frames of commercial games from YouTube, captions by GPT-4V and Gemini |
| TVQA+ | none | official frames gated (`pengxiang/tvqa`, ~137 GB); skipped | |
| ImageNet-1k (probe only) | none | skipped: its terms are non-commercial | |

**Licensing**: with ShareGPT4V's captions under CC BY-NC, run 1's weights are research-only. A commercial version needs
permissive data, e.g. `Spawning/PD12M` (public-domain images, CDLA-Permissive-2.0 metadata, synthetic captions) or
`madebyollin/megalith-10m` (CC0 Flickr photos; needs captions). Check each set's terms before use. `prep` would need a
loader for the new source; `train` takes any prep directories (`--data dir1 dir2 ...`).

## Lessons from getting run 1 going

- Two community-cloud RTX 4090 hosts in a row failed CUDA initialisation ("CUDA unknown error"); `vision/job.sh` now
  stops at once if the GPU does not initialise. The secure A40 worked.
- The pod's memory limit (55 GB here) counts the kernel's page cache. Reading and writing tens of GB of images filled
  it and prep was OOM-killed twice although the processes used 2 GB. `prep` now drops each file's cached pages after
  reading (`posix_fadvise`), flushes and drops its output every 20,000 images, parses the caption JSON in a child
  process, unzips archives to disk with `unzip`/`7z` and recycles its workers; it logs the container's memory on
  every progress line.
- One pod restart hung at `git clone`; a second restart fixed it.

## The faster path (for the next resume)

Run 1 trained at 241 images/s on an A40, about 60% of the GTS-MLM path's efficiency by a rough estimate. Changes for the
next run, all on by default, each with a flag to turn it off; none changes what the model computes:

- **The frozen LM's ternary weights are quantised once** (`GTS.freeze_quantized`, cast to bf16 under autocast), with
  the route kernels' padded copies, instead of re-quantising and re-casting all 89M of them in every forward pass
  (`--no-freeze-lm-quant`). Tested: same outputs.
- **Column-order blocks transpose inside their compiled function** (`GTSVision.compiled_blocks`): the switch to column
  order and back is fused by Inductor into the block's first and last pointwise kernels instead of two gathers of the
  whole token grid per odd block. Tested: same outputs as the gather path.
- **Decoding and cropping run ahead on a side CUDA stream** in the prefetch thread (`--no-async-decode`), so nvJPEG and
  the crops overlap the previous step instead of sitting on its critical path.
- **Blocks compile with static shapes** (`dynamic=False`; a second shape had recompiled them with dynamic shapes and hit
  an Inductor bug), evaluation runs uncompiled, and the job retries uncompiled if Inductor still fails.
- **`--resume`** continues a run: backbone, heads and sidecar, optimizer state, step and curve; the learning rate
  re-warms from the checkpoint's last rate to `--lr` over `--rewarm` steps, then a cosine to 10%. With the pod job:
  `VL_RESUME=/workspace/vl_run/checkpoint.pt VL_OUT=/workspace/vl_run2 END_UTC=...` (data and caption embeddings are
  reused from the volume). The speed-up is not measured yet: compare the new run's "images/s" line with 241.

## Picking it up

1. Run 1 answered "can GTS see?" with yes (above). Its outputs are in `runs/run1/`, its resumable checkpoint on the
   volume.
2. Next, in rough order: align the sidecar to the final GTS-MLM (phase 3) with the backbone frozen; a commercial
   rerun on permissive data; a probe on a permissively licensed labelled set in place of ImageNet; a C/CPU inference
   path for the backbone like `kernel/enc_bench.c`.

Run the prep and training yourself:

```bash
python vision/vl_pretrain.py prep --out /root/vl --work /root/raw            # ~40 min on 8 cores, ~100 GB of downloads
cat checkpoints/bert110m/phase2/binarized.pt.part* > /root/lm.pt                # the LM, from its 90 MB parts
python vision/vl_pretrain.py train --data /root/vl/sharegpt4v /root/vl/vgb --lm /root/lm.pt --out /root/run --minutes 600
python -m pytest tests/modules/test_gts_vision.py                            # CPU, seconds
```

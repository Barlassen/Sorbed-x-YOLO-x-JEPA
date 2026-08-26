# Training & Serving a Learned Segmentation Backend

This guide covers training a wound-boundary segmenter, exporting it to
ONNX, registering it with integrity verification, and serving it through Sorbed's
`onnx` backend. See `docs/MODELS.md` first — it documents the
**data-poverty reality** of this field, which shapes the expectations below.

## Starting point

There is **no verifiable public, permissively-licensed pressure-injury *staging*
dataset**, and tissue-labeled data is limited. Public data does exist for **binary
wound-boundary segmentation**, where models routinely reach
Dice in the 0.85–0.93 range. This pipeline targets that task —
segmenting *where* the wound is — and Sorbed does not classify *what*
each pixel is or *how* to stage it. A boundary model trained on
foot ulcers is not a validated pressure-injury classifier.

## 1. Prepare data

Two public binary wound-segmentation sets, both distributed from
[github.com/uwm-bigdata/wound-segmentation](https://github.com/uwm-bigdata/wound-segmentation):

- **AZH Chronic Wound** — ~1,109 images with binary masks.
- **FUSeg 2021** — ~1,210 foot-ulcer images with masks.

Arrange them as paired directories (files matched by stem):

```
dataset/
  images/   0001.png 0002.png ...     # RGB photos
  masks/    0001.png 0002.png ...     # binary masks, wound = nonzero (0/255 or 0/1)
```

Check each dataset's license individually before any deployment. DFUC is
**non-commercial**; Medetec carries **no formal license**. A permissive code
stack does not relax a data license.

## 2. Install the ML stack

Training requires the heavy optional dependencies (kept out of the core install and
never imported at module load):

```bash
pip install torch torchvision segmentation-models-pytorch
# plus the core deps already used by Sorbed: numpy, opencv-python
```

## 3. Train

```bash
python scripts/train_segmenter.py \
    --images dataset/images \
    --masks  dataset/masks \
    --out-dir artifacts/segmenter \
    --epochs 40 \
    --batch-size 8 \
    --lr 1e-3 \
    --input-size 512 \
    --encoder efficientnet-b0 \
    --decoder-attention scse
```

The script builds the chosen architecture (default: a U-Net with an
ImageNet-pretrained EfficientNet-b0 encoder), trains with a combined **Dice +
BCE** loss, holds out a validation split, reports **per-epoch validation Dice**,
and checkpoints the best model to `artifacts/segmenter/best.pt`. Use `--device
cpu` to force CPU, or `cuda` for a GPU; `auto` (default) selects CUDA when
available.

### Architecture (`--arch`)

Routed through `segmentation-models-pytorch`:

| `--arch` | Recipe | Notes |
|---|---|---|
| `unet` (default) | U-Net + EfficientNet encoder + scSE | the FUSegNet-line CNN recipe |
| `deeplabv3plus`, `manet` | CNN alternatives | with EfficientNet/ResNet encoders |
| `segformer` | **SegFormer transformer** | pair with a MiT encoder: `--encoder mit_b2` (CPU-reasonable, ~25M) or `mit_b3` |

All architectures export to ONNX for CPU inference. `--decoder-attention scse`
applies only to `unet`/`unetplusplus`; it is ignored by the others. The
transformer recipe is:

```bash
python scripts/train_segmenter.py --images imgs/ --masks masks/ \
    --arch segformer --encoder mit_b2 --encoder-weights imagenet --input-size 512
```

The CNN baseline (nnU-Net) and state-space (Mamba) / promptable
foundation models (SAM-2, MedSAM-2, BiomedParse) are **not** integrated here;
they are GPU/prompt-oriented and outside `smp`. See
[`MODELS.md`](MODELS.md) for the cited landscape.

### scSE decoder attention (`--decoder-attention`)

`scse` (the default) inserts **spatial-and-channel Squeeze-and-Excitation**
blocks into every decoder stage. scSE recalibrates decoder features by *what*
(a channel-gating branch) and *where* (a spatial-gating branch) jointly, and is
the attention mechanism the **FUSegNet** line uses to reach state of the art on
the AZH/FUSeg chronic-wound benchmark (data-based DSC 92.70% with an
EfficientNet-b7 encoder + parallel-scSE; Dhar et al., *Biomed. Signal Process.
Control*, 2024). Pass `--decoder-attention none` for a plain U-Net baseline. To
approach the published numbers, use a larger pretrained encoder and full-size
inputs:

```bash
python scripts/train_segmenter.py --images imgs/ --masks masks/ \
    --encoder efficientnet-b4 --encoder-weights imagenet \
    --input-size 512 --decoder-attention scse --epochs 80
```

Encoder weights (`imagenet`) download from the PyTorch Hub; where that host is
blocked, train from scratch with `--encoder-weights none` (lower accuracy) or
supply local weights.

## 4. Export to ONNX

At the end of training the script restores the best checkpoint and exports
`artifacts/segmenter/model.onnx` with a fixed `(1, 3, H, W)` input and a
`(1, 1, H, W)` foreground-logit output — the layout the `OnnxSegmenter`
reads. It prints the file's **SHA-256**.

The file can be re-hashed at any time:

```bash
sha256sum artifacts/segmenter/model.onnx
```

## 5. Register the weights (integrity verification)

Host `model.onnx` at a fetchable location and add an entry to `models/registry.json`
(create the file if it does not exist):

```json
{
  "models": [
    {
      "name": "unet-fuseg-azh",
      "url": "https://example.org/weights/unet-fuseg-azh.onnx",
      "sha256": "<the digest the training script printed>",
      "license": "MIT",
      "backend": "onnx",
      "input_size": [512, 512],
      "description": "U-Net (EfficientNet-b0) trained on FUSeg + AZH boundary masks."
    }
  ]
}
```

`download_weights()` streams the file, verifies the SHA-256, and **fails closed**
(deletes the file and raises) on any mismatch. Only `.onnx` / `.safetensors` are
accepted — Sorbed does not load pickle-based formats. Then copy
`model_cards/TEMPLATE.md` to `model_cards/unet-fuseg-azh.md` and complete it,
including the **required skin-tone evaluation**.

## 6. Run with the ONNX backend

Point Sorbed at the model and select the backend:

```bash
export SORBED_SEGMENTATION_BACKEND=onnx
export SORBED_ONNX_MODEL=artifacts/segmenter/model.onnx
```

The backend loads the model through ONNX Runtime (CUDA when available, otherwise
CPU), resizes each image to the model's expected input, runs inference, applies a
sigmoid (or softmax for two-class outputs), thresholds at 0.5, keeps the largest
connected component, and reports a **confidence value** — the mean predicted wound
probability inside the mask. The model's SHA-256 is recorded on every result for
provenance.

Optional normalization overrides (defaults are ImageNet statistics, matching the
training script):

```bash
export SORBED_ONNX_MEAN="0.485,0.456,0.406"
export SORBED_ONNX_STD="0.229,0.224,0.225"
export SORBED_ONNX_INPUT="512"   # fallback H=W only if the model has dynamic axes
```

If `SORBED_ONNX_MODEL` is unset or missing, the backend raises an error
directing the caller to set it or run `sorbed models pull`; it does not silently fall back
or fabricate a mask.

---

## Reproduce the FUSeg-trained model (end to end)

This trains a wound-segmentation model on clinical data and runs it in
the pipeline — no HuggingFace, no synthetic data.

```bash
pip install -e '.[dev,ml]' torch segmentation-models-pytorch

# 1. Fetch wound images + expert masks (MICCAI FUSeg, committed on GitHub).
python scripts/fetch_fuseg.py --out data/fuseg --split train --limit 300
python scripts/fetch_fuseg.py --out data/fuseg --split validation --limit 60

# 2. Train a U-Net and export ONNX. Use --encoder-weights imagenet where the
#    weight host is reachable (higher Dice); 'none' trains from scratch offline.
python scripts/train_segmenter.py \
  --images data/fuseg/train/images --masks data/fuseg/train/labels \
  --encoder mobilenet_v2 --encoder-weights none \
  --input-size 224 --epochs 20 --batch-size 8 --out-dir artifacts/segmenter

# 3. Run the trained model through Sorbed on held-out images.
export SORBED_SEGMENTATION_BACKEND=onnx
export SORBED_ONNX_MODEL=artifacts/segmenter/model.onnx
sorbed analyze data/fuseg/validation/images/0002.png --out reports
```

**Reference result (this exact recipe, CPU, from scratch):** best validation
Dice ≈ **0.64** after 20 epochs at 224 px with a randomly-initialized
MobileNetV2 encoder. This model misses small or subtle ulcers. Using an
ImageNet-pretrained encoder, 512 px input, and more epochs (as in the literature)
reaches ≈0.85+ Dice where the encoder-weight host is reachable.

**Notes:**

- The ONNX backend reads the model's fixed input size from the graph; for a
  dynamic-axis model set `SORBED_ONNX_INPUT`. Normalization defaults to ImageNet
  (`SORBED_ONNX_MEAN` / `SORBED_ONNX_STD` to override) and must match training.
- Sorbed runs the **learned segmenter on the raw image** (matching its training
  distribution); color normalization is applied only to the tissue-color
  analysis. Train the model on un-color-normalized images accordingly.
- Register the exported `model.onnx` SHA-256 in `models/registry.json` (see the
  registry) so results are tied to exact weights. Do not commit weights or the
  dataset — both are `.gitignore`d.

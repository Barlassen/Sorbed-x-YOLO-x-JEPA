#!/usr/bin/env python3
"""Fine-tune SAM / MedSAM for *automatic* (prompt-free) wound segmentation.

The stock Segment-Anything family (SAM, MedSAM, SAM-2) is **promptable**: it needs
a box or a point to emit a mask, so it cannot run unattended. This script closes
that gap without retraining the heavy image encoder. It does two cheap things on
top of a frozen SAM backbone:

1. **Auto-prompt head** (``--auto-prompt learned``) — a small CNN on the frozen
   image embedding regresses a single wound bounding box. At inference that box is
   fed to SAM's prompt encoder, so the whole thing runs with no human in the loop.
2. **Box-from-heuristic** (``--auto-prompt heuristic``) — no learned head; at
   inference the box is the bounding box of the weight-free
   :class:`sorbed.segmentation.classical.ClassicalSegmenter` mask. Nothing to
   train but the decoder.

In both modes the **image encoder is frozen** and only the **mask decoder** (and,
in learned mode, the auto-prompt head) are trained. The decoder is supervised with
GT-jittered box prompts (the standard MedSAM fine-tune recipe), which teaches it to
be robust to the imperfect boxes the auto-prompt produces at test time.

Weights source (HuggingFace)
----------------------------
This uses the ``transformers`` ``SamModel``/``SamProcessor`` API, so any SAM-format
checkpoint on the Hub works. Verified, loadable ids:

* ``facebook/sam-vit-base`` (default), ``facebook/sam-vit-large``,
  ``facebook/sam-vit-huge`` — the original Meta SAM weights.
* A MedSAM checkpoint exported to the ``transformers`` SAM format (medical-domain
  fine-tune of SAM-ViT-B). Pass its Hub id via ``--model-id``; confirm the exact
  repo name on the Hub before relying on it, since community mirrors move.

SAM-2 / MedSAM-2 (``wanglab/MedSAM2``) use a *different* code path (the
``facebookresearch/sam2`` package, not ``transformers``) and are optimized for 3D
/ video; for single 2D wound photos the ``transformers`` SAM/MedSAM path here is
the pragmatic, ONNX-friendlier choice. See ``training/EVALUATE.md`` for the SAM-2
note.

No-API download / offline fallback
----------------------------------
``training/fetch_models.py`` fetches transformers-format SAM/MedSAM weights with
**no HuggingFace token and no API key** (public ``resolve/main`` URLs), so a plain
server can grab a real MedSAM checkpoint and run fully offline::

    python -m training.fetch_models medsam-vit-base --out /data/briefer/models
    python -m training.finetune_medsam train \\
        --weights-dir /data/briefer/models/medsam-vit-base ...

``--weights-dir`` (or ``HF_HUB_OFFLINE=1``) forces ``local_files_only`` so the run
never touches the network — the required posture on an air-gapped clinical box.
``fetch_models.py medsam-vit-base`` uses the public ``flaviagiammarino/medsam-vit-base``
mirror; ``huggingface-cli download`` on any public repo also works without a token.

Data
----
Consumes a manifest written by ``training/data_prep.py`` (columns per
``training/datasets.py``): each row needs an ``image_path`` and a binary
``mask_path`` (wound = nonzero). Foot-ulcer masks (AZH/FUSeg) are the recommended
open starting point; see ``training/DATA_README.md``.

Evaluation
----------
``finetune_medsam.py eval`` reports the fine-tuned auto-SAM's Dice/IoU **and** the
supervised sorbed segmenter's Dice/IoU on the *same* test manifest, so the two are
directly comparable. The supervised baseline is the weight-free classical
segmenter, or the ONNX U-Net when ``SORBED_ONNX_MODEL`` is set.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

if TYPE_CHECKING:  # pragma: no cover - typing only
    import torch

# Import the manifest layer whether run as a script or as ``training.finetune_medsam``.
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from datasets import IMAGE_SUFFIXES, ManifestRecord, read_manifest
else:  # pragma: no cover - exercised only when imported as a package
    from training.datasets import IMAGE_SUFFIXES, ManifestRecord, read_manifest

# SAM works in a fixed 1024-px padded canvas; its low-res mask logits are 256 px.
SAM_INPUT = 1024
SAM_LOWRES = 256


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fine-tune SAM/MedSAM for automatic (prompt-free) wound masks.",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--model-id",
        default="facebook/sam-vit-base",
        help="HuggingFace SAM-format checkpoint id (e.g. facebook/sam-vit-base, or a "
        "MedSAM export). Ignored when --weights-dir is given.",
    )
    common.add_argument(
        "--weights-dir",
        type=Path,
        default=None,
        help="Local snapshot directory of a SAM checkpoint; forces offline loading.",
    )
    common.add_argument(
        "--auto-prompt",
        choices=["learned", "heuristic"],
        default="learned",
        help="'learned': train a box-regression head on frozen embeddings. "
        "'heuristic': derive the box from the classical segmenter at inference.",
    )
    common.add_argument(
        "--paths-relative-to",
        type=Path,
        default=None,
        help="Root to resolve manifest image/mask paths against (default: the "
        "manifest's own directory).",
    )
    common.add_argument("--device", default="auto", help="'auto' | 'cpu' | 'cuda' | 'cuda:N'")
    common.add_argument("--num-workers", type=int, default=4)
    common.add_argument("--seed", type=int, default=1234)

    train_p = sub.add_parser("train", parents=[common], help="Fine-tune the decoder / head.")
    train_p.add_argument("--train-manifest", type=Path, required=True)
    train_p.add_argument(
        "--val-manifest",
        type=Path,
        default=None,
        help="Optional held-out manifest for best-checkpoint selection by val Dice.",
    )
    train_p.add_argument("--out-dir", type=Path, default=Path("artifacts/medsam_auto"))
    train_p.add_argument("--epochs", type=int, default=30)
    train_p.add_argument("--batch-size", type=int, default=4)
    train_p.add_argument("--lr", type=float, default=1e-4)
    train_p.add_argument("--weight-decay", type=float, default=1e-4)
    train_p.add_argument("--warmup-frac", type=float, default=0.05)
    train_p.add_argument(
        "--box-loss-weight",
        type=float,
        default=1.0,
        help="Weight on the auto-prompt box-regression loss (learned mode only).",
    )
    train_p.add_argument(
        "--box-jitter",
        type=float,
        default=0.1,
        help="Fractional jitter applied to the GT box that prompts the decoder in "
        "training, so the decoder tolerates the auto-prompt's error at test time.",
    )
    train_p.add_argument(
        "--no-amp",
        action="store_true",
        help="Disable mixed precision (AMP is on by default on CUDA).",
    )
    train_p.add_argument(
        "--max-eval-samples",
        type=int,
        default=200,
        help="Cap on val images scored each epoch (keeps per-epoch eval cheap).",
    )

    eval_p = sub.add_parser(
        "eval", parents=[common], help="Score auto-SAM vs the supervised model."
    )
    eval_p.add_argument("--test-manifest", type=Path, required=True)
    eval_p.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="best.pt from a train run. Omit to evaluate the un-tuned SAM decoder.",
    )
    eval_p.add_argument("--out", type=Path, default=None, help="Write the metrics JSON here.")
    eval_p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only score the first N test images (smoke test).",
    )
    eval_p.add_argument(
        "--skip-supervised",
        action="store_true",
        help="Skip the sorbed supervised-segmenter comparison column.",
    )

    return parser.parse_args(argv)


# --------------------------------------------------------------------------- #
# Geometry / metric helpers (numpy only)
# --------------------------------------------------------------------------- #
def _resolve_path(rel: str, base: Path) -> Path:
    path = Path(rel)
    return path if path.is_absolute() else base / path


def _mask_bbox(mask: np.ndarray) -> tuple[int, int, int, int] | None:
    """Tight ``(x0, y0, x1, y1)`` bounding box of the nonzero region, or None."""
    ys, xs = np.nonzero(mask)
    if xs.size == 0:
        return None
    return int(xs.min()), int(ys.min()), int(xs.max()) + 1, int(ys.max()) + 1


def _corners_to_cxcywh(box: tuple[float, float, float, float]) -> tuple[float, float, float, float]:
    x0, y0, x1, y1 = box
    return (0.5 * (x0 + x1), 0.5 * (y0 + y1), x1 - x0, y1 - y0)


def _cxcywh_to_corners(
    cx: float, cy: float, w: float, h: float
) -> tuple[float, float, float, float]:
    return (cx - 0.5 * w, cy - 0.5 * h, cx + 0.5 * w, cy + 0.5 * h)


def dice_iou(pred: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    """Dice and IoU between two boolean masks (empty-vs-empty scores as 1.0)."""
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    inter = float(np.logical_and(pred_b, gt_b).sum())
    p = float(pred_b.sum())
    g = float(gt_b.sum())
    if p + g == 0.0:
        return 1.0, 1.0
    dice = 2.0 * inter / (p + g)
    union = p + g - inter
    iou = inter / union if union > 0 else 1.0
    return dice, iou


# --------------------------------------------------------------------------- #
# Model construction (lazy torch / transformers)
# --------------------------------------------------------------------------- #
def _resolve_device(spec: str) -> torch.device:
    import torch

    if spec == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(spec)


def _load_sam(model_id: str, weights_dir: Path | None) -> tuple[Any, Any]:
    """Load a ``transformers`` ``SamModel`` + ``SamProcessor``.

    ``weights_dir`` (a local snapshot) forces ``local_files_only`` so nothing is
    fetched from the network — the offline / air-gapped path.
    """
    from transformers import SamModel, SamProcessor

    source = str(weights_dir) if weights_dir is not None else model_id
    local_only = weights_dir is not None
    model = SamModel.from_pretrained(source, local_files_only=local_only)
    processor = SamProcessor.from_pretrained(source, local_files_only=local_only)
    return model, processor


def build_auto_prompt_head(embed_dim: int = 256, hidden: int = 128) -> torch.nn.Module:
    """A small CNN that regresses one normalized ``(cx, cy, w, h)`` box.

    Input is a frozen SAM image embedding ``(B, embed_dim, 64, 64)``; output is
    ``(B, 1, 4)`` in ``[0, 1]`` (fractions of the 1024-px SAM canvas). Predicting
    centre/size and clamping guarantees a well-ordered box.
    """
    import torch
    from torch import nn

    class AutoPromptHead(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.body = nn.Sequential(
                nn.Conv2d(embed_dim, hidden, 3, padding=1),
                nn.GroupNorm(16, hidden),
                nn.ReLU(inplace=True),
                nn.Conv2d(hidden, hidden, 3, stride=2, padding=1),
                nn.GroupNorm(16, hidden),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
            )
            self.fc = nn.Linear(hidden, 4)

        def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
            feats = self.body(embeddings).flatten(1)
            raw = torch.sigmoid(self.fc(feats))  # (B, 4) in [0, 1]
            return raw.unsqueeze(1)  # (B, 1, 4) as (cx, cy, w, h)

    return AutoPromptHead()


def _cxcywh_norm_to_box1024(pred: torch.Tensor) -> torch.Tensor:
    """Convert normalized ``(B, 1, 4)`` cxcywh to SAM 1024-frame corner boxes."""
    import torch

    cx, cy, w, h = pred[..., 0], pred[..., 1], pred[..., 2], pred[..., 3]
    x0 = (cx - 0.5 * w).clamp(0.0, 1.0) * SAM_INPUT
    y0 = (cy - 0.5 * h).clamp(0.0, 1.0) * SAM_INPUT
    x1 = (cx + 0.5 * w).clamp(0.0, 1.0) * SAM_INPUT
    y1 = (cy + 0.5 * h).clamp(0.0, 1.0) * SAM_INPUT
    return torch.stack([x0, y0, x1, y1], dim=-1)  # (B, 1, 4)


def _freeze_backbone(model: Any) -> None:
    """Freeze the image encoder and prompt encoder; leave the mask decoder trainable."""
    for param in model.vision_encoder.parameters():
        param.requires_grad_(False)
    for param in model.prompt_encoder.parameters():
        param.requires_grad_(False)
    for param in model.mask_decoder.parameters():
        param.requires_grad_(True)


# --------------------------------------------------------------------------- #
# Image -> SAM tensors
# --------------------------------------------------------------------------- #
def _read_rgb(path: Path) -> np.ndarray:
    import cv2

    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"could not read image {path}")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


def _read_mask(path: Path) -> np.ndarray:
    import cv2

    raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise RuntimeError(f"could not read mask {path}")
    return (raw > 0).astype(np.uint8)


def _gt_lowres(mask: np.ndarray, reshaped_hw: tuple[int, int]) -> np.ndarray:
    """Place ``mask`` in the padded 1024 canvas SAM uses, then down-sample to 256.

    SAM's low-res logits live in the *padded* frame (256 = 1024 / 4), so the GT
    target must be built the same way — resize to the aspect-preserving reshaped
    size, pad top-left to 1024, then shrink to 256 — not a naive stretch.
    """
    import cv2

    rh, rw = int(reshaped_hw[0]), int(reshaped_hw[1])
    resized = cv2.resize(mask, (rw, rh), interpolation=cv2.INTER_NEAREST)
    canvas = np.zeros((SAM_INPUT, SAM_INPUT), dtype=np.uint8)
    canvas[:rh, :rw] = resized
    low = cv2.resize(canvas, (SAM_LOWRES, SAM_LOWRES), interpolation=cv2.INTER_NEAREST)
    return (low > 0).astype(np.float32)


def _process_rgb(processor: Any, rgb: np.ndarray) -> dict[str, Any]:
    """Run the SAM processor on one RGB image; return its tensors + frame sizes."""
    inputs = processor(images=rgb, return_tensors="pt")
    return {
        "pixel_values": inputs["pixel_values"],  # (1, 3, 1024, 1024)
        "original_sizes": inputs["original_sizes"],
        "reshaped_input_sizes": inputs["reshaped_input_sizes"],
    }


# --------------------------------------------------------------------------- #
# Datasets (lazy torch)
# --------------------------------------------------------------------------- #
def _labeled_records(manifest: Path, base: Path) -> list[ManifestRecord]:
    records = [r for r in read_manifest(manifest) if r.mask_path]
    if not records:
        raise SystemExit(f"manifest {manifest} has no rows with a mask_path")
    # Drop rows whose files are missing so a bad path fails fast, not mid-epoch.
    kept: list[ManifestRecord] = []
    for record in records:
        img = _resolve_path(record.image_path, base)
        msk = _resolve_path(record.mask_path, base)
        if img.is_file() and msk.is_file() and img.suffix.lower() in IMAGE_SUFFIXES:
            kept.append(record)
    if not kept:
        raise SystemExit(f"none of {manifest}'s image/mask files exist under {base}")
    return kept


def _build_dataset(records: list[ManifestRecord], base: Path, processor: Any) -> Any:
    import torch
    from torch.utils.data import Dataset

    class SamBoxDataset(Dataset):  # type: ignore[misc]
        """Yields ``(pixel_values, gt_lowres, gt_box_cxcywh)`` for decoder training."""

        def __len__(self) -> int:
            return len(records)

        def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
            record = records[index]
            rgb = _read_rgb(_resolve_path(record.image_path, base))
            mask = _read_mask(_resolve_path(record.mask_path, base))
            proc = _process_rgb(processor, rgb)
            reshaped = proc["reshaped_input_sizes"][0].tolist()
            orig_h, orig_w = rgb.shape[0], rgb.shape[1]
            scale_h = reshaped[0] / orig_h
            scale_w = reshaped[1] / orig_w

            gt_low = _gt_lowres(mask, (reshaped[0], reshaped[1]))
            box = _mask_bbox(mask)
            if box is None:
                # No wound pixels: degenerate full-canvas box, zero target mask.
                cxcywh = (SAM_INPUT / 2, SAM_INPUT / 2, float(SAM_INPUT), float(SAM_INPUT))
            else:
                x0, y0, x1, y1 = box
                scaled = (x0 * scale_w, y0 * scale_h, x1 * scale_w, y1 * scale_h)
                cxcywh = _corners_to_cxcywh(scaled)
            cx, cy, w, h = (v / SAM_INPUT for v in cxcywh)
            return {
                "pixel_values": proc["pixel_values"][0],
                "gt_lowres": torch.from_numpy(gt_low),
                "gt_box": torch.tensor([cx, cy, w, h], dtype=torch.float32),
            }

    return SamBoxDataset()


# --------------------------------------------------------------------------- #
# Decoder forward + losses
# --------------------------------------------------------------------------- #
def _jitter_boxes(gt_box_cxcywh: torch.Tensor, amount: float) -> torch.Tensor:
    """Perturb normalized cxcywh boxes and return 1024-frame corner boxes."""
    import torch

    if amount <= 0.0:
        return _cxcywh_norm_to_box1024(gt_box_cxcywh.unsqueeze(1))
    noise = (torch.rand_like(gt_box_cxcywh) - 0.5) * 2.0 * amount
    jittered = (gt_box_cxcywh * (1.0 + noise)).clamp(0.0, 1.0)
    return _cxcywh_norm_to_box1024(jittered.unsqueeze(1))


def _decoder_masks(model: Any, embeddings: torch.Tensor, boxes1024: torch.Tensor) -> torch.Tensor:
    """Run prompt encoder + mask decoder; return ``(B, 256, 256)`` mask logits."""
    outputs = model(
        image_embeddings=embeddings,
        input_boxes=boxes1024,
        multimask_output=False,
    )
    # pred_masks: (B, point_batch=1, num_masks=1, 256, 256)
    return outputs.pred_masks[:, 0, 0]


def _dice_bce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    from torch.nn import functional

    bce = functional.binary_cross_entropy_with_logits(logits, target)
    probs = logits.sigmoid()
    inter = (probs * target).sum(dim=(1, 2))
    card = probs.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
    dice = (2.0 * inter + 1.0) / (card + 1.0)
    return bce + (1.0 - dice.mean())


# --------------------------------------------------------------------------- #
# Inference (auto mask for one image)
# --------------------------------------------------------------------------- #
def _classical_bbox_norm(rgb: np.ndarray) -> tuple[float, float, float, float] | None:
    """Bounding box of the classical segmenter's mask, as normalized cxcywh.

    Returns ``None`` when the weight-free segmenter finds nothing wound-like.
    """
    from sorbed.domain.image import Calibration
    from sorbed.imaging import RasterImage, build_metadata
    from sorbed.segmentation.classical import ClassicalSegmenter

    pixels = (rgb.astype(np.float32) / 255.0).astype(np.float32)
    metadata = build_metadata(
        pixels=pixels,
        source_format="array",
        bit_depth=8,
        channels=3,
    )
    image = RasterImage(pixels=pixels, metadata=metadata, calibration=Calibration())
    result = ClassicalSegmenter().segment(image)
    box = _mask_bbox(result.wound_mask)
    if box is None:
        return None
    h, w = rgb.shape[0], rgb.shape[1]
    x0, y0, x1, y1 = box
    cx, cy, bw, bh = _corners_to_cxcywh((x0 / w, y0 / h, x1 / w, y1 / h))
    return cx, cy, bw, bh


def _auto_mask(
    model: Any,
    head: Any | None,
    processor: Any,
    rgb: np.ndarray,
    device: torch.device,
) -> np.ndarray:
    """Produce a boolean wound mask for one image with no human prompt."""
    import torch

    proc = _process_rgb(processor, rgb)
    pixel_values = proc["pixel_values"].to(device)
    with torch.no_grad():
        embeddings = model.get_image_embeddings(pixel_values)
        if head is not None:
            pred = head(embeddings)  # (1, 1, 4) cxcywh in [0, 1]
            boxes1024 = _cxcywh_norm_to_box1024(pred)
        else:
            cxcywh = _classical_bbox_norm(rgb)
            if cxcywh is None:
                return np.zeros((rgb.shape[0], rgb.shape[1]), dtype=bool)
            corners = _cxcywh_to_corners(*cxcywh)
            boxes1024 = torch.tensor(
                [[[c * SAM_INPUT for c in corners]]], dtype=torch.float32, device=device
            )
        outputs = model(
            image_embeddings=embeddings,
            input_boxes=boxes1024,
            multimask_output=False,
        )
        masks = processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            proc["original_sizes"],
            proc["reshaped_input_sizes"],
        )
    mask = masks[0]
    arr = mask.numpy() if hasattr(mask, "numpy") else np.asarray(mask)
    arr = arr.reshape(-1, arr.shape[-2], arr.shape[-1])[0]
    return arr.astype(bool)


# --------------------------------------------------------------------------- #
# Scoring
# --------------------------------------------------------------------------- #
def _score_auto_sam(
    model: Any,
    head: Any | None,
    processor: Any,
    records: list[ManifestRecord],
    base: Path,
    device: torch.device,
    limit: int | None = None,
) -> dict[str, float]:
    dices: list[float] = []
    ious: list[float] = []
    subset = records[:limit] if limit is not None else records
    for record in subset:
        rgb = _read_rgb(_resolve_path(record.image_path, base))
        gt = _read_mask(_resolve_path(record.mask_path, base))
        pred = _auto_mask(model, head, processor, rgb, device)
        dice, iou = dice_iou(pred, gt)
        dices.append(dice)
        ious.append(iou)
    return {
        "n": float(len(dices)),
        "dice": float(np.mean(dices)) if dices else 0.0,
        "iou": float(np.mean(ious)) if ious else 0.0,
    }


def _score_supervised(
    records: list[ManifestRecord], base: Path, limit: int | None
) -> dict[str, Any]:
    """Score the sorbed supervised segmenter (ONNX if configured, else classical)."""
    import os

    from sorbed.io import load_image
    from sorbed.segmentation.classical import ClassicalSegmenter

    backend_name = "classical"
    segmenter: Any = ClassicalSegmenter()
    onnx_path = os.environ.get("SORBED_ONNX_MODEL", "").strip()
    if onnx_path:
        try:
            from sorbed.segmentation.backends.onnx_backend import OnnxSegmenter

            segmenter = OnnxSegmenter(Path(onnx_path))
            backend_name = "onnx"
        except Exception as exc:  # fall back honestly, don't crash eval
            print(f"  (ONNX backend unavailable: {exc}; using classical)")

    dices: list[float] = []
    ious: list[float] = []
    subset = records[:limit] if limit is not None else records
    for record in subset:
        image = load_image(_resolve_path(record.image_path, base))
        gt = _read_mask(_resolve_path(record.mask_path, base))
        pred = segmenter.segment(image).wound_mask
        dice, iou = dice_iou(pred, gt)
        dices.append(dice)
        ious.append(iou)
    return {
        "backend": backend_name,
        "n": float(len(dices)),
        "dice": float(np.mean(dices)) if dices else 0.0,
        "iou": float(np.mean(ious)) if ious else 0.0,
    }


# --------------------------------------------------------------------------- #
# Train / eval entry points
# --------------------------------------------------------------------------- #
def _train(args: argparse.Namespace) -> int:
    import torch
    from torch.utils.data import DataLoader

    if __package__ in (None, ""):
        from utils import (  # type: ignore[import-not-found]
            CheckpointManager,
            autocast_context,
            cosine_warmup_scheduler,
            make_grad_scaler,
            set_seed,
        )
    else:  # pragma: no cover
        from training.utils import (
            CheckpointManager,
            autocast_context,
            cosine_warmup_scheduler,
            make_grad_scaler,
            set_seed,
        )

    set_seed(args.seed)
    device = _resolve_device(args.device)
    base = args.paths_relative_to or args.train_manifest.expanduser().parent

    model, processor = _load_sam(args.model_id, args.weights_dir)
    _freeze_backbone(model)
    model.to(device)

    learned = args.auto_prompt == "learned"
    head = build_auto_prompt_head().to(device) if learned else None

    train_records = _labeled_records(args.train_manifest, base)
    train_ds = _build_dataset(train_records, base, processor)
    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )

    val_records: list[ManifestRecord] = []
    if args.val_manifest is not None:
        val_base = args.paths_relative_to or args.val_manifest.expanduser().parent
        val_records = _labeled_records(args.val_manifest, val_base)

    params: list[torch.nn.Parameter] = list(model.mask_decoder.parameters())
    if head is not None:
        params += list(head.parameters())
    optimizer = torch.optim.AdamW(params, lr=args.lr, weight_decay=args.weight_decay)
    total_steps = max(1, args.epochs * len(train_loader))
    scheduler = cosine_warmup_scheduler(
        optimizer,
        total_steps=total_steps,
        warmup_steps=int(args.warmup_frac * total_steps),
    )
    use_amp = not args.no_amp
    scaler = make_grad_scaler(device, enabled=use_amp)

    ckpt = CheckpointManager(args.out_dir, mode="max")
    from torch.nn import functional

    for epoch in range(1, args.epochs + 1):
        model.mask_decoder.train()
        if head is not None:
            head.train()
        running = 0.0
        seen = 0
        for batch in train_loader:
            pixel_values = batch["pixel_values"].to(device)
            gt_low = batch["gt_lowres"].to(device)
            gt_box = batch["gt_box"].to(device)

            with torch.no_grad():
                embeddings = model.get_image_embeddings(pixel_values)

            with autocast_context(device, enabled=use_amp):
                boxes1024 = _jitter_boxes(gt_box, args.box_jitter)
                logits = _decoder_masks(model, embeddings, boxes1024)
                loss = _dice_bce(logits, gt_low)
                if head is not None:
                    pred_box = head(embeddings).squeeze(1)  # (B, 4) cxcywh
                    loss = loss + args.box_loss_weight * functional.smooth_l1_loss(
                        pred_box, gt_box
                    )

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            running += float(loss.item()) * pixel_values.size(0)
            seen += pixel_values.size(0)

        train_loss = running / max(1, seen)

        val_dice = train_loss * -1.0  # fall back to loss-driven selection w/o val set
        if val_records:
            model.mask_decoder.eval()
            if head is not None:
                head.eval()
            scores = _score_auto_sam(
                model, head, processor, val_records, base, device, limit=args.max_eval_samples
            )
            val_dice = scores["dice"]
            print(
                f"epoch {epoch:03d}/{args.epochs}  loss={train_loss:.4f}  "
                f"val_dice={val_dice:.4f}  val_iou={scores['iou']:.4f}"
            )
        else:
            print(f"epoch {epoch:03d}/{args.epochs}  loss={train_loss:.4f}")

        state = {
            "mask_decoder": model.mask_decoder.state_dict(),
            "auto_prompt": head.state_dict() if head is not None else None,
            "auto_prompt_mode": args.auto_prompt,
            "model_id": args.model_id,
            "epoch": epoch,
        }
        ckpt.save(state, metric=val_dice)

    print(f"best metric: {ckpt.best_metric:.4f} (checkpoint: {ckpt.best_path})")
    return 0


def _load_checkpoint_into(
    model: Any, head: Any | None, checkpoint: Path, device: torch.device
) -> None:
    import torch

    state = torch.load(checkpoint, map_location=device)
    model.mask_decoder.load_state_dict(state["mask_decoder"])
    if head is not None and state.get("auto_prompt") is not None:
        head.load_state_dict(state["auto_prompt"])


def _evaluate(args: argparse.Namespace) -> int:
    device = _resolve_device(args.device)
    base = args.paths_relative_to or args.test_manifest.expanduser().parent

    model, processor = _load_sam(args.model_id, args.weights_dir)
    _freeze_backbone(model)
    model.to(device)
    model.mask_decoder.eval()

    learned = args.auto_prompt == "learned"
    head = build_auto_prompt_head().to(device) if learned else None
    if head is not None:
        head.eval()
    if args.checkpoint is not None:
        _load_checkpoint_into(model, head, args.checkpoint, device)

    records = _labeled_records(args.test_manifest, base)
    print(f"scoring auto-SAM ({args.auto_prompt} prompt) on {len(records)} images...")
    auto = _score_auto_sam(model, head, processor, records, base, device, limit=args.limit)

    report: dict[str, Any] = {
        "test_manifest": str(args.test_manifest),
        "model_id": args.model_id,
        "auto_prompt": args.auto_prompt,
        "checkpoint": str(args.checkpoint) if args.checkpoint else None,
        "auto_sam": auto,
    }
    print(f"  auto-SAM   n={auto['n']:.0f}  Dice={auto['dice']:.4f}  IoU={auto['iou']:.4f}")

    if not args.skip_supervised:
        print("scoring the supervised sorbed segmenter on the same images...")
        supervised = _score_supervised(records, base, args.limit)
        report["supervised"] = supervised
        print(
            f"  supervised ({supervised['backend']})  n={supervised['n']:.0f}  "
            f"Dice={supervised['dice']:.4f}  IoU={supervised['iou']:.4f}"
        )
        report["delta_dice"] = auto["dice"] - supervised["dice"]
        report["delta_iou"] = auto["iou"] - supervised["iou"]
        print(
            f"  delta (auto - supervised)  Dice={report['delta_dice']:+.4f}  "
            f"IoU={report['delta_iou']:+.4f}"
        )

    if args.out is not None:
        args.out.expanduser().parent.mkdir(parents=True, exist_ok=True)
        args.out.expanduser().write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.command == "train":
        return _train(args)
    if args.command == "eval":
        return _evaluate(args)
    raise SystemExit(f"unknown command {args.command!r}")


if __name__ == "__main__":
    raise SystemExit(main())

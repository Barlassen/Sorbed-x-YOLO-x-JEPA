#!/usr/bin/env python3
"""Patch-level tissue linear-probe: does JEPA help wound-tissue detection?

This is the tissue-side counterpart to ``train_grade_jepa.py``. Instead of one
label per image (too few, too imbalanced for tissue), we evaluate the JEPA encoder
as a *dense* feature extractor: every 16x16 patch of a wound image gets a tissue
label from the DFUTissue annotation (0=background, 1..3 = fibrin/granulation/callus),
and a linear classifier is trained on the frozen patch features. This is the
standard self-supervised evaluation (linear probe) for dense prediction, and it
turns 110 images into ~21k labelled patches — enough to measure representation
quality.

The controlled comparison (``--init``):
* ``--init jepa``   — frozen JEPA-pretrained encoder features.
* ``--init random`` — frozen random-weight encoder features (same architecture).

A gap in macro-F1 over the tissue classes is attributable to the JEPA pretraining.

    python -m training.train_tissue_probe --init jepa   --weights runs_jepa/jepa_encoder_tissue.pt
    python -m training.train_tissue_probe --init random
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score

from training.train_jepa import (
    _IMAGENET_MEAN,
    _IMAGENET_STD,
    Encoder,
    JepaConfig,
    _relief_map,
    _standardize,
    wound_box,
)

_ROOT = Path("data/corpus/dfutissue/DFUTissue/Labeled/Original")
_TISSUE = {1: "fibrin", 2: "granulation", 3: "callus"}


def _prepare(img_path: Path, ann_path: Path, cfg: JepaConfig, depth_dir, crop: bool, relief: bool = False):
    """Load image (+depth) and tissue annotation, optionally crop to the wound box
    (defined by the annotation), then return (input tensor, patch labels).

    The crop and depth handling mirror WoundJepaDataset exactly, so the encoder
    sees the same kind of input at test time as it did during pre-training.
    """
    bgr = cv2.imread(str(img_path), cv2.IMREAD_COLOR)
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    ann = cv2.imread(str(ann_path), cv2.IMREAD_GRAYSCALE)
    depth = None
    if depth_dir is not None:
        depth = cv2.imread(str(Path(depth_dir) / f"{img_path.stem}.png"), cv2.IMREAD_GRAYSCALE)
        if depth is None:
            depth = np.zeros(rgb.shape[:2], np.uint8)
        elif depth.shape != rgb.shape[:2]:
            depth = cv2.resize(depth, (rgb.shape[1], rgb.shape[0]))

    if crop:
        box = wound_box((ann > 0).astype(np.uint8))
        if box is not None:
            x0, y0, x1, y1 = box
            rgb, ann = rgb[y0:y1, x0:x1], ann[y0:y1, x0:x1]
            depth = None if depth is None else depth[y0:y1, x0:x1]

    s, g = cfg.img_size, cfg.grid
    t = torch.from_numpy(cv2.resize(rgb, (s, s))).permute(2, 0, 1).float() / 255.0
    t = (t - _IMAGENET_MEAN) / _IMAGENET_STD
    if depth is not None:
        dd = cv2.resize(depth, (s, s)).astype(np.float32)
        t = torch.cat([t, _standardize(torch.from_numpy(dd).unsqueeze(0) / 255.0)], dim=0)
        if relief:
            t = torch.cat([t, _standardize(_relief_map(dd, 15).unsqueeze(0))], dim=0)
    labels = cv2.resize(ann, (g, g), interpolation=cv2.INTER_NEAREST).astype(np.int64).flatten()
    return t, labels


@torch.no_grad()
def _build_xy(encoder: Encoder, split: str, cfg: JepaConfig, device: str, depth_dir=None, crop=False, relief=False):
    img_dir = _ROOT / "Images" / split
    ann_dir = _ROOT / "Annotations" / split
    feats, labels = [], []
    for img_path in sorted(img_dir.glob("*.png")):
        ann = ann_dir / img_path.name
        if not ann.exists():
            continue
        t, y = _prepare(img_path, ann, cfg, depth_dir, crop, relief)
        feats.append(encoder(t.unsqueeze(0).to(device))[0].cpu().numpy())
        labels.append(y)
    return np.concatenate(feats), np.concatenate(labels)


def run(args) -> None:
    torch.manual_seed(args.seed)
    device = args.device
    in_chans = 3 + (1 if args.depth else 0) + (1 if (args.relief and args.depth) else 0)
    cfg = JepaConfig(img_size=args.img_size, in_chans=in_chans)

    encoder = Encoder(cfg).to(device).eval()
    if args.init == "jepa":
        state = torch.load(args.weights, map_location="cpu")
        encoder.load_state_dict(state, strict=False)
        print(f"loaded JEPA encoder from {args.weights}  (in_chans={cfg.in_chans})")
    else:
        print(f"random-init encoder (no pretraining)  (in_chans={cfg.in_chans})")

    Xtr, ytr = _build_xy(encoder, "TrainVal", cfg, device, args.depth, args.crop, args.relief)
    Xte, yte = _build_xy(encoder, "Test", cfg, device, args.depth, args.crop, args.relief)
    print(f"init={args.init}  train patches={len(ytr)}  test patches={len(yte)}")

    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)

    acc = accuracy_score(yte, pred)
    # Macro-F1 over the tissue classes only (background is easy and dominates).
    tissue_ids = [c for c in _TISSUE if c in set(yte)]
    f1_tissue = f1_score(yte, pred, labels=tissue_ids, average="macro", zero_division=0)
    print(f"\nRESULT[{args.init}]  overall_acc={acc:.3f}  tissue_macroF1={f1_tissue:.3f}")
    for c in tissue_ids:
        f1c = f1_score(yte, pred, labels=[c], average="macro", zero_division=0)
        print(f"   {_TISSUE[c]:12s} F1={f1c:.3f}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Patch-level tissue linear probe + JEPA ablation.")
    ap.add_argument("--init", choices=["jepa", "random"], default="jepa")
    ap.add_argument("--weights", default="runs_jepa/jepa_encoder_tissue.pt")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--depth", type=Path, help="Directory of depth PNGs (stem-matched) → 2.5D input.")
    ap.add_argument("--crop", action="store_true", help="Crop to the wound box (must match pretraining).")
    ap.add_argument("--relief", action="store_true", help="Add MinIP/MIP relief channel (must match pretraining).")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    run(ap.parse_args())


if __name__ == "__main__":
    main()

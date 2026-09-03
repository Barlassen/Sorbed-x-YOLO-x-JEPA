#!/usr/bin/env python3
"""Sanity-check the wound masks that mask-guided JEPA relies on.

Two questions, both raised as a concern that the masks might be poor:

1. **Do the masks actually cover the wound?** — a visual montage per image:
   original | ground-truth wound (green, if given) | predicted mask (red overlay).
2. **How good are they numerically?** — Dice between the YOLO wound mask and a
   ground-truth wound mask, where ground truth exists (e.g. DFUTissue: wound =
   annotation > 0). Reports mean/median Dice and how many images the segmenter
   effectively misses (Dice < 0.1 or an empty prediction).

Masks can come from a precomputed dir (exactly what JEPA saw) or be produced live
from the YOLO seg weights with the *same* inference as ``precompute_rgbd.py``
(conf 0.25, per-pixel max over instances, threshold 0.5).

    # DFUTissue, live YOLO masks vs ground-truth wound (Dice + montage):
    python -m training.check_mask_quality \
        --images data/corpus/dfutissue/DFUTissue/Labeled/Original/Images/Test \
        --gt     data/corpus/dfutissue/DFUTissue/Labeled/Original/Annotations/Test \
        --seg-weights runs/segment/runs_wound/yolo26n_fuseg/weights/best.pt \
        --out runs_jepa/mask_check/dfutissue_test

    # Pressure-injury pool, inspect the precomputed masks JEPA used (no GT → visual only):
    python -m training.check_mask_quality \
        --images data/pi_new/images --masks data/pi_new/masks \
        --out runs_jepa/mask_check/pi_pool
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def _dice(pred: np.ndarray, gt: np.ndarray) -> float:
    """Dice over boolean wound masks. Both-empty counts as perfect agreement (1.0)."""
    p, g = pred.astype(bool), gt.astype(bool)
    ps, gs = int(p.sum()), int(g.sum())
    if ps == 0 and gs == 0:
        return 1.0
    return 2.0 * int((p & g).sum()) / (ps + gs)


def _overlay(bgr: np.ndarray, mask: np.ndarray, color) -> np.ndarray:
    """Blend a colored mask onto a copy of the image for visual inspection."""
    out = bgr.copy()
    sel = mask.astype(bool)
    if sel.any():
        tint = np.zeros_like(out); tint[sel] = color
        out[sel] = (0.5 * out[sel] + 0.5 * tint[sel]).astype(np.uint8)
    return out


def _predict_mask(seg, bgr, conf: float) -> np.ndarray:
    """YOLO wound mask, identical inference to precompute_rgbd (0/255 uint8)."""
    h, w = bgr.shape[:2]
    r = seg.predict(bgr, conf=conf, verbose=False)[0]
    if r.masks is not None and len(r.masks.data) > 0:
        u = cv2.resize(r.masks.data.cpu().numpy().max(0).astype(np.float32), (w, h))
        return (u >= 0.5).astype(np.uint8) * 255
    return np.zeros((h, w), np.uint8)


def main() -> None:
    ap = argparse.ArgumentParser(description="Wound-mask quality check (visual + Dice).")
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--gt", type=Path, help="Ground-truth annotation dir (wound = nonzero).")
    ap.add_argument("--masks", type=Path, help="Precomputed mask dir (stem-matched). If unset, run YOLO.")
    ap.add_argument("--seg-weights", default="runs/segment/runs_wound/yolo26n_fuseg/weights/best.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--out", type=Path, default=Path("runs_jepa/mask_check"))
    ap.add_argument("--n-montage", type=int, default=12, help="How many montage images to save.")
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    paths = sorted(p for p in args.images.iterdir() if p.suffix.lower() in _EXTS)
    if not paths:
        raise FileNotFoundError(f"no images under {args.images}")

    seg = None
    if args.masks is None:
        from ultralytics import YOLO
        seg = YOLO(args.seg_weights)

    dices, n_empty, saved = [], 0, 0
    for p in paths:
        bgr = cv2.imread(str(p))
        if bgr is None:
            continue
        h, w = bgr.shape[:2]

        if args.masks is not None:
            mp = args.masks / f"{p.stem}.png"
            pred = cv2.imread(str(mp), cv2.IMREAD_GRAYSCALE)
            pred = np.zeros((h, w), np.uint8) if pred is None else (pred > 127).astype(np.uint8) * 255
            if pred.shape != (h, w):
                pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_NEAREST)
        else:
            pred = _predict_mask(seg, bgr, args.conf)
        if int((pred > 0).sum()) == 0:
            n_empty += 1

        gt = None
        if args.gt is not None:
            g = cv2.imread(str(args.gt / f"{p.stem}.png"), cv2.IMREAD_GRAYSCALE)
            if g is not None:
                if g.shape != (h, w):
                    g = cv2.resize(g, (w, h), interpolation=cv2.INTER_NEAREST)
                gt = (g > 0).astype(np.uint8)
                dices.append(_dice(pred > 0, gt))

        if saved < args.n_montage:
            panels = [bgr]
            if gt is not None:
                panels.append(_overlay(bgr, gt * 255, (0, 255, 0)))   # GT green
            panels.append(_overlay(bgr, pred, (0, 0, 255)))            # pred red
            montage = np.hstack([cv2.resize(x, (256, 256)) for x in panels])
            tag = f"{_dice(pred > 0, gt):.2f}_" if gt is not None else ""
            cv2.imwrite(str(args.out / f"{tag}{p.stem}.png"), montage)
            saved += 1

    n = len(paths)
    print(f"images={n}  montages saved -> {args.out}  (image | "
          + ("GT[green] | " if args.gt else "") + "pred[red])")
    print(f"empty predictions (no wound found): {n_empty}/{n}  ({100 * n_empty / n:.0f}%)")
    if dices:
        d = np.array(dices)
        print(f"Dice vs ground truth  (n={len(d)}):  mean={d.mean():.3f}  median={np.median(d):.3f}  "
              f"min={d.min():.3f}  max={d.max():.3f}")
        print(f"  images with Dice<0.10 (segmenter misses the wound): "
              f"{int((d < 0.10).sum())}/{len(d)}  ({100 * (d < 0.10).mean():.0f}%)")
        print(f"  images with Dice<0.50: {int((d < 0.50).sum())}/{len(d)}  ({100 * (d < 0.50).mean():.0f}%)")
    else:
        print("no ground truth given → visual montage only (open the PNGs under --out).")


if __name__ == "__main__":
    main()

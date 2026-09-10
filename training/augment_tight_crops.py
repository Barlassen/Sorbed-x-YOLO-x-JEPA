#!/usr/bin/env python3
"""Add tight wound-filling crops to a YOLO segmentation training set.

Measured failure mode: the wound segmenter, fine-tuned on FUSeg-style **wide**
shots where the lesion occupies a small part of the frame, silently returns *no
mask* when the wound fills the frame instead. On the JEPA pool that is not a
rounding error — at conf 0.05 it produces empty masks for **48.5%** of the
close-cropped roboflow pressure images (and ~18% of piid / our own pressure
photos), versus 0.9% on azh_fuseg and 1.1% on DFUTissue, the wide-shot corpora it
was trained on. A quarter of the pool therefore reaches mask-guided JEPA with no
guidance at all.

The cause is framing, not tissue: the model never saw "wound occupies most of the
image" during training. This script manufactures exactly those examples from the
data we already have — for every training image that has a wound polygon, it
writes an extra sample cropped to the wound's bounding box plus a random margin,
with the polygons re-normalised into the crop. Only the *train* split is touched;
validation and held-out rulers are never augmented.

    python -m training.augment_tight_crops --dataset data/yolo_seg_combined \
        --copies 1 --min-margin 0.02 --max-margin 0.35

Crops are written as real image files (they are new pixels, not a view), named
``<stem>__crop<k>``; labels follow. Re-running replaces previous crops rather
than stacking them. Pure data preparation — no GPU, no network.
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np

_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
_CROP_TAG = "__crop"


def read_label(path: Path) -> list[np.ndarray]:
    """Read a YOLO-seg label file into a list of (N, 2) normalised polygons."""
    polygons: list[np.ndarray] = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < 7:  # class + at least 3 (x, y) pairs
            continue
        coords = np.asarray([float(v) for v in parts[1:]], dtype=np.float32)
        if coords.size % 2:
            coords = coords[:-1]
        polygons.append(coords.reshape(-1, 2))
    return polygons


def write_label(path: Path, polygons: list[np.ndarray], cls: int = 0) -> None:
    lines = [f"{cls} " + " ".join(f"{v:.6f}" for v in poly.reshape(-1)) for poly in polygons]
    path.write_text("\n".join(lines) + ("\n" if lines else ""))


def crop_window(
    polygons: list[np.ndarray], width: int, height: int, rng: random.Random,
    min_margin: float, max_margin: float,
) -> tuple[int, int, int, int] | None:
    """Pixel window (x0, y0, x1, y1) around all wound polygons, plus a random margin."""
    pts = np.concatenate(polygons, axis=0)
    x0, y0 = pts.min(axis=0)
    x1, y1 = pts.max(axis=0)
    bw, bh = (x1 - x0) * width, (y1 - y0) * height
    if bw < 2 or bh < 2:
        return None
    mx = rng.uniform(min_margin, max_margin) * bw
    my = rng.uniform(min_margin, max_margin) * bh
    cx0 = int(max(0, x0 * width - mx))
    cy0 = int(max(0, y0 * height - my))
    cx1 = int(min(width, x1 * width + mx))
    cy1 = int(min(height, y1 * height + my))
    if cx1 - cx0 < 16 or cy1 - cy0 < 16:
        return None
    return cx0, cy0, cx1, cy1


def reproject(polygons: list[np.ndarray], window: tuple[int, int, int, int],
              width: int, height: int) -> list[np.ndarray]:
    """Re-normalise polygons from the full image into the crop window."""
    cx0, cy0, cx1, cy1 = window
    cw, ch = cx1 - cx0, cy1 - cy0
    out: list[np.ndarray] = []
    for poly in polygons:
        pix = poly * np.array([width, height], dtype=np.float32)
        rel = (pix - np.array([cx0, cy0], dtype=np.float32)) / np.array([cw, ch], dtype=np.float32)
        rel = np.clip(rel, 0.0, 1.0)
        # A polygon that collapsed to a line/point inside the crop is not usable.
        if np.ptp(rel[:, 0]) < 1e-3 or np.ptp(rel[:, 1]) < 1e-3:
            continue
        out.append(rel)
    return out


def augment(dataset: Path, copies: int, min_margin: float, max_margin: float, seed: int) -> dict[str, int]:
    images_dir = dataset / "images" / "train"
    labels_dir = dataset / "labels" / "train"
    if not images_dir.is_dir() or not labels_dir.is_dir():
        raise SystemExit(f"expected {images_dir} and {labels_dir}")

    rng = random.Random(seed)
    # Drop crops from a previous run so re-running is idempotent.
    removed = 0
    for directory in (images_dir, labels_dir):
        for path in list(directory.iterdir()):
            if _CROP_TAG in path.stem:
                path.unlink()
                removed += 1

    sources = [p for p in sorted(images_dir.iterdir())
               if p.suffix.lower() in _EXTS and _CROP_TAG not in p.stem]
    written = skipped = no_wound = 0
    for img_path in sources:
        label_path = labels_dir / f"{img_path.stem}.txt"
        if not label_path.is_file():
            continue
        polygons = read_label(label_path)
        if not polygons:
            no_wound += 1
            continue
        image = cv2.imread(str(img_path))
        if image is None:
            skipped += 1
            continue
        h, w = image.shape[:2]
        for k in range(copies):
            window = crop_window(polygons, w, h, rng, min_margin, max_margin)
            if window is None:
                skipped += 1
                continue
            moved = reproject(polygons, window, w, h)
            if not moved:
                skipped += 1
                continue
            cx0, cy0, cx1, cy1 = window
            crop = image[cy0:cy1, cx0:cx1]
            stem = f"{img_path.stem}{_CROP_TAG}{k}"
            cv2.imwrite(str(images_dir / f"{stem}.png"), crop)
            write_label(labels_dir / f"{stem}.txt", moved)
            written += 1

    return {"sources": len(sources), "removed_previous": removed,
            "crops_written": written, "skipped": skipped, "unlabelled": no_wound}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Add wound-filling crops to a YOLO-seg train split.")
    ap.add_argument("--dataset", required=True, type=Path, help="Merged YOLO dataset directory.")
    ap.add_argument("--copies", type=int, default=1, help="Crops to add per source image.")
    ap.add_argument("--min-margin", type=float, default=0.02, help="Min margin as a fraction of wound size.")
    ap.add_argument("--max-margin", type=float, default=0.35, help="Max margin as a fraction of wound size.")
    ap.add_argument("--seed", type=int, default=0)
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    stats = augment(args.dataset, args.copies, args.min_margin, args.max_margin, args.seed)
    print(f"tight crops -> {args.dataset}/images/train")
    for key, value in stats.items():
        print(f"    {key:20s} {value}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

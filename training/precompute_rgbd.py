#!/usr/bin/env python3
"""Precompute YOLO wound masks and YOLO26 monocular-depth maps for a set of images.

This produces the extra channels the 2.5D mask-guided JEPA needs: for every image
it writes a binary wound mask PNG (from the fine-tuned YOLO26-seg) and a per-image
min-max normalised depth PNG (from YOLO26-depth) into ``<out>/masks`` and
``<out>/depth``, matched by file stem. Run once; JEPA then reads them off disk.

    python -m training.precompute_rgbd --images IMG_DIR --out data/rgbd/pool \
        --seg-weights runs/segment/runs_wound/yolo26n_fuseg/weights/best.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np

_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def main() -> None:
    ap = argparse.ArgumentParser(description="Precompute YOLO masks + depth maps.")
    ap.add_argument("--images", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--seg-weights", default="runs/segment/runs_wound/yolo26n_fuseg/weights/best.pt")
    ap.add_argument("--depth-model", default="yolo26n-depth.pt")
    ap.add_argument("--conf", type=float, default=0.25)
    ap.add_argument("--masks", action="store_true", help="Also write wound masks (default: yes).")
    ap.add_argument("--no-masks", dest="masks", action="store_false")
    ap.set_defaults(masks=True)
    args = ap.parse_args()

    from ultralytics import YOLO

    seg = YOLO(args.seg_weights) if args.masks else None
    depth = YOLO(args.depth_model)

    mask_dir = args.out / "masks"
    depth_dir = args.out / "depth"
    if args.masks:
        mask_dir.mkdir(parents=True, exist_ok=True)
    depth_dir.mkdir(parents=True, exist_ok=True)

    paths = sorted(p for p in Path(args.images).iterdir() if p.suffix.lower() in _EXTS)
    found = 0
    for i, p in enumerate(paths):
        bgr = cv2.imread(str(p))
        if bgr is None:
            continue
        h, w = bgr.shape[:2]

        # Depth → per-image min-max to [0,255] uint8 (relative depth is what we use).
        dr = depth.predict(bgr, verbose=False)[0].depth.data.cpu().numpy().astype(np.float32)
        if dr.shape != (h, w):
            dr = cv2.resize(dr, (w, h), interpolation=cv2.INTER_LINEAR)
        rng = float(np.ptp(dr))
        d8 = (255 * (dr - dr.min()) / (rng + 1e-9)).astype(np.uint8)
        cv2.imwrite(str(depth_dir / f"{p.stem}.png"), d8)

        if args.masks:
            r = seg.predict(bgr, conf=args.conf, verbose=False)[0]
            if r.masks is not None and len(r.masks.data) > 0:
                u = cv2.resize(r.masks.data.cpu().numpy().max(0).astype(np.float32), (w, h))
                m = (u >= 0.5).astype(np.uint8) * 255
                found += 1
            else:
                m = np.zeros((h, w), np.uint8)
            cv2.imwrite(str(mask_dir / f"{p.stem}.png"), m)

        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(paths)} done")

    print(f"depth for {len(paths)} imgs -> {depth_dir}"
          + (f"; masks ({found} with a wound) -> {mask_dir}" if args.masks else ""))


if __name__ == "__main__":
    main()

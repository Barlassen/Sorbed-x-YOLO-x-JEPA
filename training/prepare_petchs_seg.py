#!/usr/bin/env python3
"""Roboflow petchs pressure-ulcer set -> YOLO wound-seg dataset (its test split held out).

The wound region is the annotator's stage outline united with the tissue polygons (``coco_items`` paints both).
Most petchs photos are close-ups where the wound fills the frame -- exactly where the old FUSeg segmenter fails --
so these outlines teach the segmenter in-domain close-ups. ``train`` (minus near-duplicates of a test image) ->
train, ``valid`` -> val; ``test`` stays out and is the second ruler.

    python -m training.prepare_petchs_seg --out data/petchs_yolo
"""

from __future__ import annotations

import argparse
from pathlib import Path

from training.prepare_yolo_seg import _write_label, mask_to_polygons
from training.train_tissue_cnn import RF_PETCHS, coco_items, petchs_test_dups


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--region", choices=("outline", "tissue"), default="outline",
                    help="outline = stage outline + tissue (whole lesion, incl. intact discoloured skin); "
                         "tissue = granulation/slough/eschar polygons only (open wound surface). Images without "
                         "any tissue polygon are skipped in 'tissue' mode.")
    args = ap.parse_args()
    dups = petchs_test_dups()
    for src, split in (("train", "train"), ("valid", "val")):
        (args.out / "images" / split).mkdir(parents=True, exist_ok=True)
        (args.out / "labels" / split).mkdir(parents=True, exist_ok=True)
        n = empty = 0
        for name, img, lab in coco_items(RF_PETCHS, src, dups):
            region = (lab != 0) if args.region == "outline" else ((lab >= 1) & (lab <= 3))
            polys = mask_to_polygons(region.astype("uint8"))
            if not polys:
                empty += 1
                continue
            link = args.out / "images" / split / name
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to((RF_PETCHS / src / name).resolve())
            _write_label(args.out / "labels" / split / f"{Path(name).stem}.txt", polys)
            n += 1
        print(f"{split}: {n} images ({empty} without a wound region skipped)")
    (args.out / "data.yaml").write_text(f"path: {args.out.resolve()}\ntrain: images/train\nval: images/val\nnames:\n  0: wound\n")


if __name__ == "__main__":
    main()

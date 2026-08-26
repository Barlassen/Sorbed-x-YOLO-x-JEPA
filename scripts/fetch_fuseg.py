#!/usr/bin/env python3
"""Fetch the FUSeg (MICCAI Foot Ulcer Segmentation Challenge) dataset.

Downloads real clinical wound images and their expert binary masks from the
public ``uwm-bigdata/wound-segmentation`` repository (the images are committed
directly, so no git-LFS or challenge login is needed). The result is laid out
exactly the way ``scripts/train_segmenter.py`` expects::

    <out>/train/images/0011.png       <out>/train/labels/0011.png
    <out>/validation/images/0002.png  <out>/validation/labels/0002.png

Attribution & license
---------------------
The dataset is provided by the AZH Wound and Vascular Center via
https://github.com/uwm-bigdata/wound-segmentation and accompanies:
C. Wang et al., "Fully Automatic Wound Segmentation with Deep Convolutional
Neural Networks", Scientific Reports 10:21897, 2020. Review that repository's
terms before any commercial use.

Example
-------
    python scripts/fetch_fuseg.py --out data/fuseg --split train --limit 300
    python scripts/fetch_fuseg.py --out data/fuseg --split validation --limit 60
"""

from __future__ import annotations

import argparse
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

_BASE = (
    "https://raw.githubusercontent.com/uwm-bigdata/wound-segmentation/master/"
    "data/Foot%20Ulcer%20Segmentation%20Challenge"
)
_MIN_PNG_BYTES = 200  # anything smaller is a 404 body, not an image


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Download the FUSeg wound dataset.")
    parser.add_argument("--out", type=Path, required=True, help="output dataset directory")
    parser.add_argument("--split", choices=["train", "validation"], default="train")
    parser.add_argument("--limit", type=int, default=300, help="max image/mask pairs")
    parser.add_argument("--max-index", type=int, default=1300, help="highest id to probe")
    parser.add_argument("--workers", type=int, default=16)
    return parser.parse_args(argv)


def _get(url: str) -> bytes | None:
    try:
        with urllib.request.urlopen(url, timeout=30) as response:
            data: bytes = response.read()
    except (urllib.error.HTTPError, urllib.error.URLError, TimeoutError):
        return None
    return data if len(data) >= _MIN_PNG_BYTES else None


def _fetch_pair(index: int, split: str, out: Path) -> str | None:
    ident = f"{index:04d}"
    img_path = out / split / "images" / f"{ident}.png"
    lbl_path = out / split / "labels" / f"{ident}.png"
    if img_path.exists() and lbl_path.exists():
        return ident
    image = _get(f"{_BASE}/{split}/images/{ident}.png")
    if image is None:
        return None
    label = _get(f"{_BASE}/{split}/labels/{ident}.png")
    if label is None:
        return None
    img_path.write_bytes(image)
    lbl_path.write_bytes(label)
    return ident


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    (args.out / args.split / "images").mkdir(parents=True, exist_ok=True)
    (args.out / args.split / "labels").mkdir(parents=True, exist_ok=True)

    downloaded = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [
            pool.submit(_fetch_pair, i, args.split, args.out)
            for i in range(args.max_index + 1)
        ]
        for future in futures:
            if future.result() is not None:
                downloaded += 1
                if downloaded >= args.limit:
                    break

    total = len(list((args.out / args.split / "images").glob("*.png")))
    print(f"{args.split}: {total} image/mask pairs in {args.out / args.split}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

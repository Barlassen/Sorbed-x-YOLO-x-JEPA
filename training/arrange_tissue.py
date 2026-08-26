#!/usr/bin/env python3
"""Remap a tissue-mask dataset's native classes onto Sorbed's unified schema.

The single-model tissue segmenter (``configs/seg_tissue_segformer.yaml``) expects
mask pixel values in ONE fixed order::

    0 background   1 epithelial   2 granulation   3 slough
    4 eschar       5 adipose      6 deep_structure   (muscle/tendon/bone)

No public tissue set ships this schema — DFUTissue, WoundTissue, Wounds-307 and
ComplexWoundDB each use their own class ids (some as grayscale indices, some as
RGB colours). This tool applies an operator-supplied, per-dataset mapping to
produce a ``generic`` ``images/`` + ``masks/`` layout in the unified order, plus a
``mapping_report.json`` that lists every native value it saw and how many pixels
it covered — so you can VERIFY the mapping against the real masks before training.

It never guesses: a pixel value with no mapping entry is an error by default
(``--on-unmapped error``), because silently folding an unknown tissue into
background would fabricate a clinical label. Use ``--on-unmapped background`` only
after you have inspected the report and are sure the residual values are truly
off-wound.

    python training/arrange_tissue.py \
        --images /data/sorbed/corpus/dfutissue/DFUTissue/images \
        --masks  /data/sorbed/corpus/dfutissue/DFUTissue/labels \
        --mapping training/configs/tissue_maps/dfutissue.yaml \
        --out /data/sorbed/corpus/arranged/dfutissue

The mapping file is YAML::

    mode: grayscale        # or 'rgb'
    unmapped: error        # optional default for --on-unmapped
    map:                   # native value -> unified class NAME (not index)
      0: background
      1: granulation
      2: slough
      3: eschar
      4: deep_structure    # e.g. DFUTissue 'bone'/'tendon' collapse here
    # rgb mode keys are "R,G,B" strings, e.g. "255,0,0": granulation
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from collections import Counter
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
else:  # pragma: no cover
    pass

# The unified class order — the ONLY source of truth for index assignment.
UNIFIED_CLASSES: tuple[str, ...] = (
    "background", "epithelial", "granulation", "slough",
    "eschar", "adipose", "deep_structure",
)
UNIFIED_INDEX: dict[str, int] = {name: i for i, name in enumerate(UNIFIED_CLASSES)}

IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})


class MappingError(ValueError):
    """Raised when a mask contains a value the mapping does not cover."""


def load_mapping(path: Path) -> tuple[str, dict, str]:
    """Load a tissue mapping YAML -> ``(mode, native->index, unmapped_policy)``.

    ``mode`` is ``"grayscale"`` or ``"rgb"``. For grayscale the returned map keys
    are ``int`` native values; for rgb they are ``(r, g, b)`` int tuples. Values
    are validated against :data:`UNIFIED_INDEX` and stored as unified indices.
    """
    import yaml

    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        spec = yaml.safe_load(handle) or {}
    mode = str(spec.get("mode", "grayscale")).lower()
    if mode not in ("grayscale", "rgb"):
        raise MappingError(f"mapping mode must be 'grayscale' or 'rgb', got {mode!r}")
    unmapped = str(spec.get("unmapped", "error")).lower()
    raw = spec.get("map")
    if not raw:
        raise MappingError(f"mapping {path} has no 'map:' entries")

    resolved: dict = {}
    for native, class_name in raw.items():
        name = str(class_name).strip()
        if name not in UNIFIED_INDEX:
            raise MappingError(
                f"mapping {path}: class {name!r} is not one of {list(UNIFIED_CLASSES)}"
            )
        index = UNIFIED_INDEX[name]
        if mode == "grayscale":
            resolved[int(native)] = index
        else:
            parts = tuple(int(x) for x in str(native).split(","))
            if len(parts) != 3:
                raise MappingError(f"rgb key {native!r} must be 'R,G,B'")
            resolved[parts] = index
    return mode, resolved, unmapped


def _pair_by_stem(images_dir: Path, masks_dir: Path) -> list[tuple[Path, Path]]:
    masks = {m.stem: m for m in masks_dir.iterdir() if m.suffix.lower() in IMAGE_SUFFIXES}
    pairs = []
    for image in sorted(images_dir.iterdir()):
        if image.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        mask = masks.get(image.stem)
        if mask is not None:
            pairs.append((image, mask))
    if not pairs:
        raise SystemExit(f"no image/mask pairs between {images_dir} and {masks_dir}")
    return pairs


def remap_mask(
    mask_path: Path,
    mode: str,
    native_to_index: dict,
    *,
    on_unmapped: str,
) -> tuple[np.ndarray, Counter]:
    """Return the unified-index mask and a per-native-value pixel-count Counter.

    ``on_unmapped`` is ``"error"`` (raise) or ``"background"`` (map to class 0).
    """
    import cv2

    counts: Counter = Counter()
    if mode == "grayscale":
        raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if raw is None:
            raise SystemExit(f"could not read mask {mask_path}")
        out = np.zeros_like(raw, dtype=np.uint8)
        for value in np.unique(raw):
            native = int(value)
            pixels = int((raw == value).sum())
            counts[native] = pixels
            if native in native_to_index:
                out[raw == value] = native_to_index[native]
            elif on_unmapped == "background":
                out[raw == value] = 0
            else:
                raise MappingError(
                    f"{mask_path.name}: native value {native} ({pixels} px) has no "
                    f"mapping; add it or pass --on-unmapped background"
                )
        return out, counts

    bgr = cv2.imread(str(mask_path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise SystemExit(f"could not read mask {mask_path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    out = np.zeros(rgb.shape[:2], dtype=np.uint8)
    flat = rgb.reshape(-1, 3)
    colours, inverse = np.unique(flat, axis=0, return_inverse=True)
    inverse = inverse.reshape(rgb.shape[:2])
    for ci, colour in enumerate(colours):
        key = (int(colour[0]), int(colour[1]), int(colour[2]))
        pixels = int((inverse == ci).sum())
        counts[key] = pixels
        if key in native_to_index:
            out[inverse == ci] = native_to_index[key]
        elif on_unmapped == "background":
            out[inverse == ci] = 0
        else:
            raise MappingError(
                f"{mask_path.name}: colour {key} ({pixels} px) has no mapping; "
                f"add it or pass --on-unmapped background"
            )
    return out, counts


def arrange(
    images_dir: Path,
    masks_dir: Path,
    mapping_path: Path,
    out_dir: Path,
    *,
    on_unmapped: str | None = None,
) -> dict[str, object]:
    """Remap every mask and copy images into ``out_dir/{images,masks}``."""
    import cv2

    mode, native_to_index, policy = load_mapping(mapping_path)
    on_unmapped = (on_unmapped or policy or "error").lower()
    if on_unmapped not in ("error", "background"):
        raise SystemExit("--on-unmapped must be 'error' or 'background'")

    out_dir = out_dir.expanduser()
    out_images = out_dir / "images"
    out_masks = out_dir / "masks"
    out_images.mkdir(parents=True, exist_ok=True)
    out_masks.mkdir(parents=True, exist_ok=True)

    pairs = _pair_by_stem(images_dir, masks_dir)
    total_native: Counter = Counter()
    class_pixels: Counter = Counter()
    for image_path, mask_path in pairs:
        unified, counts = remap_mask(
            mask_path, mode, native_to_index, on_unmapped=on_unmapped
        )
        total_native.update(counts)
        for idx in range(len(UNIFIED_CLASSES)):
            class_pixels[UNIFIED_CLASSES[idx]] += int((unified == idx).sum())
        shutil.copy2(image_path, out_images / image_path.name)
        cv2.imwrite(str(out_masks / f"{mask_path.stem}.png"), unified)

    report: dict[str, object] = {
        "mode": mode,
        "pairs": len(pairs),
        "on_unmapped": on_unmapped,
        "native_value_pixels": {str(k): v for k, v in sorted(total_native.items())},
        "unified_class_pixels": dict(class_pixels),
        "unified_classes": list(UNIFIED_CLASSES),
    }
    report_path = out_dir / "mapping_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--images", type=Path, required=True)
    parser.add_argument("--masks", type=Path, required=True)
    parser.add_argument("--mapping", type=Path, required=True,
                        help="YAML mapping (see module docstring / configs/tissue_maps)")
    parser.add_argument("--out", type=Path, required=True,
                        help="output dir; images/ + masks/ written in the unified 7-class order")
    parser.add_argument("--on-unmapped", choices=["error", "background"], default=None,
                        help="override the mapping file's 'unmapped:' policy")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = arrange(
        args.images, args.masks, args.mapping, args.out, on_unmapped=args.on_unmapped
    )
    print(f"arranged {report['pairs']} pairs -> {args.out}/images,masks")
    print("unified class pixel coverage:")
    for name in UNIFIED_CLASSES:
        print(f"  {name:14s} {report['unified_class_pixels'].get(name, 0):>12,} px")
    print(f"see {args.out}/mapping_report.json to verify native-value coverage")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

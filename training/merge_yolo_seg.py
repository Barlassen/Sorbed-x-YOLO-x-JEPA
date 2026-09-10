#!/usr/bin/env python3
"""Merge several YOLO segmentation datasets into one clean, leakage-free set.

``build_pool.py`` does this for the *unlabeled* JEPA pool; this is its labelled
counterpart. The wound segmenter is now trained from three prepared sources
(FUSeg/AZH, CO2Wounds-V2, DFUTissue), and the same two hazards apply — with a
sharper edge, because here a leaked image corrupts a **scored** ruler:

* **Leakage.** Public wound sets re-host each other's photos, and some ship
  splits that overlap themselves (measured: 7 AZH/FUSeg training images are
  near-identical to images in its own held-out validation split; 10 CO2Wounds
  training images duplicate its own val). Every ``--exclude`` directory is
  hashed, and any training image within ``--leak-hamming`` of one is dropped.
* **Duplication.** One representative is kept per near-duplicate group, at the
  looser ``--hamming`` radius.

Sources are given as *image* directories in Ultralytics layout; the label
directory is found by swapping ``/images/`` for ``/labels/``. Images and labels
are sym-linked (never copied) into ``<out>/{images,labels}/{train,val}`` under
names prefixed by their source tag, so identical stems from different corpora
cannot overwrite one another. A ``data.yaml`` and a JSON report are written.

    python -m training.merge_yolo_seg \
        --train data/yolo_fuseg/images/train:fuseg \
        --train data/yolo_co2wounds/images/train:co2 \
        --train data/dfutissue_yolo/images/train:dfut \
        --val   data/yolo_co2wounds/images/val:co2 \
        --val   data/dfutissue_yolo/images/val:dfut \
        --exclude data/yolo_fuseg/images/val \
        --out data/yolo_seg_combined

Held-out test images belong in ``--exclude``, never in ``--train``/``--val``.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from dedup import dhash, hamming
else:  # pragma: no cover - exercised only when imported as a package
    from training.dedup import dhash, hamming

_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


def labels_dir_for(images_dir: Path) -> Path:
    """The Ultralytics label directory paired with an image directory."""
    parts = list(images_dir.parts)
    for i in range(len(parts) - 1, -1, -1):
        if parts[i] == "images":
            parts[i] = "labels"
            return Path(*parts)
    raise SystemExit(f"no 'images' component in {images_dir} — cannot locate labels")


def list_images(directory: Path) -> list[Path]:
    """Image files directly inside ``directory`` (symlinks included)."""
    if not directory.is_dir():
        raise SystemExit(f"not a directory: {directory}")
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in _EXTS)


class Item:
    """One labelled image: where it lives, its source tag, and its hash."""

    __slots__ = ("image", "label", "tag", "split", "hash")

    def __init__(self, image: Path, label: Path, tag: str, split: str):
        self.image = image
        self.label = label
        self.tag = tag
        self.split = split
        self.hash: int | None = None


def collect(specs: list[str], split: str) -> list[Item]:
    """Turn ``DIR[:TAG]`` specs into items, pairing each image with its label."""
    items: list[Item] = []
    for spec in specs:
        raw, _, tag = spec.rpartition(":")
        if not raw or Path(spec).is_dir():
            raw, tag = spec, Path(spec).parent.parent.name
        images_dir = Path(raw)
        labels_dir = labels_dir_for(images_dir)
        for image in list_images(images_dir):
            label = labels_dir / f"{image.stem}.txt"
            if not label.is_file():
                continue  # an image without a label is not usable supervision
            items.append(Item(image, label, tag or "src", split))
    return items


def drop_leaks(items: list[Item], excludes: list[Path], radius: int) -> tuple[list[Item], dict[str, int]]:
    """Remove items near-identical to any excluded (held-out) image."""
    excl_hashes = [
        h for d in excludes for h in (dhash(p) for p in list_images(d)) if h is not None
    ]
    excl_set = set(excl_hashes)
    kept: list[Item] = []
    dropped: dict[str, int] = defaultdict(int)
    for item in items:
        if item.hash is None:
            kept.append(item)
            continue
        leaked = item.hash in excl_set or (
            radius > 0 and any(hamming(item.hash, e) <= radius for e in excl_hashes)
        )
        if leaked:
            dropped[item.tag] += 1
        else:
            kept.append(item)
    return kept, dict(dropped)


def dedupe(items: list[Item], radius: int) -> tuple[list[Item], int]:
    """Keep one representative per near-duplicate group; return (kept, dropped)."""
    by_hash: dict[int, list[int]] = defaultdict(list)
    for i, item in enumerate(items):
        if item.hash is not None:
            by_hash[item.hash].append(i)
    unique = list(by_hash)
    parent = {h: h for h in unique}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    if radius > 0:
        for a in range(len(unique)):
            for b in range(a + 1, len(unique)):
                if hamming(unique[a], unique[b]) <= radius:
                    parent[find(unique[a])] = find(unique[b])

    clusters: dict[int, list[int]] = defaultdict(list)
    for h in unique:
        clusters[find(h)].extend(by_hash[h])

    kept: list[Item] = []
    dropped = 0
    seen: set[int] = set()
    for members in clusters.values():
        members.sort()
        kept.append(items[members[0]])
        dropped += len(members) - 1
        seen.update(members)
    kept.extend(items[i] for i in range(len(items)) if items[i].hash is None and i not in seen)
    return kept, dropped


def write_dataset(items: list[Item], out: Path) -> dict[str, int]:
    """Sym-link kept items into the merged dataset; return per-split counts."""
    counts: dict[str, int] = defaultdict(int)
    for split in ("train", "val"):
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
    for idx, item in enumerate(items):
        name = f"{item.tag}_{idx:05d}_{item.image.stem}"
        img_link = out / "images" / item.split / f"{name}{item.image.suffix.lower()}"
        lbl_link = out / "labels" / item.split / f"{name}.txt"
        for link, target in ((img_link, item.image), (lbl_link, item.label)):
            if link.exists() or link.is_symlink():
                link.unlink()
            link.symlink_to(target.resolve())
        counts[item.split] += 1
    return dict(counts)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Merge YOLO segmentation datasets, leakage-free.")
    ap.add_argument("--train", action="append", default=[], metavar="DIR[:TAG]",
                    help="Training image dir in Ultralytics layout (repeatable).")
    ap.add_argument("--val", action="append", default=[], metavar="DIR[:TAG]",
                    help="Validation (model-selection) image dir (repeatable).")
    ap.add_argument("--exclude", action="append", default=[], type=Path, metavar="DIR",
                    help="Held-out image dir; anything matching it is dropped (repeatable).")
    ap.add_argument("--out", required=True, type=Path, help="Merged dataset directory.")
    ap.add_argument("--hamming", type=int, default=2, help="Radius for train-vs-train duplicates.")
    ap.add_argument("--leak-hamming", type=int, default=2, help="Tight radius for held-out leakage.")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.train:
        raise SystemExit("at least one --train is required")

    items = collect(args.train, "train") + collect(args.val, "val")
    scanned = defaultdict(int)
    for item in items:
        scanned[f"{item.tag}/{item.split}"] += 1
        item.hash = dhash(item.image)

    # Validation images are held-out for model selection: training must not see them.
    val_dirs = [Path(spec.rpartition(":")[0] or spec) for spec in args.val]
    train_items = [i for i in items if i.split == "train"]
    val_items = [i for i in items if i.split == "val"]

    train_items, leaked = drop_leaks(train_items, list(args.exclude) + val_dirs, args.leak_hamming)
    train_items, dupes = dedupe(train_items, args.hamming)
    val_items, val_leaked = drop_leaks(val_items, list(args.exclude), args.leak_hamming)

    counts = write_dataset(train_items + val_items, args.out)

    yaml_path = args.out / "data.yaml"
    yaml_path.write_text(
        "# Merged YOLO segmentation dataset — training/merge_yolo_seg.py\n"
        f"path: {args.out.resolve()}\n"
        "train: images/train\n"
        "val: images/val\n"
        "names:\n"
        "  0: wound\n"
    )
    report = {
        "scanned": dict(scanned),
        "train_leakage_dropped": leaked,
        "val_leakage_dropped": val_leaked,
        "train_duplicates_dropped": dupes,
        "final": counts,
        "hamming": args.hamming,
        "leak_hamming": args.leak_hamming,
    }
    (args.out / "merge_report.json").write_text(json.dumps(report, indent=2))

    print(f"merged -> {args.out}")
    for key, n in sorted(scanned.items()):
        print(f"    scanned {key:24s} {n:5d}")
    print(f"  train: {dupes} duplicates, {sum(leaked.values())} leakage dropped {leaked or ''}")
    print(f"  val:   {sum(val_leaked.values())} leakage dropped {val_leaked or ''}")
    print(f"  final: train={counts.get('train', 0)}  val={counts.get('val', 0)}")
    print(f"  {yaml_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Perceptual-hash near-duplicate removal across a combined wound manifest.

Several public wound sets share the *same* photographs (AZH and FUSeg overlap;
Kaggle re-hosts of Medetec; WSNet aggregates earlier corpora). If a duplicate
lands in both the train and val split it inflates the reported score and hides
the very cross-hospital generalization we care about. This tool computes a
difference-hash (dHash) for every image in a manifest, groups images whose
hashes are within a Hamming radius, and keeps exactly one representative per
group — writing a de-duplicated manifest plus an auditable report of what was
dropped and why.

It is deliberately dependency-light: Pillow + numpy (both Sorbed core deps). No
network, no learned model — dHash is a deterministic, explainable fingerprint.

    python training/dedup.py \
        --manifest /data/sorbed/manifests/combined/train.jsonl \
        --out /data/sorbed/manifests/combined/train.dedup.jsonl \
        --hamming 6

The representative kept per duplicate group is the one whose source appears
*first* in ``--source-priority`` (default: the manifest order), so you can prefer
the set with the better masks. Cross-split leakage is a separate concern: run
this BEFORE ``data_prep.py`` splits, or across the union of splits, so a photo
cannot survive in two of them.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from datasets import ManifestRecord, read_manifest, write_manifest
else:  # pragma: no cover - exercised only when imported as a package
    from training.datasets import ManifestRecord, read_manifest, write_manifest

# dHash side length: the image is resized to (HASH_SIZE+1) x HASH_SIZE and each
# row's adjacent-pixel comparisons yield HASH_SIZE*HASH_SIZE bits. 8 -> 64 bits.
HASH_SIZE = 8


def dhash(path: Path, *, hash_size: int = HASH_SIZE) -> int | None:
    """Difference hash of an image as an integer, or ``None`` if unreadable.

    Grayscale, resize to ``(hash_size+1, hash_size)``, then set a bit per
    left-to-right "is brighter than the next pixel" comparison. Robust to scale,
    mild compression, and colour shifts; sensitive to real content changes.
    """
    from PIL import Image

    try:
        with Image.open(path) as handle:
            gray = handle.convert("L").resize(
                (hash_size + 1, hash_size), Image.Resampling.LANCZOS
            )
    except (OSError, ValueError):
        return None
    pixels = np.asarray(gray, dtype=np.int16)
    diff = pixels[:, 1:] > pixels[:, :-1]
    bits = 0
    for bit in diff.flatten():
        bits = (bits << 1) | int(bit)
    return bits


def hamming(a: int, b: int) -> int:
    """Number of differing bits between two hashes."""
    return int(bin(a ^ b).count("1"))


def _resolve(record: ManifestRecord, root: Path | None) -> Path:
    path = Path(record.image_path)
    if not path.is_absolute() and root is not None:
        path = root / path
    return path


def group_duplicates(
    records: list[ManifestRecord],
    *,
    root: Path | None = None,
    hamming_radius: int = 6,
    hash_size: int = HASH_SIZE,
) -> tuple[list[list[int]], list[int]]:
    """Group record indices into near-duplicate clusters.

    Returns ``(groups, unreadable)`` where each group is a list of record indices
    with mutually close hashes (single-link over the Hamming radius) and
    ``unreadable`` lists indices whose image could not be hashed. A radius of 0
    means exact-hash only; the default 6/64 bits tolerates re-compression and
    resizing while keeping distinct wounds apart.

    The clustering buckets by exact hash first (the common case — literal
    re-hosts), then single-links buckets within the radius, so it is near-linear
    for the exact-duplicate majority and only pays the pairwise cost across the
    far smaller set of distinct hashes.
    """
    hashes: list[int | None] = []
    unreadable: list[int] = []
    for i, record in enumerate(records):
        h = dhash(_resolve(record, root), hash_size=hash_size)
        hashes.append(h)
        if h is None:
            unreadable.append(i)

    by_hash: dict[int, list[int]] = defaultdict(list)
    for i, h in enumerate(hashes):
        if h is not None:
            by_hash[h].append(i)

    unique_hashes = list(by_hash)
    # Union-find over distinct hashes; link two buckets within the radius.
    parent = {h: h for h in unique_hashes}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        parent[find(x)] = find(y)

    for a_idx in range(len(unique_hashes)):
        for b_idx in range(a_idx + 1, len(unique_hashes)):
            ha, hb = unique_hashes[a_idx], unique_hashes[b_idx]
            if hamming(ha, hb) <= hamming_radius:
                union(ha, hb)

    clusters: dict[int, list[int]] = defaultdict(list)
    for h in unique_hashes:
        clusters[find(h)].extend(by_hash[h])
    groups = [sorted(members) for members in clusters.values()]
    groups.sort(key=lambda g: g[0])
    return groups, unreadable


def _priority_key(
    record: ManifestRecord, source_priority: list[str]
) -> tuple[int, int, str]:
    """Sort key that prefers higher-priority sources, then masked, then name."""
    try:
        rank = source_priority.index(record.source)
    except ValueError:
        rank = len(source_priority)
    has_mask = 0 if record.mask_path else 1  # masked first
    return (rank, has_mask, record.image_path)


def deduplicate(
    records: list[ManifestRecord],
    *,
    root: Path | None = None,
    hamming_radius: int = 6,
    source_priority: list[str] | None = None,
    hash_size: int = HASH_SIZE,
) -> tuple[list[ManifestRecord], dict[str, object]]:
    """Keep one representative per near-duplicate group; return kept + report."""
    source_priority = source_priority or []
    groups, unreadable = group_duplicates(
        records, root=root, hamming_radius=hamming_radius, hash_size=hash_size
    )
    kept: list[ManifestRecord] = []
    dropped_pairs: list[dict[str, str]] = []
    for group in groups:
        members = sorted(group, key=lambda i: _priority_key(records[i], source_priority))
        representative = members[0]
        kept.append(records[representative])
        for other in members[1:]:
            dropped_pairs.append(
                {
                    "dropped": records[other].image_path,
                    "dropped_source": records[other].source,
                    "kept": records[representative].image_path,
                    "kept_source": records[representative].source,
                }
            )
    # Unreadable images are kept (we cannot judge them) but flagged.
    for i in unreadable:
        if records[i] not in kept:
            kept.append(records[i])
    kept.sort(key=lambda r: r.image_path)
    report: dict[str, object] = {
        "input": len(records),
        "kept": len(kept),
        "dropped": len(dropped_pairs),
        "duplicate_groups": sum(1 for g in groups if len(g) > 1),
        "unreadable": len(unreadable),
        "hamming_radius": hamming_radius,
        "dropped_pairs": dropped_pairs,
    }
    return kept, report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--manifest", type=Path, required=True,
                        help="input manifest (.jsonl/.csv)")
    parser.add_argument("--out", type=Path, required=True,
                        help="output de-duplicated manifest (.jsonl; .csv written alongside)")
    parser.add_argument("--root", type=Path, default=None,
                        help="root to resolve relative image paths against")
    parser.add_argument("--hamming", type=int, default=6,
                        help="max Hamming distance (of 64 bits) to call two images duplicates")
    parser.add_argument("--source-priority", nargs="*", default=None,
                        help="source ids in keep-preference order (best masks first)")
    parser.add_argument("--no-csv", action="store_true", help="write only the JSONL output")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    records = read_manifest(Path(args.manifest))
    if not records:
        raise SystemExit(f"no records in {args.manifest}")
    kept, report = deduplicate(
        records, root=args.root, hamming_radius=int(args.hamming),
        source_priority=args.source_priority,
    )
    out = Path(args.out).expanduser()
    write_manifest(kept, out, also_csv=not args.no_csv)
    report_path = out.with_suffix(".dedup_report.json")
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    print(
        f"dedup: {report['input']} -> {report['kept']} kept "
        f"({report['dropped']} near-duplicates dropped across "
        f"{report['duplicate_groups']} groups; {report['unreadable']} unreadable) "
        f"-> {out.name}, {report_path.name}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Build a leakage-free, de-duplicated image pool for JEPA pre-training.

The mask-guided JEPA reads its pre-training images from a single flat directory
(``<pool>/images``); ``precompute_rgbd.py`` then fills ``<pool>/masks`` and
``<pool>/depth`` keyed by file *stem*. This tool assembles that ``images``
directory from several scattered source folders while enforcing the two data
rules the honest evaluation depends on:

1. **De-duplication (perceptual).** Public wound sets re-host the same photos,
   and our own pressure-ulcer set has literal copies. We reuse the difference-
   hash (dHash) from :mod:`training.dedup` to group near-identical images
   (Hamming radius over 64 bits, so re-compressed / resized copies are caught,
   not just byte-identical ones) and keep exactly one representative per group.

2. **Leakage exclusion.** The tissue-probe *test* images (DFUTissue ``Labeled``)
   must never enter pre-training. Every ``--exclude`` image is hashed too; any
   duplicate group that contains an excluded image is dropped **whole**, so even
   a re-compressed copy of a test image cannot slip into the pool.

Survivors are sym-linked into ``<pool>/images`` under unique names
(``NNNNN_<tag>_<stem><ext>``) so no two images share a stem — otherwise
``precompute_rgbd`` would overwrite one image's mask/depth with another's. The
source files are never copied or modified. A JSON report records, per source,
how many images were scanned, dropped as near-duplicates, or excluded as
leakage, and the final pool size.

Symlinks are skipped while scanning (``--follow-symlinks`` to include them), so
pointing ``--source`` at an existing pool directory picks up only its real
files — handy for the 732 pressure-ulcer photos, which live as real files in
``data/rgbd_pool/images`` alongside symlinks to the foot-wound corpora.

Example (dry-run first, then drop --dry-run to create the links)::

    python -m training.build_pool \
        --source data/rgbd_pool/images \
        --source data/corpus/azh_fuseg \
        --source data/corpus/dfutissue/DFUTissue/Unlabeled \
        --source data/corpus/piid_data \
        --source data/corpus/kaggle_stages --ignore-dirname Invalid \
        --source data/corpus/roboflow_fr7kn \
        --exclude data/corpus/dfutissue/DFUTissue/Labeled \
        --out data/rgbd_pool_v2 --hamming 6 \
        --priority pressure fr7kn --dry-run
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import defaultdict
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from dedup import dhash, hamming
else:  # pragma: no cover - exercised only when imported as a package
    from training.dedup import dhash, hamming

_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
_TAG_RE = re.compile(r"[^a-z0-9]+")


def _tag(source_root: Path, override: str | None) -> str:
    """Short, filesystem-safe source label used in the sym-link name."""
    raw = override if override else source_root.name
    return _TAG_RE.sub("-", raw.lower()).strip("-") or "src"


def scan_images(
    root: Path, *, follow_symlinks: bool, ignore_dirnames: set[str]
) -> list[Path]:
    """Recursively list image files under ``root``.

    Symlinks are skipped unless ``follow_symlinks`` is set, so an existing pool
    directory yields only its real files. Any file under a directory segment in
    ``ignore_dirnames`` (case-insensitive) is skipped.
    """
    lowered = {d.lower() for d in ignore_dirnames}
    out: list[Path] = []
    for path in sorted(root.rglob("*")):
        if path.suffix.lower() not in _EXTS:
            continue
        if not follow_symlinks and path.is_symlink():
            continue
        if lowered & {part.lower() for part in path.relative_to(root).parts}:
            continue
        out.append(path)
    return out


class _Item:
    """One scanned image: its path, source tag, and keep/exclude role."""

    __slots__ = ("path", "tag", "priority", "is_exclude", "hash")

    def __init__(self, path: Path, tag: str, priority: int, is_exclude: bool):
        self.path = path
        self.tag = tag
        self.priority = priority
        self.is_exclude = is_exclude
        self.hash: int | None = None


def _cluster(items: list[_Item], hamming_radius: int) -> list[list[int]]:
    """Group item indices into near-duplicate clusters (single-link over radius).

    Exact-hash bucketing first (the common re-host case), then a union-find that
    links distinct hashes within the Hamming radius — the same strategy as
    :func:`training.dedup.group_duplicates`, but over plain items so keep and
    exclude images can share a cluster.
    """
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

    if hamming_radius > 0:
        for a in range(len(unique)):
            for b in range(a + 1, len(unique)):
                if hamming(unique[a], unique[b]) <= hamming_radius:
                    parent[find(unique[a])] = find(unique[b])

    clusters: dict[int, list[int]] = defaultdict(list)
    for h in unique:
        clusters[find(h)].extend(by_hash[h])
    groups = [sorted(m) for m in clusters.values()]
    groups.sort(key=lambda g: g[0])
    return groups


def build(
    sources: list[tuple[Path, str, int]],
    excludes: list[Path],
    *,
    hamming_radius: int,
    leak_hamming: int,
    follow_symlinks: bool,
    ignore_dirnames: set[str],
) -> tuple[list[_Item], dict[str, object]]:
    """Resolve sources + excludes into the kept item list and a report.

    Two independent radii, because "leakage" and "duplicate" are different
    questions. Leakage asks *is this the same test photo re-appearing* — a tight
    ``leak_hamming`` (near-exact), so genuinely distinct but visually similar
    wounds are not nuked. Dedup asks *are these two training photos the same
    shot* — the looser ``hamming_radius`` that tolerates re-hosting.
    """
    keep_items: list[_Item] = []
    per_source_scanned: dict[str, int] = defaultdict(int)
    for root, tag, priority in sources:
        for path in scan_images(
            root, follow_symlinks=follow_symlinks, ignore_dirnames=ignore_dirnames
        ):
            keep_items.append(_Item(path, tag, priority, is_exclude=False))
            per_source_scanned[tag] += 1

    excl_paths: list[Path] = []
    for root in excludes:
        excl_paths += scan_images(
            root, follow_symlinks=follow_symlinks, ignore_dirnames=ignore_dirnames
        )

    unreadable = 0
    for item in keep_items:
        item.hash = dhash(item.path)
        if item.hash is None:
            unreadable += 1
    excl_hashes = [h for h in (dhash(p) for p in excl_paths) if h is not None]
    excl_set = set(excl_hashes)

    # --- Step 1: drop test-set leakage (tight radius) -----------------------
    survivors: list[_Item] = []
    leaked_dropped = 0
    leaked_by_source: dict[str, int] = defaultdict(int)
    for item in keep_items:
        if item.hash is None:
            survivors.append(item)
            continue
        leaked = item.hash in excl_set
        if not leaked and leak_hamming > 0:
            leaked = any(hamming(item.hash, eh) <= leak_hamming for eh in excl_hashes)
        if leaked:
            leaked_dropped += 1
            leaked_by_source[item.tag] += 1
        else:
            survivors.append(item)

    # --- Step 2: de-duplicate survivors among themselves (loose radius) -----
    groups = _cluster(survivors, hamming_radius)
    kept: list[_Item] = []
    dropped_dupe = 0
    hashed_ids: set[int] = set()
    for group in groups:
        members = [survivors[i] for i in group]
        for m in members:
            hashed_ids.add(id(m))
        members.sort(key=lambda m: (m.priority, str(m.path)))
        kept.append(members[0])
        dropped_dupe += len(members) - 1
    unreadable_kept = 0
    for item in survivors:
        if item.hash is None and id(item) not in hashed_ids:
            kept.append(item)
            unreadable_kept += 1

    kept.sort(key=lambda m: (m.priority, str(m.path)))
    report: dict[str, object] = {
        "scanned_keep": len(keep_items),
        "scanned_exclude": len(excl_paths),
        "per_source_scanned": dict(per_source_scanned),
        "unreadable": unreadable,
        "unreadable_kept": unreadable_kept,
        "near_duplicate_groups": sum(1 for g in groups if len(g) > 1),
        "dropped_near_duplicates": dropped_dupe,
        "leakage_dropped": leaked_dropped,
        "leakage_by_source": dict(leaked_by_source),
        "kept": len(kept),
        "hamming_radius": hamming_radius,
        "leak_hamming": leak_hamming,
    }
    return kept, report


def write_pool(kept: list[_Item], out: Path, *, dry_run: bool) -> Path:
    """Sym-link kept items into ``out/images`` under unique names."""
    images = out / "images"
    if not dry_run:
        images.mkdir(parents=True, exist_ok=True)
    kept_per_source: dict[str, int] = defaultdict(int)
    for idx, item in enumerate(kept):
        kept_per_source[item.tag] += 1
        name = f"{idx:05d}_{item.tag}_{item.path.stem}{item.path.suffix.lower()}"
        link = images / name
        if dry_run:
            continue
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(item.path.resolve())
    return images


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--source", action="append", default=[], metavar="DIR[:TAG]",
        help="Source image dir (repeatable). Optional ':TAG' sets the label used "
        "in link names and priority (default: the directory name).",
    )
    parser.add_argument(
        "--exclude", action="append", default=[], type=Path, metavar="DIR",
        help="Leakage dir (repeatable): any duplicate group touching these is "
        "dropped whole (e.g. the tissue-probe test set).",
    )
    parser.add_argument(
        "--priority", nargs="*", default=[], metavar="TAG",
        help="Source tags in keep-preference order; the surviving copy of a "
        "duplicate group is taken from the earliest-listed source.",
    )
    parser.add_argument(
        "--ignore-dirname", action="append", default=[], metavar="NAME",
        help="Skip any file under a directory with this name (e.g. Invalid).",
    )
    parser.add_argument("--out", type=Path, required=True, help="Pool root; images land in <out>/images.")
    parser.add_argument("--hamming", type=int, default=6, help="Max Hamming distance (of 64 bits) for a train-vs-train duplicate.")
    parser.add_argument("--leak-hamming", type=int, default=2, help="Tight Hamming distance for keep-vs-test leakage (near-exact; 0 = byte-identical hash only).")
    parser.add_argument("--follow-symlinks", action="store_true", help="Include sym-linked files while scanning.")
    parser.add_argument("--dry-run", action="store_true", help="Report only; create no links.")
    return parser.parse_args(argv)


def _resolve_sources(specs: list[str], priority: list[str]) -> list[tuple[Path, str, int]]:
    """Turn '--source DIR[:TAG]' specs into (path, tag, priority-rank) tuples."""
    resolved: list[tuple[Path, str, int]] = []
    for spec in specs:
        if ":" in spec and not Path(spec).exists():
            raw, _, tag_override = spec.rpartition(":")
        else:
            raw, tag_override = spec, ""
        root = Path(raw)
        if not root.is_dir():
            raise SystemExit(f"source is not a directory: {root}")
        tag = _tag(root, tag_override or None)
        try:
            rank = priority.index(tag)
        except ValueError:
            rank = len(priority)
        resolved.append((root, tag, rank))
    return resolved


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.source:
        raise SystemExit("at least one --source is required")
    sources = _resolve_sources(args.source, args.priority)
    kept, report = build(
        sources,
        [Path(e) for e in args.exclude],
        hamming_radius=int(args.hamming),
        leak_hamming=int(args.leak_hamming),
        follow_symlinks=args.follow_symlinks,
        ignore_dirnames=set(args.ignore_dirname),
    )
    images = write_pool(kept, args.out, dry_run=args.dry_run)

    print(
        f"pool: scanned {report['scanned_keep']} keep + {report['scanned_exclude']} exclude "
        f"-> kept {report['kept']} "
        f"({report['dropped_near_duplicates']} near-dups across "
        f"{report['near_duplicate_groups']} groups, "
        f"{report['leakage_dropped']} leakage-excluded, "
        f"{report['unreadable']} unreadable)"
    )
    for tag, n in sorted(report["per_source_scanned"].items()):
        leaked = report["leakage_by_source"].get(tag, 0)
        print(f"    {tag:24s} scanned {n:5d}" + (f"  ({leaked} leaked-excluded)" if leaked else ""))
    print(("[dry-run] would link" if args.dry_run else "linked") + f" {len(kept)} images -> {images}")

    if not args.dry_run:
        report_path = args.out / "pool_report.json"
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
        print(f"report -> {report_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

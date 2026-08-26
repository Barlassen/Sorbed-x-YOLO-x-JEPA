#!/usr/bin/env python3
"""Scan wound datasets on disk and write leakage-free training manifests.

This CLI turns a folder of prepared (possibly gated) datasets into three manifest
files — ``train``, ``val``, ``test`` — split at the **patient** level so no
patient's visits straddle two splits. It never downloads anything and never
invents data: it consumes directories you populated by following
``training/DATA_README.md``.

Two ways to drive it:

1. **Single source, flags only** — point at one dataset and pick an adapter::

       python training/data_prep.py \
           --data-root /data/briefer/datasets/azh \
           --adapter azh_fuseg --source azh --license research-only \
           --out-dir /data/briefer/manifests/azh

2. **Multiple sources, config file** — declare every source once and merge them
   into one manifest set (the intended production path)::

       python training/data_prep.py \
           --config training/configs/datasets.example.yaml \
           --out-dir /data/briefer/manifests/combined

The written files are ``{split}.jsonl`` and ``{split}.csv`` for each split, plus a
``summary.json`` describing counts. Paths in the manifest are stored relative to
``--paths-relative-to`` when given (default: absolute), which keeps manifests
portable across machines that mount the data at different roots.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

# Allow running the file directly ("python training/data_prep.py") as well as
# importing it as "training.data_prep".
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from datasets import (
        ADAPTERS,
        SPLITS,
        DataPrepConfig,
        ManifestRecord,
        SourceSpec,
        SplitRatios,
        assert_no_patient_leakage,
        split_by_patient,
        write_manifest,
    )
else:  # pragma: no cover - exercised only when imported as a package
    from training.datasets import (
        ADAPTERS,
        SPLITS,
        DataPrepConfig,
        ManifestRecord,
        SourceSpec,
        SplitRatios,
        assert_no_patient_leakage,
        split_by_patient,
        write_manifest,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Scan wound datasets and write patient-level train/val/test manifests.",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=None,
        help="YAML config declaring one or more sources (see configs/datasets.example.yaml). "
        "When given, --data-root/--adapter/--source are ignored.",
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=None,
        help="Single-source mode: root directory of one prepared dataset.",
    )
    parser.add_argument(
        "--adapter",
        choices=sorted(ADAPTERS),
        default="generic",
        help="Single-source mode: layout adapter to use.",
    )
    parser.add_argument("--source", default="dataset", help="Single-source mode: source id.")
    parser.add_argument(
        "--license",
        default="unknown",
        help="Single-source mode: license/usage tag recorded in the manifest.",
    )
    parser.add_argument(
        "--body-part",
        default="unknown",
        help="Single-source mode: anatomical site tag (adapters may override).",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        required=True,
        help="Directory to write {split}.jsonl / {split}.csv and summary.json.",
    )
    parser.add_argument(
        "--paths-relative-to",
        type=Path,
        default=None,
        help="Store image/mask paths relative to this root (default: absolute paths).",
    )
    parser.add_argument("--train", type=float, default=0.7, help="Train fraction (flags mode).")
    parser.add_argument("--val", type=float, default=0.15, help="Val fraction (flags mode).")
    parser.add_argument("--test", type=float, default=0.15, help="Test fraction (flags mode).")
    parser.add_argument(
        "--stratify-by",
        choices=["none", "stage_label", "body_part"],
        default="none",
        help="Preserve class balance across splits by this column.",
    )
    parser.add_argument("--seed", type=int, default=1234, help="Deterministic split seed.")
    parser.add_argument(
        "--no-csv",
        action="store_true",
        help="Write only JSONL manifests (skip the sibling CSVs).",
    )
    return parser.parse_args(argv)


def _collect_records(
    args: argparse.Namespace,
) -> tuple[list[ManifestRecord], DataPrepConfig]:
    """Gather records either from a config file or single-source flags."""
    paths_root = args.paths_relative_to

    if args.config is not None:
        config = DataPrepConfig.load(args.config)
        records: list[ManifestRecord] = []
        for spec in config.sources:
            found = spec.scan(paths_root)
            print(f"  scanned {spec.name}: {len(found)} samples from {spec.root}")
            records.extend(found)
        return records, config

    if args.data_root is None:
        raise SystemExit("provide --config or --data-root (single-source mode)")

    stratify = None if args.stratify_by == "none" else args.stratify_by
    config = DataPrepConfig(
        sources=[
            SourceSpec(
                name=args.source,
                root=args.data_root,
                adapter=args.adapter,
                body_part=args.body_part,
                license=args.license,
            )
        ],
        ratios=SplitRatios(train=args.train, val=args.val, test=args.test),
        seed=args.seed,
        stratify_by=stratify,
    )
    records = config.sources[0].scan(paths_root)
    print(f"  scanned {args.source}: {len(records)} samples from {args.data_root}")
    return records, config


def _summarize(splits: dict[str, list[ManifestRecord]]) -> dict[str, object]:
    """Build a JSON-serializable summary of the split."""
    summary: dict[str, object] = {}
    for split in SPLITS:
        records = splits[split]
        patients = {r.patient_id for r in records}
        with_mask = sum(1 for r in records if r.mask_path)
        summary[split] = {
            "images": len(records),
            "patients": len(patients),
            "with_mask": with_mask,
            "by_source": dict(Counter(r.source for r in records)),
            "by_body_part": dict(Counter(r.body_part for r in records)),
            "by_stage_label": dict(Counter(r.stage_label or "(none)" for r in records)),
        }
    return summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    print("scanning sources...")
    records, config = _collect_records(args)
    if not records:
        raise SystemExit("no samples discovered; check your paths and DATA_README.md")

    splits = split_by_patient(
        records,
        ratios=config.ratios,
        seed=config.seed,
        stratify_by=config.stratify_by,
    )
    assert_no_patient_leakage(splits)

    out_dir: Path = args.out_dir.expanduser()
    out_dir.mkdir(parents=True, exist_ok=True)
    for split in SPLITS:
        written = write_manifest(
            splits[split],
            out_dir / f"{split}.jsonl",
            also_csv=not args.no_csv,
        )
        print(f"  wrote {split}: {len(splits[split])} rows -> {[p.name for p in written]}")

    summary = _summarize(splits)
    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, ensure_ascii=False)
    print(f"  wrote summary -> {summary_path.name}")

    total = sum(len(splits[s]) for s in SPLITS)
    print(
        f"done: {total} samples across "
        + ", ".join(f"{s}={len(splits[s])}" for s in SPLITS)
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

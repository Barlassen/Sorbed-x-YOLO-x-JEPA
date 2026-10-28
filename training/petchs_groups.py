"""Source-grouped folds for the Roboflow petchs pressure-ulcer set.

Roboflow made augmented copies (flips, rotations, crops) *before* splitting, so the same source photo appears in
train, valid and test under names like ``20009_jpg.rf.<hash>.jpg``; dHash misses rotated/cropped copies. Every
evaluation on petchs therefore groups images by source id and keeps all copies of a source in one fold.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

RF_PETCHS = Path("data/corpus/rf_petchs")
SPLITS = ("train", "valid", "test")
_SRC = re.compile(r"^(.*?)_(?:jpe?g|png)\.rf\.", re.I)
_STAGE_POLY = {"stage1": "1", "stage2": "2", "stage3": "3", "stage4": "4"}


def source(file_name: str) -> str:
    m = _SRC.match(file_name)
    return m.group(1) if m else file_name


def image_stage(category_names: set[str]) -> str | None:
    """Highest stage polygon (overlapping polygons are regions of one wound); U if unstageable and no stage 4."""
    stages = [_STAGE_POLY[n] for n in category_names if n in _STAGE_POLY]
    if "unstageable" in category_names and "4" not in stages:
        return "U"
    return max(stages) if stages else None


def stage_table() -> list[tuple[str, str, str | None]]:
    """(split, file name, stage) for every petchs image."""
    rows = []
    for split in SPLITS:
        j = json.loads((RF_PETCHS / split / "_annotations.coco.json").read_text())
        cats = {c["id"]: c["name"] for c in j["categories"]}
        names: dict = {}
        for a in j["annotations"]:
            names.setdefault(a["image_id"], set()).add(cats[a["category_id"]])
        rows += [(split, im["file_name"], image_stage(names.get(im["id"], set()))) for im in j["images"]]
    return rows


def source_folds(n: int = 3) -> dict[str, int]:
    """source id -> fold, stratified by the source's (most common) stage."""
    by: dict[str, Counter] = {}
    for _, f, st in stage_table():
        by.setdefault(source(f), Counter())[st or "-"] += 1
    order = sorted(by, key=lambda s: (by[s].most_common(1)[0][0], s))
    return {s: i % n for i, s in enumerate(order)}


def eval_representatives() -> set[str]:
    """One file per source (alphabetically first) -> the independent evaluation unit."""
    first: dict[str, str] = {}
    for _, f, st in sorted(stage_table(), key=lambda r: r[1]):
        if st is not None:
            first.setdefault(source(f), f)
    return set(first.values())

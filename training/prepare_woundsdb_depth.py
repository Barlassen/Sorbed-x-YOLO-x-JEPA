"""WoundsDB -> Ultralytics depth dataset (patient-level split) for fine-tuning YOLO26 depth.

WoundsDB ships 8-bit, relative depth rendered into the thermal frame (``thermal-stereo.png``,
brighter = nearer). YOLO26 depth predicts metres and trains with a scale-invariant log loss, so each
scene's target keeps the *measured shape* but is affinely fitted onto the base model's own metric
range for that photo: the network learns wound relief, not an absolute distance it cannot know.
Pixels without a measurement (0) stay 0 and are ignored by the loss.

Cases are split by patient (every ``--test-every``-th case -> test), so no patient crosses the split.

    python -m training.prepare_woundsdb_depth --out data/woundsdb_depth
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np

SRC = Path("data/corpus/woundsdb/results")


def case_id(scene: Path) -> int:
    return int(re.search(r"case_(\d+)", str(scene)).group(1))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(SRC))
    ap.add_argument("--out", default="data/woundsdb_depth")
    ap.add_argument("--target", default="thermal-stereo.png", help="aligned real-depth file used as target")
    ap.add_argument("--base", default="yolo26n-depth.pt", help="model whose metric range the targets adopt")
    ap.add_argument("--test-every", type=int, default=5)
    args = ap.parse_args()

    from ultralytics import YOLO

    base = YOLO(args.base)
    out = Path(args.out)
    scenes = sorted(p.parent for p in Path(args.src).glob("*/day_*/results/scene_*/thermal-photo.png"))
    cases = sorted({case_id(s) for s in scenes})
    test_cases = set(cases[args.test_every - 1 :: args.test_every])
    split_of = {c: "val" if c in test_cases else "train" for c in cases}
    stats = {"train": 0, "val": 0, "skipped": []}

    for s in scenes:
        photo = cv2.imread(str(s / "thermal-photo.png"))
        gt = cv2.imread(str(s / args.target), cv2.IMREAD_GRAYSCALE)
        name = f"c{case_id(s):02d}_{s.parts[-3]}_{s.parts[-1]}"
        if photo is None or gt is None:
            stats["skipped"].append(name)
            continue
        valid = (gt > 0) & (photo.sum(axis=2) > 0)
        h, w = gt.shape
        pred = base.predict(photo, verbose=False)[0].depth.data.cpu().numpy().astype(np.float32)
        pred = cv2.resize(pred, (w, h), interpolation=cv2.INTER_LINEAR)
        g = gt[valid].astype(np.float32)
        b, a = np.polyfit(g, pred[valid], 1)  # metres ~ a + b * gray
        if b >= 0:  # brighter must mean nearer; a positive slope means the fit is unusable
            stats["skipped"].append(name)
            continue
        target = np.zeros((h, w), np.float32)
        target[valid] = np.clip(a + b * g, 0.05, None)
        split = split_of[case_id(s)]
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "depth" / split).mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(out / "images" / split / f"{name}.png"), photo)
        cv2.imwrite(str(out / "depth" / split / f"{name}.png"), (target * 1000).round().astype(np.uint16))
        stats[split] += 1

    (out / "data.yaml").write_text(
        f"path: {out.resolve()}\ntrain: images/train\nval: images/val\nnc: 1\nnames:\n  0: depth\n"
        "channels: 3\ndepth_scale: 1000\n"
    )
    stats["test_cases"] = sorted(test_cases)
    (out / "split.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps({k: v for k, v in stats.items()}, indent=None))


if __name__ == "__main__":
    main()

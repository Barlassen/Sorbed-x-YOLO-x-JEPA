"""Second tissue ruler: Roboflow petchs pressure-ulcer *test* split (31 images, one annotator, never trained on).

Same methods and scoring as ``tissue_ab_compare`` (pooled Dice per tissue, image bootstrap), with two differences
fixed before any result was seen: (1) only pixels that carry a tissue polygon (granulation / slough / eschar) are
scored, because the stage outline around them is unlabeled and ambiguous; (2) there is a single annotator.
R1 = methods get the annotated wound region (stage outline + tissue polygons); R2 = methods get OUR segmenter's mask.

    python -m training.eval_tissue_petchs --cnn runs/tissue_cnn_P/seed0.pt ... --out runs/tissue_petchs/report.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from training.tissue_ab_compare import BG, G, MASK_RULES, SEG_NEW, SEG_OLD, wound_mask, boot, crop_resize, fmt, inter_sums, method_a, paste, pooled
from training.train_tissue_cnn import RF_PETCHS, coco_items, load, predict

HUE0 = 38  # A's only parameter, tuned on DFUTissue TrainVal (tissue_ab_compare); not re-tuned here


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cnn", nargs="*", default=[])
    ap.add_argument("--boot", type=int, default=2000)
    ap.add_argument("--out", default="runs/tissue_petchs/report.json")
    ap.add_argument("--mask-rule", choices=MASK_RULES, default="old-first")
    args = ap.parse_args()

    methods = {"A": lambda im, r: method_a(im, r, HUE0),
               "trivial(all G)": lambda im, r: np.where(r, G, BG).astype(np.uint8)}
    for ck in args.cnn:
        net = load(ck)
        methods[f"B-CNN {Path(ck).parent.name}/{Path(ck).stem}"] = lambda im, r, net=net: predict(net, im, r)

    from ultralytics import YOLO

    segs = (YOLO(SEG_OLD), YOLO(SEG_NEW))

    def our_mask(bgr):
        return wound_mask(segs, bgr, args.mask_rule)

    R = {s: {m: [] for m in methods} for s in ("R1", "R2")}
    n_img = 0
    for _, img, lab in coco_items(RF_PETCHS, "test"):
        scope = (lab >= 1) & (lab <= 3)  # tissue-annotated pixels only
        if scope.sum() < 30:
            continue
        n_img += 1
        gt = np.where(scope, lab, 0)
        for setting, region in (("R1", lab != 0), ("R2", our_mask(img))):
            for m, f in methods.items():
                if region.sum() < 30:
                    pred = np.zeros(img.shape[:2], np.uint8)
                else:
                    crop, reg, _, box = crop_resize(img, region)
                    pred = paste(f(crop, reg), box, img.shape[:2])
                R[setting][m].append(inter_sums(pred, gt, scope)[None])  # one annotator -> expert axis of 1

    report = {"images_scored": n_img}
    for setting in R:
        report[setting] = {m: fmt(pooled(np.array(v)), boot(np.array(v), args.boot)) for m, v in R[setting].items()}
        a = np.array(R[setting]["A"])
        for m in methods:
            if not m.startswith("B-CNN"):
                continue
            b = np.array(R[setting][m])
            rng = np.random.default_rng(1)
            diffs = []
            for _ in range(args.boot):
                idx = rng.integers(0, len(a), len(a))
                diffs.append(np.nanmean(pooled(a[idx])[:2]) - np.nanmean(pooled(b[idx])[:2]))
            diffs = np.array(diffs)
            report[f"A_minus_{m}_GS_{setting}"] = {
                "diff": round(float(np.nanmean(pooled(a)[:2]) - np.nanmean(pooled(b)[:2])), 3),
                "ci95": [round(float(x), 3) for x in np.percentile(diffs, [2.5, 97.5])],
                "P(A>B)": round(float((diffs > 0).mean()), 3)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(json.dumps(report, indent=1, ensure_ascii=False))


if __name__ == "__main__":
    main()

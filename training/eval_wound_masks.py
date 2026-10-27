"""Final one-shot wound-mask test of the mask v4 programme (pre-registered in HANDOFF, 2026-10-21).

Test sets (never used for training or selection): ComplexWoundDB 27 (expert consensus), petchs test (tissue polygons =
open surface, images with >=100 px), DFUTissue Test wound bed (classes 1-2). Success: CWDB >= 0.85 AND petchs-test
>= 0.75 (mean Dice) -> the model becomes the production mask.

    python -m training.eval_wound_masks --out runs/segment/mask_v4_final.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from training import tissue_ab_compare as T
from training.train_tissue_cnn import RF_PETCHS, coco_items
from training.train_wound_deeplab import load as load_deeplab, predict_mask

YOLO_V4 = "runs/segment/runs/segment/yolo_v4/weights/epoch18.pt"  # epoch 19, chosen on the pressure val set
YOLO_V4_CONF = 0.1  # inference rule chosen on the pressure val set (all parts)
DEEPLAB_V4 = "runs/wound_deeplab/v4.pt"
DFUT = Path("data/corpus/dfutissue/DFUTissue/Labeled/Original")


def dice(m, g) -> float:
    return float(2 * (m & g).sum() / (m.sum() + g.sum() + 1e-9))


def test_sets():
    for p in sorted(Path("data/ruler_complexwound/images").iterdir(), key=lambda q: int(q.stem)):
        yield "ComplexWoundDB (27)", cv2.imread(str(p)), cv2.imread(str(Path("data/ruler_complexwound/masks") / p.name), 0) > 0
    for _, img, lab in coco_items(RF_PETCHS, "test"):
        tis = (lab >= 1) & (lab <= 3)
        if tis.sum() >= 100:
            yield "petchs testi, açık yüzey", img, tis
    for p in sorted((DFUT / "Images/Test").glob("*.png")):
        a = cv2.imread(str(DFUT / "Annotations/Test" / p.name), 0)
        if a is not None and np.isin(a, (1, 2)).any():
            yield "DFUTissue testi, yara yatağı", cv2.imread(str(p)), np.isin(a, (1, 2))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/segment/mask_v4_final.json")
    args = ap.parse_args()
    from ultralytics import YOLO

    old, v1, v4 = YOLO(T.SEG_OLD), YOLO(T.SEG_NEW), YOLO(YOLO_V4)
    dl = load_deeplab(DEEPLAB_V4)

    def masks(img):
        mo, m1 = T._yolo_mask(old, img), T._yolo_mask(v1, img)
        return {"eski-önce (üretim)": mo if mo.any() else m1, "yalnız v1": m1,
                "YOLO-v4": T._yolo_mask(v4, img, conf=YOLO_V4_CONF), "DeepLab-v4": predict_mask(dl, img)}

    res: dict = {}
    for name, img, gt in test_sets():
        for k, m in masks(img).items():
            res.setdefault(name, {}).setdefault(k, []).append(
                (dice(m, gt), float((m & gt).sum() / (m.sum() + 1e-9)), float((m & gt).sum() / (gt.sum() + 1e-9)),
                 float(m.sum() / (gt.sum() + 1e-9))))
    out = {}
    for s, d in res.items():
        print(f"== {s} (n={len(next(iter(d.values())))})")
        for k, v in d.items():
            v = np.array(v)
            dm, dmed = v[:, 0].mean(), np.median(v[:, 0])
            print(f"   {k:20s} Dice ort {dm:.3f} medyan {dmed:.3f} <0.5: {(v[:, 0] < 0.5).sum()}/{len(v)}  "
                  f"isabet {np.median(v[:, 1]):.2f} kapsama {np.median(v[:, 2]):.2f} alan {np.median(v[:, 3]):.2f}x")
            out.setdefault(s, {})[k] = {"dice_mean": round(float(dm), 3), "dice_median": round(float(dmed), 3),
                                        "precision_med": round(float(np.median(v[:, 1])), 2),
                                        "recall_med": round(float(np.median(v[:, 2])), 2),
                                        "area_ratio_med": round(float(np.median(v[:, 3])), 2),
                                        "per_image": [round(float(x), 3) for x in v[:, 0]]}
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

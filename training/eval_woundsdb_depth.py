"""Does a depth model see the wound's real relief? Base vs WoundsDB-fine-tuned YOLO26 depth.

Same protocol as the 2026-09-11 depth check, restricted to the held-out patients of
``prepare_woundsdb_depth`` (``split.json``). Wound masks come from our segmenter (old -> new fallback).

(a) WoundsDB held-out: after removing body curvature (2nd-order surface fitted on a ring around the
    wound), does the model agree with the REAL sensor on whether the wound interior is a pit or a
    bump? 50% = coin flip; the real stereo sensor is internally consistent 92% of the time.
    Also the direction-free inside-vs-ring AUC and rank correlation with the sensor.
    ToF was never used for training, so it is an independent check.
(b) Pressure ruler (ComplexWoundDB 27, expert masks, no real depth): inside-vs-ring AUC and how many
    wounds come out as a pit (indirect: no ground-truth geometry exists there).

    python -m training.eval_woundsdb_depth --models yolo26n-depth.pt runs/.../last.pt
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import cv2
import numpy as np
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

SEG_OLD = "runs/segment/runs_wound/yolo26n_fuseg/weights/best.pt"
SEG_NEW = "runs/segment/runs/segment/combined_v1/weights/best.pt"


def gray(p: Path):
    a = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    return None if a is None else a.astype(np.float32)


def detrend(d, g, valid):
    """Residual after a quadratic surface fitted on the ring around the wound; + effect size."""
    k = max(6, int(0.25 * np.sqrt(g.sum())))
    ring = (cv2.dilate(g.astype(np.uint8), np.ones((2 * k + 1, 2 * k + 1), np.uint8)) > 0) & ~g
    gi, ri = g & valid, ring & valid
    if gi.sum() < 30 or ri.sum() < 60:
        return None
    h, w = d.shape
    s = max(h, w)
    ys, xs = np.nonzero(ri)
    X = lambda x, y: np.c_[np.ones_like(x), x, y, x * x, y * y, x * y]  # noqa: E731
    coef, *_ = np.linalg.lstsq(X(xs / s, ys / s), d[ri], rcond=None)
    yy, xx = np.mgrid[0:h, 0:w]
    res = d - (X(xx.ravel() / s, yy.ravel() / s) @ coef).reshape(h, w)
    a, b = res[gi], res[ri]
    auc = roc_auc_score(np.r_[np.ones(len(a)), np.zeros(len(b))], np.r_[a, b])
    return max(auc, 1 - auc), float((a.mean() - b.mean()) / (b.std() + 1e-9))


def srho(a, b, sel, seed=0):
    idx = np.flatnonzero(sel)
    if len(idx) < 200:
        return np.nan
    idx = np.random.default_rng(seed).choice(idx, min(20000, len(idx)), replace=False)
    return spearmanr(a.ravel()[idx], b.ravel()[idx]).correlation


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--split", default="data/woundsdb_depth/split.json")
    ap.add_argument("--src", default="data/corpus/woundsdb/results")
    ap.add_argument("--ruler", default="data/ruler_complexwound")
    ap.add_argument("--out", default="runs/depth/woundsdb_eval.json")
    args = ap.parse_args()

    from ultralytics import YOLO

    old, new = YOLO(SEG_OLD), YOLO(SEG_NEW)

    def wound_mask(bgr):
        for m in (old, new):
            r = m.predict(bgr, conf=0.01, verbose=False, retina_masks=True)[0]
            if r.masks is not None and len(r.masks.data):
                u = r.masks.data.cpu().numpy().max(0)
                if u.shape != bgr.shape[:2]:
                    u = cv2.resize(u.astype(np.float32), (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
                return u > 0.5
        return None

    test_cases = set(json.loads(Path(args.split).read_text())["test_cases"])
    scenes = [
        p.parent
        for p in sorted(Path(args.src).glob("*/day_*/results/scene_*/thermal-photo.png"))
        if int(re.search(r"case_(\d+)", str(p)).group(1)) in test_cases
    ]
    items = []  # (photo, mask, {sensor: (depth, effect)})
    for s in scenes:
        photo = cv2.imread(str(s / "thermal-photo.png"))
        if photo is None:
            continue
        cov = photo.sum(axis=2) > 0
        g = wound_mask(photo)
        if g is None or g.sum() < 30:
            continue
        g &= cov
        real = {}
        for name, f in (("stereo", "thermal-stereo.png"), ("tof", "thermal-depth.png")):
            d = gray(s / f)
            if d is None:
                continue
            o = detrend(d, g, cov & (d > 0))
            if o:
                real[name] = (d, o[1])
        items.append((photo, cov, g, real))

    ruler = Path(args.ruler)
    ruler_items = [
        (cv2.imread(str(ip)), cv2.imread(str(ruler / "masks" / ip.name), 0) > 0)
        for ip in sorted((ruler / "images").iterdir(), key=lambda p: int(p.stem))
    ]

    report = {"held_out_cases": sorted(test_cases), "scenes_with_mask": len(items)}
    for mp in args.models:
        model = YOLO(mp)

        def depth(bgr):
            h, w = bgr.shape[:2]
            d = model.predict(bgr, verbose=False)[0].depth.data.cpu().numpy().astype(np.float32)
            return cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)

        agree = {"stereo": [], "tof": []}
        rhos = {"stereo": [], "tof": []}
        aucs = []
        for i, (photo, cov, g, real) in enumerate(items):
            d = depth(photo)
            o = detrend(d, g, cov)
            if not o:
                continue
            aucs.append(o[0])
            for name, (dr, real_eff) in real.items():
                r = srho(d, dr, cov & (dr > 0), i)
                if np.isnan(r) or r == 0:
                    continue
                rhos[name].append(abs(r))
                # align the model's sign convention to the sensor's (metres vs brighter=nearer)
                agree[name].append(bool(np.sign(np.sign(r) * o[1]) == np.sign(real_eff)))
        r_auc, pits = [], 0
        for img, g in ruler_items:
            o = detrend(depth(img), g, np.ones(g.shape, bool))
            if o:
                r_auc.append(o[0])
                pits += o[1] > 0  # metres: interior farther than the ring = pit
        report[mp] = {
            "woundsdb_agree_pct": {k: round(100 * float(np.mean(v)), 1) for k, v in agree.items() if v},
            "woundsdb_agree_n": {k: f"{int(np.sum(v))}/{len(v)}" for k, v in agree.items() if v},
            "woundsdb_abs_rho_median": {k: round(float(np.median(v)), 3) for k, v in rhos.items() if v},
            "woundsdb_auc_median": round(float(np.median(aucs)), 3),
            "ruler_auc_median": round(float(np.median(r_auc)), 3),
            "ruler_pits": f"{pits}/{len(r_auc)}",
        }
        print(mp, json.dumps(report[mp], ensure_ascii=False))
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print("->", args.out)


if __name__ == "__main__":
    main()

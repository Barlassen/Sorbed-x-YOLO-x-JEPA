"""Layered wound map v1: DeepLab outline split into healing edge and open wound by the tissue CNN.

Layers (outer to inner): D = DeepLab wound mask (threshold 0.5); E = D where the tissue CNN says epithelial/other (O);
A = D where it says granulation, slough or eschar (open wound, with its tissue types). Pre-registered in HANDOFF
(2026-10-30):

* ``--cwdb``   M2: ComplexWoundDB 27 consensus Dice of old-first, D and A (DeepLab-v4 x final tissue CNN).
* ``--petchs`` M1 + M3: the 166 petchs sources, every image through the DeepLab and tissue CNN of its own source fold.
  M1 = open-surface Dice vs the tissue polygons; M3 = product rule engine with region D, ungated and gated.
* ``--viz``    layered overlays of the CWDB images into runs/audit/layered/ (patient photos: local only).

    python -m training.eval_layered_map --cwdb --viz
    python -m training.eval_layered_map --petchs
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch

from training.train_wound_deeplab import letterbox, load as load_deeplab, to_tensor

DEEPLAB_V4 = "runs/wound_deeplab/v4.pt"
TISSUE_FINAL = "runs/tissue_cnn_final/seed0.pt"
OUT = Path("runs/layered")
VIZ = Path("runs/audit/layered")
OPEN = (1, 2, 3)  # tissue_ab_compare ids: granulation, slough, eschar; 4 = epithelial/other
# BGR overlay colours: healing edge pink, granulation red, slough yellow, eschar black
COLOURS = {4: (199, 143, 255), 1: (27, 0, 224), 2: (15, 196, 241), 3: (34, 34, 34)}


@torch.no_grad()
def deeplab_prob(model, bgr: np.ndarray) -> np.ndarray:
    x, (nh, nw) = letterbox(bgr, cv2.INTER_AREA)
    prob = torch.softmax(model(to_tensor(x)[None].to(next(model.parameters()).device))["out"], 1)[0, 1, :nh, :nw]
    return cv2.resize(prob.cpu().numpy(), (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_LINEAR)


def layers(bgr: np.ndarray, deeplab, cnn) -> dict:
    """D/E/A masks, the tissue label map inside D, and the seg confidence (mean foreground probability in D)."""
    from training.tissue_ab_compare import crop_resize, paste
    from training.train_tissue_cnn import predict

    p = deeplab_prob(deeplab, bgr)
    d = p > 0.5
    lbl = np.zeros(d.shape, np.uint8)
    if d.sum() >= 30:
        crop, reg, _, box = crop_resize(bgr, d)
        lbl = paste(predict(cnn, crop, reg), box, d.shape)
        lbl[~d] = 0
    return {"D": d, "E": d & (lbl == 4), "A": d & np.isin(lbl, OPEN), "lbl": lbl,
            "conf": float(p[d].mean()) if d.any() else 0.0}


def dice(a: np.ndarray, b: np.ndarray) -> float:
    return float(2 * (a & b).sum() / (a.sum() + b.sum() + 1e-9))


def paired_ci(x: np.ndarray, y: np.ndarray, n: int = 2000, seed: int = 0) -> list[float]:
    """95% bootstrap CI of mean(x - y) over images (sources)."""
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(x), (n, len(x)))
    return [round(float(v), 3) for v in np.percentile((x - y)[idx].mean(1), [2.5, 97.5])]


def overlay(bgr: np.ndarray, lay: dict) -> np.ndarray:
    o = bgr.copy()
    for v, col in COLOURS.items():
        m = lay["D"] & (lay["lbl"] == v)
        o[m] = (0.45 * o[m] + 0.55 * np.array(col)).astype(np.uint8)
    t = max(2, bgr.shape[0] // 250)
    for m, col in ((lay["D"], (255, 255, 255)), (lay["A"], (0, 0, 255))):  # contour lines: outline, open wound
        cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        cv2.drawContours(o, cnts, -1, col, t)
    return o


def cwdb(viz: bool) -> dict:
    from ultralytics import YOLO

    from training.tissue_ab_compare import SEG_NEW, SEG_OLD, wound_mask
    from training.train_tissue_cnn import load as load_cnn

    dl, cnn, segs = load_deeplab(DEEPLAB_V4), load_cnn(TISSUE_FINAL), (YOLO(SEG_OLD), YOLO(SEG_NEW))
    rows = []
    src = Path("data/ruler_complexwound")
    if viz:
        VIZ.mkdir(parents=True, exist_ok=True)
    for p in sorted((src / "images").iterdir(), key=lambda q: int(q.stem)):
        img = cv2.imread(str(p))
        gt = cv2.imread(str(src / "masks" / p.name), 0) > 0
        lay = layers(img, dl, cnn)
        old = wound_mask(segs, img, "old-first")
        rows.append({"image": int(p.stem), "old_first": dice(old, gt), "D": dice(lay["D"], gt),
                     "A": dice(lay["A"], gt), "E_share_of_D": float(lay["E"].sum() / max(lay["D"].sum(), 1))})
        print(rows[-1], flush=True)
        if viz:
            gto = img.copy()
            cnts, _ = cv2.findContours(gt.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
            cv2.drawContours(gto, cnts, -1, (0, 200, 0), max(2, img.shape[0] // 250))
            h = 420

            def rs(a):
                return cv2.resize(a, (round(a.shape[1] * h / a.shape[0]), h), interpolation=cv2.INTER_AREA)

            cv2.imwrite(str(VIZ / f"cwdb_{int(p.stem):02d}.png"), np.hstack([rs(img), rs(gto), rs(overlay(img, lay))]))
    col = lambda k: np.array([r[k] for r in rows])  # noqa: E731
    return {"n": len(rows), "mean": {k: round(float(col(k).mean()), 3) for k in ("old_first", "D", "A")},
            "A_minus_D_ci": paired_ci(col("A"), col("D")),
            "A_minus_old_first_ci": paired_ci(col("A"), col("old_first")),
            "E_share_of_D_median": round(float(np.median(col("E_share_of_D"))), 3), "per_image": rows}


def petchs() -> dict:
    from ultralytics import YOLO

    from sorbed.preprocess.color_norm import gray_world_normalize
    from sorbed.staging.engine import StagingEngine
    from training.eval_staging_cv import bal, merge4
    from training.eval_staging_petchs import _OUT, features, items
    from training.petchs_groups import RF_PETCHS, SPLITS, eval_representatives, source, source_folds
    from training.tissue_ab_compare import SEG_NEW, SEG_OLD, wound_mask
    from training.train_tissue_cnn import coco_items, load as load_cnn

    folds, reps = source_folds(), eval_representatives()
    dls = [load_deeplab(f"runs/wound_deeplab/fold{k}.pt") for k in range(3)]
    cnns = [load_cnn(f"runs/tissue_cnn_fold/fold{k}.pt") for k in range(3)]
    segs = (YOLO(SEG_OLD), YOLO(SEG_NEW))
    engine = StagingEngine()
    product = {r["file"]: r for r in json.loads(Path("runs/staging/product_check.json").read_text())["per_image"]}
    tissue_gt = {}
    for split in SPLITS:
        for name, _, lab in coco_items(RF_PETCHS, split):
            if name in reps:
                tissue_gt[name] = (lab >= 1) & (lab <= 3)
    code = lambda s: _OUT.get(s, "belirsiz") if s is not None else "belirsiz"  # noqa: E731

    rows = []
    for split in SPLITS:
        for name, img, _, label in items(split):
            if name not in reps:
                continue
            k = folds[source(name)]
            lay = layers(img, dls[k], cnns[k])
            old = wound_mask(segs, img, "old-first")
            gt = tissue_gt[name]
            rgb = gray_world_normalize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
            fv = features(rgb, lay["D"], lay["lbl"]) if lay["D"].sum() >= 30 else None
            gated = engine.decide(fv.__class__(**{**fv.__dict__, "seg_confidence": lay["conf"]})) if fv else None
            row = {"file": name, "label": label, "fold": k, "has_tissue_gt": bool(gt.sum() >= 100),
                   "old_first": dice(old, gt), "D": dice(lay["D"], gt), "A": dice(lay["A"], gt),
                   "conf": round(lay["conf"], 3),
                   "rule_stage": code(engine.decide(fv).rule_stage) if fv else "belirsiz",
                   "shown_stage": code(gated.stage) if gated else "belirsiz",
                   "old_first_rule_stage": product[name]["product_rule_stage_conf1"],
                   "old_first_shown_stage": product[name]["product_stage"]}
            rows.append(row)
            print(f"{len(rows):3d} {name[:26]:26s} {label} D {row['D']:.2f} A {row['A']:.2f} "
                  f"old {row['old_first']:.2f} "
                  f"conf {row['conf']:.2f} kural {row['rule_stage']:8s} gösterilen {row['shown_stage']}", flush=True)

    m1 = [r for r in rows if r["has_tissue_gt"]]
    col = lambda rs, k: np.array([r[k] for r in rs])  # noqa: E731
    y = col(rows, "label")
    out = {"M1_open_surface": {
        "n": len(m1),
        "mean": {k: round(float(col(m1, k).mean()), 3) for k in ("old_first", "D", "A")},
        "missed_share_dice_lt_0.1": {k: round(float((col(m1, k) < 0.1).mean()), 3) for k in ("old_first", "D", "A")},
        "A_minus_old_first_ci": paired_ci(col(m1, "A"), col(m1, "old_first")),
        "D_minus_old_first_ci": paired_ci(col(m1, "D"), col(m1, "old_first")),
        "A_minus_D_ci": paired_ci(col(m1, "A"), col(m1, "D"))}}
    from scipy.stats import spearmanr

    out["M3_staging"] = {"n": len(rows)}
    for tag, rule, shown in (("deeplab_D", "rule_stage", "shown_stage"),
                             ("old_first", "old_first_rule_stage", "old_first_shown_stage")):
        s = col(rows, shown)
        sh = s != "belirsiz"
        out["M3_staging"][tag] = {
            "bal_acc_4_ungated": round(bal(merge4(y), merge4(col(rows, rule))), 3),
            "withheld": int((~sh).sum()),
            "bal_acc_4_shown_only": round(bal(merge4(y[sh]), merge4(s[sh])), 3) if sh.any() else None,
            "bal_acc_4_shown_as_wrong_when_withheld": round(bal(merge4(y), merge4(s)), 3),
            "shown_counts": {v: int((s == v).sum()) for v in sorted(set(s))}}
    out["M3_staging"]["deeplab_conf_vs_D_dice_spearman"] = round(float(spearmanr(col(m1, "conf"), col(m1, "D"))[0]), 3)
    out["per_image"] = rows
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cwdb", action="store_true")
    ap.add_argument("--petchs", action="store_true")
    ap.add_argument("--viz", action="store_true")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    for flag, fn, name in ((args.cwdb, lambda: cwdb(args.viz), "m2_cwdb.json"),
                           (args.petchs, petchs, "m1_m3_petchs.json")):
        if flag:
            res = fn()
            (OUT / name).write_text(json.dumps(res, ensure_ascii=False, indent=1))
            print(json.dumps({k: v for k, v in res.items() if k != "per_image"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

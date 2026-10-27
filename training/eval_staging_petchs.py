"""Staging step 1: the existing Sorbed rule engine, unchanged, on the petchs pressure-injury stage labels.

Pre-registered in HANDOFF (2026-10-28). Each image gets one stage: the highest stage polygon (multi-stage images are
overlapping regions of one wound), or U if an ``unstageable`` polygon is present. The engine's features are rebuilt
from (region x tissue source) so the error can be traced to the mask, the tissue classifier, or the rules:

* region: R1 = annotated outline (stage polygons + tissue polygons), R2 = our masks (old-first, DeepLab-v4)
* tissue: Sorbed colour model (production) or CNN-PS (G/S/N/O -> granulation/slough/eschar/epithelial)

    python -m training.eval_staging_petchs --out runs/staging/step1.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from sorbed.domain.enums import PressureInjuryStage as PS
from sorbed.domain.enums import TissueClass as TC
from sorbed.preprocess.color_norm import gray_world_normalize
from sorbed.staging.engine import StagingEngine
from sorbed.staging.features import FeatureVector
from sorbed.tissue.color_model import ColorTissueClassifier
from training.tissue_ab_compare import G, N, O, S, SEG_NEW, SEG_OLD, crop_resize, paste, wound_mask
from training.train_tissue_cnn import RF_PETCHS, petchs_test_dups
from training.train_tissue_cnn import load as load_cnn
from training.train_tissue_cnn import predict as cnn_predict
from training.train_wound_deeplab import load as load_deeplab
from training.train_wound_deeplab import predict_mask

CNN_PS = "runs/tissue_cnn_PS/seed0.pt"
DEEPLAB_V4 = "runs/wound_deeplab/v4.pt"
CLASSES = ("1", "2", "3", "4", "U")
FULL = {"3", "4", "U"}
_STAGE_POLY = {"stage1": "1", "stage2": "2", "stage3": "3", "stage4": "4"}
_TISSUE_POLY = {"granulation", "slough", "eschar"}
_OUT = {PS.STAGE_1: "1", PS.STAGE_2: "2", PS.STAGE_3: "3", PS.STAGE_4: "4", PS.UNSTAGEABLE: "U",
        PS.DEEP_TISSUE: "DTI"}
_COLOR = ColorTissueClassifier()
_ENGINE = StagingEngine()


def items(split: str, skip=frozenset()):
    """(file name, BGR image, annotated outline, stage label) for every petchs image with a stage polygon."""
    j = json.loads((RF_PETCHS / split / "_annotations.coco.json").read_text())
    cats = {c["id"]: c["name"] for c in j["categories"]}
    by: dict = {}
    for a in j["annotations"]:
        by.setdefault(a["image_id"], []).append(a)
    for im in j["images"]:
        if im["file_name"] in skip:
            continue
        anns = by.get(im["id"], [])
        names = {cats[a["category_id"]] for a in anns}
        stages = [_STAGE_POLY[n] for n in names if n in _STAGE_POLY]
        if "unstageable" in names and "4" not in stages:
            label = "U"
        elif stages:
            label = max(stages)
        else:
            continue
        img = cv2.imread(str(RF_PETCHS / split / im["file_name"]))
        outline = np.zeros(img.shape[:2], np.uint8)
        for a in anns:
            if cats[a["category_id"]] in _STAGE_POLY or cats[a["category_id"]] in _TISSUE_POLY | {"unstageable"}:
                for seg in a["segmentation"] if isinstance(a["segmentation"], list) else []:
                    pts = np.array(seg, np.float32).reshape(-1, 2).astype(np.int32)
                    if len(pts) >= 3:
                        cv2.fillPoly(outline, [pts], 1)
        yield im["file_name"], img, outline > 0, label


def features(rgb: np.ndarray, region: np.ndarray, cnn_lbl: np.ndarray | None = None) -> FeatureVector:
    """The engine's feature vector; tissue fractions from the colour model or, if given, the CNN label map."""
    t = _COLOR.classify(rgb, region)
    comp, open_bed = t.composition, t.open_bed_fraction
    if cnn_lbl is not None and region.any():
        lbl = np.where(region, cnn_lbl, 0)
        counts = {c: int((lbl == v).sum()) for c, v in
                  ((TC.GRANULATION, G), (TC.SLOUGH, S), (TC.ESCHAR, N), (TC.EPITHELIAL, O))}
        counts = {c: n for c, n in counts.items() if n}
        total = sum(counts.values())
        if total:
            comp = _COLOR._build_composition(counts, {c: float(n) for c, n in counts.items()}, total)
            open_bed = (total - counts.get(TC.EPITHELIAL, 0)) / total
    return FeatureVector(
        skin_intact=open_bed < 0.15, open_bed_fraction=open_bed, obscured_fraction=comp.obscured_fraction,
        granulation=comp.fraction_of(TC.GRANULATION), slough=comp.fraction_of(TC.SLOUGH),
        eschar=comp.fraction_of(TC.ESCHAR), epithelial=comp.fraction_of(TC.EPITHELIAL),
        adipose=comp.fraction_of(TC.ADIPOSE), muscle=comp.fraction_of(TC.MUSCLE),
        tendon_bone=comp.fraction_of(TC.TENDON_BONE), has_deep_structure=comp.has_deep_structure,
        maroon_purple=t.maroon_purple_fraction, erythema=t.erythema_fraction, depth_index=0.0, area_cm2=None,
        is_calibrated=False, seg_confidence=1.0, wound_fraction_of_image=float(region.mean()))


def stage(f: FeatureVector) -> str:
    rs = _ENGINE.decide(f).rule_stage
    return _OUT.get(rs, "belirsiz") if rs is not None else "belirsiz"


def score(y: np.ndarray, p: np.ndarray) -> tuple[float, float, float]:
    acc = float((y == p).mean())
    bal = float(np.mean([(p[y == c] == c).mean() for c in CLASSES if (y == c).any()]))
    graded = np.isin(p, CLASSES)
    binary = float((graded & (np.isin(y, list(FULL)) == np.isin(p, list(FULL)))).mean())
    return acc, bal, binary


def report(y: list[str], p: list[str], n_boot: int = 2000) -> dict:
    y, p = np.array(y), np.array(p)
    rng = np.random.default_rng(0)
    boots = np.array([score(y[i], p[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(n_boot))])
    lo, hi = np.percentile(boots, [2.5, 97.5], axis=0)
    cols = list(CLASSES) + ["DTI", "belirsiz"]
    return {"n": len(y),
            **{k: {"value": round(v, 3), "ci95": [round(float(a), 3), round(float(b), 3)]}
               for k, v, a, b in zip(("accuracy", "balanced_accuracy", "full_vs_partial"), score(y, p), lo, hi)},
            "ungraded_rate": round(float((~np.isin(p, CLASSES)).mean()), 3),
            "confusion": {c: {k: int(((y == c) & (p == k)).sum()) for k in cols} for c in CLASSES if (y == c).any()}}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/staging/step1.json")
    args = ap.parse_args()
    from ultralytics import YOLO

    segs = (YOLO(SEG_OLD), YOLO(SEG_NEW))
    dl, cnn = load_deeplab(DEEPLAB_V4), load_cnn(CNN_PS)
    dups = petchs_test_dups()

    def run(split, combos):
        rows = []
        for name, img, outline, label in items(split, dups if split != "test" else frozenset()):
            rgb = gray_world_normalize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
            regions = {"R1 GT bölge": outline}
            if any(c[0] == "R2 old-first" for c in combos):
                regions["R2 old-first"] = wound_mask(segs, img, "old-first")
            if any(c[0] == "R2 DeepLab-v4" for c in combos):
                regions["R2 DeepLab-v4"] = predict_mask(dl, img)
            row = {"file": name, "split": split, "label": label}
            for reg_name, tis in combos:
                region = regions[reg_name]
                cnn_lbl = None
                if tis == "CNN-PS" and region.sum() >= 30:
                    crop, reg, _, box = crop_resize(img, region)
                    cnn_lbl = paste(cnn_predict(cnn, crop, reg), box, img.shape[:2])
                row[f"{reg_name} × {tis}"] = stage(features(rgb, region, cnn_lbl))
            rows.append(row)
        return rows

    all_combos = [(r, t) for r in ("R1 GT bölge", "R2 old-first", "R2 DeepLab-v4") for t in ("renk", "CNN-PS")]
    ev = run("valid", all_combos) + run("test", all_combos)
    tr = run("train", [("R1 GT bölge", "renk"), ("R2 old-first", "renk")])

    out: dict = {"label_counts": {s: {c: sum(r["label"] == c for r in rows) for c in CLASSES}
                                  for s, rows in (("valid+test", ev), ("train", tr))}}
    for set_name, rows in (("valid+test", ev), ("test", [r for r in ev if r["split"] == "test"]), ("train", tr)):
        keys = [k for k in rows[0] if "×" in k]
        y = [r["label"] for r in rows]
        out[set_name] = {"hep Evre 3 (önemsiz taban)": report(y, ["3"] * len(y))}
        for k in keys:
            out[set_name][k] = report(y, [r[k] for r in rows])
    out["per_image"] = ev + tr
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    for set_name in ("valid+test", "test", "train"):
        print(f"== {set_name} {out['label_counts'].get(set_name, '')}")
        for k, v in out[set_name].items():
            print(f"  {k:32s} doğruluk {v['accuracy']['value']:.2f} {v['accuracy']['ci95']}  dengeli "
                  f"{v['balanced_accuracy']['value']:.2f} {v['balanced_accuracy']['ci95']}  tam/kısmi kat "
                  f"{v['full_vs_partial']['value']:.2f}  evresiz {v['ungraded_rate']:.2f}")


if __name__ == "__main__":
    main()

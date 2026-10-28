"""Staging step 2: arm A (corrected Sorbed rule engine, thresholds tuned per fold) vs arm B (stage CNN, from
``train_stage_cnn``) on source-grouped petchs folds. Pre-registered in HANDOFF (2026-10-28).

Evaluation unit: one image per source (n=166). Every prediction is out-of-fold: the tissue map comes from the fold's
tissue CNN, A's thresholds are grid-searched on the other two folds, the GBM head and B are trained on them.

    python -m training.eval_staging_cv --out runs/staging/step2.json
"""

from __future__ import annotations

import argparse
import dataclasses
import importlib.util
import itertools
import json
import subprocess
import tempfile
from pathlib import Path

import cv2
import numpy as np

from sorbed.domain.enums import PressureInjuryStage as PS
from sorbed.preprocess.color_norm import gray_world_normalize
from sorbed.staging.engine import StagingEngine
from sorbed.staging.features import FeatureVector
from sorbed.staging.ml_head import GBMStagingHead
from sorbed.staging.rules import Thresholds
from training.eval_staging_petchs import _OUT, CLASSES, FULL, features, items
from training.petchs_groups import SPLITS, eval_representatives, source, source_folds

DEFAULT = {"u": 0.5, "g": 0.15, "d": 0.1, "open": 0.15, "e": 0.15}
GRID = {"u": [round(x, 2) for x in np.arange(0.1, 0.85, 0.05)], "g": [0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4],
        "d": [0.01, 0.02, 0.05, 0.1, 0.15, 0.2], "open": [0.05, 0.1, 0.15, 0.2, 0.3], "e": [0.0, 0.05, 0.15]}
MAROON_MIN = 0.15
_CODE_TO_STAGE = {"1": PS.STAGE_1, "2": PS.STAGE_2, "3": PS.STAGE_3, "4": PS.STAGE_4, "U": PS.UNSTAGEABLE}


# ---------------------------------------------------------------- features (cached; slow part)
def extract(cache: Path) -> dict:
    if cache.exists():
        return json.loads(cache.read_text())
    from ultralytics import YOLO

    from training.tissue_ab_compare import SEG_NEW, SEG_OLD, crop_resize, paste, wound_mask
    from training.train_tissue_cnn import load, predict

    folds, reps = source_folds(), eval_representatives()
    cnns = [load(f"runs/tissue_cnn_fold/fold{k}.pt") for k in range(3)]
    segs = (YOLO(SEG_OLD), YOLO(SEG_NEW))
    out = {}
    for split in SPLITS:
        for name, img, outline, label in items(split):
            if name not in reps:
                continue
            k = folds[source(name)]
            rgb = gray_world_normalize(cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0)
            row = {"label": label, "fold": k}
            for reg_name, region in (("R1", outline), ("R2", wound_mask(segs, img, "old-first"))):
                lbl = None
                if region.sum() >= 30:
                    crop, reg, _, box = crop_resize(img, region)
                    lbl = paste(predict(cnns[k], crop, reg), box, img.shape[:2])
                row[reg_name] = dataclasses.asdict(features(rgb, region, lbl))
            out[name] = row
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out))
    return out


# ---------------------------------------------------------------- rule engine: fast replica + the real thing
def columns(rows: list[dict]) -> dict[str, np.ndarray]:
    return {k: np.array([r[k] for r in rows], float) for k in
            ("open_bed_fraction", "granulation", "obscured_fraction", "epithelial", "maroon_purple", "erythema",
             "wound_fraction_of_image")}


def rules_fast(X: dict, th: dict, gap: bool = True) -> np.ndarray:
    """Vectorised copy of src/sorbed/staging/rules.py (seg_confidence 1, no fat/deep-structure classes)."""
    n = len(X["open_bed_fraction"])
    out, prio = np.full(n, "belirsiz", object), np.zeros(n)

    def put(mask, label, pr):
        sel = mask & (pr > prio)
        out[sel], prio[sel] = label, pr

    intact = X["open_bed_fraction"] < th["open"]
    obs, gran = X["obscured_fraction"], X["granulation"]
    put(X["wound_fraction_of_image"] < 0.001, "belirsiz", 200)
    put(~intact & (obs >= th["u"]), "U", 150)
    put(intact & (X["maroon_purple"] >= MAROON_MIN), "DTI", 130)
    put(~intact & (obs < th["u"]) & ((gran >= th["g"]) | (gap & (obs >= th["d"]))), "3", 120)
    put(~intact & (X["epithelial"] > 0.05) & (gran < th["g"]) & (obs < th["d"]), "2", 100)
    put(intact & (X["maroon_purple"] < MAROON_MIN) & (X["erythema"] >= th["e"]), "1", 90)
    return out


def engine_stage(row: dict, th: dict) -> str:
    f = FeatureVector(**{**row, "skin_intact": row["open_bed_fraction"] < th["open"]})
    t = Thresholds(obscured_unstageable=th["u"], granulation_stage3=th["g"], devitalized_max_stage2=th["d"],
                   erythema_min=th["e"])
    rs = StagingEngine(t).decide(f).rule_stage
    return _OUT.get(rs, "belirsiz") if rs is not None else "belirsiz"


def old_engine_stages(rows: list[dict]) -> np.ndarray:
    """Step-1 rules (commit fadd04d, before the gap fix), evaluated exactly as the engine does."""
    src = subprocess.run(["git", "show", "fadd04d:src/sorbed/staging/rules.py"], capture_output=True, text=True,
                         check=True).stdout
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "rules_step1.py"
        path.write_text(src)
        spec = importlib.util.spec_from_file_location("rules_step1", path)
        old = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(old)
    out = []
    for r in rows:
        f = FeatureVector(**{**r, "skin_intact": r["open_bed_fraction"] < DEFAULT["open"]})
        fired = [rule for rule in old.RULES if rule.predicate(f, old.Thresholds())]
        rs = max(fired, key=lambda rule: rule.priority).implied_stage if fired else None
        out.append(_OUT.get(rs, "belirsiz") if rs is not None else "belirsiz")
    return np.array(out, object)


# ---------------------------------------------------------------- scoring
def merge4(a: np.ndarray) -> np.ndarray:
    return np.where(a == "4", "3", a)


def bal(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean([(p[y == c] == c).mean() for c in np.unique(y)]))


def scores(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    graded = np.isin(p, CLASSES)
    return {"bal_acc_4 (birincil)": bal(merge4(y), merge4(p)), "accuracy_5": float((y == p).mean()),
            "bal_acc_5": bal(y, p),
            "full_vs_partial": float((graded & (np.isin(y, list(FULL)) == np.isin(p, list(FULL)))).mean()),
            "ungraded": float((~graded).mean())}


def tune(X: dict, y: np.ndarray) -> dict:
    best, best_key = None, None
    for vals in itertools.product(*GRID.values()):
        th = dict(zip(GRID, vals))
        s = bal(merge4(y), merge4(rules_fast(X, th)))
        dist = sum(abs(th[k] - DEFAULT[k]) / DEFAULT[k] for k in DEFAULT)
        key = (s, -dist)
        if best_key is None or key > best_key:
            best, best_key = th, key
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/staging/step2.json")
    ap.add_argument("--features", default="runs/staging/cv_features.json")
    ap.add_argument("--boot", type=int, default=2000)
    args = ap.parse_args()

    data = extract(Path(args.features))
    names = sorted(data)
    y = np.array([data[n]["label"] for n in names], object)
    fold = np.array([data[n]["fold"] for n in names])
    rows = {r: [data[n][r] for n in names] for r in ("R1", "R2")}
    X = {r: columns(rows[r]) for r in rows}

    preds: dict[str, np.ndarray] = {"hep Evre 3": np.full(len(y), "3", object)}
    preds["adım-1 motoru (düzeltilmemiş) × old-first"] = old_engine_stages(rows["R2"])
    preds["A0: boşluk düzeltmesi, varsayılan eşik × old-first"] = rules_fast(X["R2"], DEFAULT)

    a_r2, a_r1, gbm = (np.full(len(y), "", object) for _ in range(3))
    fold_th = {}
    for k in range(3):
        tr, te = fold != k, fold == k
        th = tune({c: v[tr] for c, v in X["R2"].items()}, y[tr])
        fold_th[k] = th
        a_r2[te] = rules_fast({c: v[te] for c, v in X["R2"].items()}, th)
        a_r1[te] = rules_fast({c: v[te] for c, v in X["R1"].items()}, th)
        head = GBMStagingHead.train([FeatureVector(**rows["R2"][i]) for i in np.where(tr)[0]],
                                    [_CODE_TO_STAGE[c] for c in y[tr]], random_state=0)
        for i in np.where(te)[0]:
            gbm[i] = _OUT.get(head.predict(FeatureVector(**rows["R2"][i])).stage, "belirsiz")
        # the replica must agree with the real (corrected) engine
        for i in np.where(te)[0]:
            assert engine_stage(rows["R2"][i], th) == a_r2[i], (names[i], th)
    assert all(engine_stage(r, DEFAULT) == p for r, p in zip(rows["R2"], preds["A0: boşluk düzeltmesi, varsayılan eşik × old-first"]))
    assert (rules_fast(X["R2"], DEFAULT, gap=False) == preds["adım-1 motoru (düzeltilmemiş) × old-first"]).all()
    preds["A: düzeltilmiş + ayarlı eşik × old-first"] = a_r2
    preds["A eşikleri × R1 GT bölge (ayrıştırma)"] = a_r1
    preds["Sorbed GBM başlığı × old-first (ikincil)"] = gbm
    for init in ("coco", "tissue"):
        files = [Path(f"runs/staging/b_{init}_fold{k}.json") for k in range(3)]
        if all(f.exists() for f in files):
            b = {}
            for f in files:
                b.update({n: v["pred"] for n, v in json.loads(f.read_text())["pred"].items()})
            preds[f"B-{init}: doğrudan evre CNN'i"] = np.array([b[n] for n in names], object)

    rng = np.random.default_rng(0)
    idx = [rng.integers(0, len(y), len(y)) for _ in range(args.boot)]
    report: dict = {"n": len(y), "label_counts": {c: int((y == c).sum()) for c in CLASSES},
                    "fold_thresholds": fold_th, "final_thresholds_all_folds": tune(X["R2"], y), "methods": {}}
    for m, p in preds.items():
        s = scores(y, p)
        boots = {k: [scores(y[i], p[i])[k] for i in idx] for k in ("bal_acc_4 (birincil)", "accuracy_5")}
        report["methods"][m] = {k: round(v, 3) for k, v in s.items()}
        for k, v in boots.items():
            report["methods"][m][k + " ci95"] = [round(float(x), 3) for x in np.percentile(v, [2.5, 97.5])]
        cols = list(CLASSES) + ["DTI", "belirsiz"]
        report["methods"][m]["confusion"] = {c: {k: int(((y == c) & (p == k)).sum()) for k in cols} for c in CLASSES}

    def diff(a, b):
        d = [bal(merge4(y[i]), merge4(preds[a][i])) - bal(merge4(y[i]), merge4(preds[b][i])) for i in idx]
        return {"diff": round(bal(merge4(y), merge4(preds[a])) - bal(merge4(y), merge4(preds[b])), 3),
                "ci95": [round(float(x), 3) for x in np.percentile(d, [2.5, 97.5])]}

    A, OLD = "A: düzeltilmiş + ayarlı eşik × old-first", "adım-1 motoru (düzeltilmemiş) × old-first"
    report["A_minus_step1 (karar)"] = diff(A, OLD)
    for m in preds:
        if m.startswith("B-"):
            report[f"{m}_minus_A"] = diff(m, A)
    report["per_image"] = {n: {"label": str(y[i]), **{m: str(p[i]) for m, p in preds.items()}} for i, n in enumerate(names)}
    Path(args.out).write_text(json.dumps(report, ensure_ascii=False, indent=1))

    print(f"n={len(y)} {report['label_counts']}")
    print("katlara göre eşikler:", fold_th, "\ntüm OOF'ta eşik:", report["final_thresholds_all_folds"])
    for m, s in report["methods"].items():
        print(f"  {m:52s} 4-sınıf dengeli {s['bal_acc_4 (birincil)']:.2f} {s['bal_acc_4 (birincil) ci95']}  "
              f"5-sınıf doğruluk {s['accuracy_5']:.2f}  dengeli-5 {s['bal_acc_5']:.2f}  tam/kısmi "
              f"{s['full_vs_partial']:.2f}  evresiz {s['ungraded']:.2f}")
    for k, v in report.items():
        if "minus" in k:
            print(f"  {k}: {v}")


if __name__ == "__main__":
    main()

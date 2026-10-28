"""Does the product pipeline reproduce the staging step-2 evaluation pipeline?

Runs the real ``WoundAnalyzer`` (YOLO old-first mask + CNN tissue backend + current rule thresholds) on the 166
petchs source representatives and compares it with the cached evaluation features (``runs/staging/cv_features.json``,
R2 = old-first mask). Every image uses the tissue CNN of its own source fold, so the tissue model never saw it.

This is a fidelity check, not a new performance estimate: the rule thresholds in ``rules.py`` were tuned on all three
folds, so scores here are in-sample for the thresholds. The out-of-fold estimate is the one in HANDOFF (step 2).

    python -m training.check_product_pipeline --out runs/staging/product_check.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from sorbed.config.settings import Settings
from sorbed.domain.enums import TissueClass as TC
from sorbed.io.loader import load_image
from sorbed.pipeline.analyzer import WoundAnalyzer
from sorbed.staging.engine import StagingEngine
from sorbed.staging.features import FeatureVector, build_features
from training.eval_staging_cv import bal, merge4
from training.eval_staging_petchs import _OUT
from training.petchs_groups import RF_PETCHS, SPLITS, eval_representatives, source, source_folds
from training.tissue_ab_compare import SEG_NEW, SEG_OLD, wound_mask

TISSUES = {"granulation": TC.GRANULATION, "slough": TC.SLOUGH, "eschar": TC.ESCHAR, "epithelial": TC.EPITHELIAL}


def code(stage) -> str:
    return _OUT.get(stage, "belirsiz") if stage is not None else "belirsiz"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="runs/staging/product_check.json")
    args = ap.parse_args()
    from ultralytics import YOLO

    cache = json.loads(Path("runs/staging/cv_features.json").read_text())
    folds, reps = source_folds(), eval_representatives()
    analyzers = [WoundAnalyzer(Settings(
        segmentation_backend="yolo", yolo_seg_model=SEG_OLD, yolo_seg_fallback_model=SEG_NEW,
        yolo_seg_confidence=0.01, yolo_seg_keep_largest=False,
        tissue_backend="cnn", tissue_cnn_weights=f"runs/tissue_cnn_fold/fold{k}.pt")) for k in range(3)]
    segs = (YOLO(SEG_OLD), YOLO(SEG_NEW))
    engine = StagingEngine()

    rows = []
    for split in SPLITS:
        for path in sorted((RF_PETCHS / split).glob("*.jpg")):
            if path.name not in reps or path.name not in cache:
                continue
            k = folds[source(path.name)]
            analyzer = analyzers[k]
            image = load_image(path)
            bundle = analyzer.analyze(image)
            a = bundle.analysis
            seg_conf = analyzer._segmenter.segment(image).confidence
            ref_mask = wound_mask(segs, cv2.imread(str(path)), "old-first")
            inter = (bundle.wound_mask & ref_mask).sum()
            dice = 1.0 if not ref_mask.any() and not bundle.wound_mask.any() else \
                2 * inter / (bundle.wound_mask.sum() + ref_mask.sum())
            ref = cache[path.name]["R2"]
            got = {n: a.metrics.tissue.fraction_of(t) for n, t in TISSUES.items()}
            ref_rule = code(engine.decide(FeatureVector(**ref)).rule_stage)
            conf1 = code(engine.decide(build_features(a.metrics, 1.0)).rule_stage)
            rows.append({
                "file": path.name, "label": cache[path.name]["label"], "fold": k,
                "mask_dice_vs_eval": round(float(dice), 4),
                "max_tissue_diff": round(max(abs(got[n] - ref[n]) for n in TISSUES), 4),
                "seg_confidence": round(seg_conf, 3),
                "eval_rule_stage": ref_rule, "product_rule_stage_conf1": conf1,
                "product_rule_stage": code(a.decision.rule_stage), "product_stage": code(a.decision.stage),
                "abstained": a.decision.abstained,
            })
            print(f"{len(rows):3d} {path.name[:28]:28s} dice {dice:.3f} doku-fark {rows[-1]['max_tissue_diff']:.3f} "
                  f"eval {ref_rule:8s} ürün {rows[-1]['product_rule_stage']:8s} gösterilen {rows[-1]['product_stage']}",
                  flush=True)

    y = np.array([r["label"] for r in rows])
    col = lambda k: np.array([r[k] for r in rows])  # noqa: E731
    out = {
        "n": len(rows),
        "note": "fidelity check; thresholds in-sample (tuned on all folds) - not a performance estimate",
        "mask_dice_vs_eval_median": float(np.median(col("mask_dice_vs_eval"))),
        "mask_identical_share": float((col("mask_dice_vs_eval") > 0.999).mean()),
        "max_tissue_diff_median": float(np.median(col("max_tissue_diff"))),
        "max_tissue_diff_p95": float(np.percentile(col("max_tissue_diff"), 95)),
        "rule_stage_agreement_conf1": float((col("product_rule_stage_conf1") == col("eval_rule_stage")).mean()),
        "rule_stage_agreement": float((col("product_rule_stage") == col("eval_rule_stage")).mean()),
        "seg_confidence_below_0.2": int((col("seg_confidence") < 0.2).sum()),
        "abstained": int(col("abstained").sum()),
        "shown_stage_counts": {s: int((col("product_stage") == s).sum()) for s in sorted(set(col("product_stage")))},
        "bal_acc_4_in_sample": {k: round(bal(merge4(y), merge4(col(k))), 3) for k in
                                ("eval_rule_stage", "product_rule_stage_conf1", "product_rule_stage", "product_stage")},
        "per_image": rows,
    }
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, indent=1))
    print(json.dumps({k: v for k, v in out.items() if k != "per_image"}, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()

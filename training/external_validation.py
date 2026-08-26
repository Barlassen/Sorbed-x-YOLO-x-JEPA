#!/usr/bin/env python3
"""External pressure-injury validation: score against the human κ ceiling.

The single largest gap in a photo-based grading system is not architecture — it
is that a foot-ulcer-trained model has never been scored on real, off-domain
pressure injuries (sacral / ischial / heel), against a *consensus* of raters, and
read against the fact that humans themselves agree on PI staging only κ≈0.57
(Fulbrook 2023). A model κ of 0.7 means something very different at a 0.57 human
ceiling than at 0.95. This harness reports all of it together.

It consumes a predictions CSV (the model already ran; this never touches ONNX or
the pipeline) with columns::

    image, consensus, pred, body_part[, patient_id]
    [, rater_1, rater_2, ...]            # individual rater stages (for the ceiling)
    [, prob_<class> ...]                 # calibrated stage probabilities

and reports, on top of the standard grading metrics reused from
``training.evaluate``:

* the **human inter-rater ceiling** — mean pairwise quadratic-weighted κ across
  the rater columns — so every model number is read against it, not in a vacuum;
* **per-body-site subgroups** — foot vs sacral/ischial/heel — the domain-shift
  lens that a single pooled κ hides;
* **conformal coverage** — if a ``training.conformal`` calibrator JSON is passed,
  the empirical coverage, mean set size, and defer rate on this external set.

numpy/scipy only. Nothing is fabricated: missing rater columns simply omit the
ceiling; missing probabilities omit calibration and conformal sections.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from evaluate import (  # type: ignore[no-redef]
        DEFAULT_GRADE_CLASSES,
        _label_to_index,
        balanced_accuracy,
        evaluate_grading,
        quadratic_weighted_kappa,
    )
else:  # pragma: no cover - package import path
    from training.evaluate import (
        DEFAULT_GRADE_CLASSES,
        _label_to_index,
        balanced_accuracy,
        evaluate_grading,
        quadratic_weighted_kappa,
    )

# Body-site buckets: the training domain (foot) vs the pressure-injury-typical
# sites the model has never seen masks for. Anything unmatched -> "other".
_FOOT_SITES = frozenset({"foot", "feet", "toe", "heel_foot", "plantar"})
_PI_SITES = frozenset({"sacrum", "sacral", "ischium", "ischial", "heel", "coccyx",
                       "trochanter", "buttock", "hip"})


def interrater_ceiling(
    rater_matrix: np.ndarray, n_classes: int
) -> dict[str, float | int]:
    """Mean pairwise quadratic-weighted κ across raters (the human ceiling).

    ``rater_matrix`` is ``(n_samples, n_raters)`` of class indices. Returns the
    mean/min/max pairwise κ and the rater count. Undefined for < 2 raters.
    """
    n_raters = rater_matrix.shape[1]
    if n_raters < 2:
        raise ValueError("need >= 2 rater columns for an inter-rater ceiling")
    pairwise: list[float] = []
    for i in range(n_raters):
        for j in range(i + 1, n_raters):
            pairwise.append(
                quadratic_weighted_kappa(rater_matrix[:, i], rater_matrix[:, j], n_classes)
            )
    arr = np.asarray(pairwise, dtype=np.float64)
    return {
        "mean_pairwise_qwk": float(arr.mean()),
        "min_pairwise_qwk": float(arr.min()),
        "max_pairwise_qwk": float(arr.max()),
        "n_raters": int(n_raters),
        "n_pairs": int(arr.size),
    }


def _site_bucket(body_part: str) -> str:
    value = body_part.strip().lower()
    if value in _FOOT_SITES:
        return "foot"
    if value in _PI_SITES:
        return "pressure_injury_site"
    return "other"


def subgroup_metrics(
    y_true: np.ndarray, y_pred: np.ndarray, sites: Sequence[str], n_classes: int
) -> dict[str, dict[str, float | int]]:
    """Per-body-site κ and balanced accuracy — the domain-shift breakdown."""
    buckets: dict[str, list[int]] = {}
    for i, site in enumerate(sites):
        buckets.setdefault(_site_bucket(site), []).append(i)
    out: dict[str, dict[str, float | int]] = {}
    for bucket, idx in sorted(buckets.items()):
        sel = np.asarray(idx, dtype=np.int64)
        entry: dict[str, float | int] = {"n": int(sel.size)}
        # κ needs >= 2 samples and some label variation to be meaningful.
        if sel.size >= 2:
            entry["quadratic_weighted_kappa"] = quadratic_weighted_kappa(
                y_true[sel], y_pred[sel], n_classes)
            entry["balanced_accuracy"] = balanced_accuracy(
                y_true[sel], y_pred[sel], n_classes)
        out[bucket] = entry
    return out


def conformal_report(
    probs: np.ndarray, y_true: np.ndarray, calibrator_path: Path, classes: Sequence[str]
) -> dict[str, Any]:
    """Empirical coverage / mean set size / defer rate for a fitted calibrator."""
    if __package__ in (None, ""):
        from conformal import ConformalCalibrator  # type: ignore[no-redef]
    else:  # pragma: no cover
        from training.conformal import ConformalCalibrator
    calibrator = ConformalCalibrator.from_json(calibrator_path)
    if list(calibrator.class_names) != list(classes):
        raise SystemExit(
            f"calibrator classes {calibrator.class_names} != eval classes {list(classes)}"
        )
    sets = [calibrator.predict_set(probs[i]) for i in range(probs.shape[0])]
    covered = sum(int(y_true[i]) in sets[i] for i in range(probs.shape[0]))
    defers = sum(calibrator.should_defer(probs[i]) for i in range(probs.shape[0]))
    return {
        "target_coverage": 1.0 - calibrator.alpha,
        "empirical_coverage": covered / max(1, probs.shape[0]),
        "mean_set_size": float(np.mean([len(s) for s in sets])),
        "defer_rate": defers / max(1, probs.shape[0]),
        "calibration_size": calibrator.calibration_size,
    }


def load_external_csv(
    path: Path, classes: Sequence[str]
) -> dict[str, Any]:
    """Load the external predictions CSV into arrays + optional rater/site/probs."""
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        rows = list(reader)
    truth_col = "consensus" if "consensus" in fieldnames else "true"
    if truth_col not in fieldnames or "pred" not in fieldnames:
        raise SystemExit(f"{path} needs a '{truth_col}'/'consensus' and a 'pred' column")
    y_true = np.array([_label_to_index(r[truth_col], classes) for r in rows], dtype=np.int64)
    y_pred = np.array([_label_to_index(r["pred"], classes) for r in rows], dtype=np.int64)
    groups = np.array(
        [r.get("patient_id", "").strip() or str(i) for i, r in enumerate(rows)],
        dtype=object,
    )
    sites = [r.get("body_part", "").strip() for r in rows]

    rater_cols = sorted(c for c in fieldnames if c.startswith("rater_"))
    rater_matrix = None
    if len(rater_cols) >= 2:
        rater_matrix = np.array(
            [[_label_to_index(r[c], classes) for c in rater_cols] for r in rows],
            dtype=np.int64,
        )
    prob_cols = [f"prob_{c}" for c in classes]
    probs = None
    if all(c in fieldnames for c in prob_cols):
        probs = np.array([[float(r[c]) for c in prob_cols] for r in rows], dtype=np.float64)
        probs = probs / probs.sum(axis=1, keepdims=True).clip(min=1e-12)
    return {
        "y_true": y_true, "y_pred": y_pred, "groups": groups, "sites": sites,
        "rater_matrix": rater_matrix, "probs": probs, "n": len(rows),
    }


def evaluate_external(
    path: Path,
    *,
    classes: Sequence[str],
    calibrator_path: Path | None = None,
    n_boot: int = 1000,
    seed: int = 0,
) -> dict[str, Any]:
    """Full external report: grading metrics + ceiling + subgroups + conformal."""
    data = load_external_csv(path, classes)
    n_classes = len(classes)
    report: dict[str, Any] = {
        "n_samples": data["n"],
        "classes": list(classes),
        "grading": evaluate_grading(
            data["y_true"], data["y_pred"], groups=data["groups"], classes=classes,
            probs=data["probs"], n_boot=n_boot, seed=seed),
        "subgroups_by_site": subgroup_metrics(
            data["y_true"], data["y_pred"], data["sites"], n_classes),
    }
    if data["rater_matrix"] is not None:
        ceiling = interrater_ceiling(data["rater_matrix"], n_classes)
        report["human_ceiling"] = ceiling
        model_qwk = report["grading"]["quadratic_weighted_kappa"]["value"]
        report["model_vs_ceiling"] = {
            "model_qwk": model_qwk,
            "human_mean_pairwise_qwk": ceiling["mean_pairwise_qwk"],
            "model_reaches_ceiling": bool(model_qwk >= ceiling["mean_pairwise_qwk"]),
        }
    if calibrator_path is not None and data["probs"] is not None:
        report["conformal"] = conformal_report(
            data["probs"], data["y_true"], Path(calibrator_path), classes)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--csv", type=Path, required=True,
                        help="external predictions CSV (image,consensus,pred,body_part,...)")
    parser.add_argument("--classes", nargs="*", default=list(DEFAULT_GRADE_CLASSES),
                        help="class order (defaults to the NPIAP order)")
    parser.add_argument("--conformal", type=Path, default=None,
                        help="optional training.conformal calibrator JSON for coverage")
    parser.add_argument("--out", type=Path, default=None, help="write the report JSON here")
    parser.add_argument("--n-boot", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = evaluate_external(
        Path(args.csv), classes=args.classes, calibrator_path=args.conformal,
        n_boot=int(args.n_boot), seed=int(args.seed))
    if args.out is not None:
        with Path(args.out).expanduser().open("w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2, ensure_ascii=False)
    grade = report["grading"]["quadratic_weighted_kappa"]
    print(f"external validation: n={report['n_samples']}")
    print(f"  model QWK = {grade['value']:.4f} "
          f"[{grade['ci_low']:.4f}, {grade['ci_high']:.4f}]")
    if "human_ceiling" in report:
        ceil = report["human_ceiling"]
        print(f"  human ceiling (mean pairwise QWK, {ceil['n_raters']} raters) = "
              f"{ceil['mean_pairwise_qwk']:.4f}")
    for site, m in report["subgroups_by_site"].items():
        qwk = m.get("quadratic_weighted_kappa")
        qtxt = f"{qwk:.4f}" if isinstance(qwk, float) else "n/a"
        print(f"  [{site}] n={m['n']}  QWK={qtxt}")
    if "conformal" in report:
        c = report["conformal"]
        print(f"  conformal: coverage={c['empirical_coverage']:.3f} "
              f"(target {c['target_coverage']:.2f}), mean set={c['mean_set_size']:.2f}, "
              f"defer={c['defer_rate']:.3f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

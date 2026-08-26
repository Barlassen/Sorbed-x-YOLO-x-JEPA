"""External-validation harness: ceiling, subgroups, conformal coverage.

Skipped when numpy/scipy are absent (evaluate.py needs scipy). Exercises the
real CSV path on a tiny synthetic external set.
"""

from __future__ import annotations

import csv

import numpy as np
import pytest

pytest.importorskip("scipy")

from training.evaluate import DEFAULT_GRADE_CLASSES
from training.external_validation import (
    evaluate_external,
    interrater_ceiling,
    subgroup_metrics,
)

CLASSES = list(DEFAULT_GRADE_CLASSES)


def test_interrater_ceiling_perfect_agreement() -> None:
    # Three raters, identical labels -> κ = 1 everywhere.
    labels = np.array([0, 1, 2, 3, 0, 1])
    matrix = np.stack([labels, labels, labels], axis=1)
    ceiling = interrater_ceiling(matrix, n_classes=6)
    assert ceiling["n_raters"] == 3
    assert ceiling["n_pairs"] == 3
    assert ceiling["mean_pairwise_qwk"] == pytest.approx(1.0)


def test_subgroup_split_foot_vs_pi_site() -> None:
    y_true = np.array([0, 1, 2, 3])
    y_pred = np.array([0, 1, 2, 3])
    sites = ["foot", "foot", "sacrum", "heel"]
    groups = subgroup_metrics(y_true, y_pred, sites, n_classes=6)
    assert groups["foot"]["n"] == 2
    assert groups["pressure_injury_site"]["n"] == 2


def _write_external_csv(path, *, with_raters=True, with_probs=True) -> None:
    classes = CLASSES
    rows = []
    for i in range(40):
        true_idx = i % 4
        pred_idx = true_idx if i % 5 else (true_idx + 1) % 4  # mostly correct
        row = {
            "image": f"img_{i}.png",
            "consensus": classes[true_idx],
            "pred": classes[pred_idx],
            "body_part": "sacrum" if i % 2 else "foot",
            "patient_id": f"p{i}",
        }
        if with_raters:
            row["rater_1"] = classes[true_idx]
            row["rater_2"] = classes[true_idx if i % 3 else (true_idx + 1) % 4]
        if with_probs:
            p = np.full(len(classes), 0.02)
            p[pred_idx] = 0.9
            p = p / p.sum()
            for c, val in zip(classes, p, strict=True):
                row[f"prob_{c}"] = f"{val:.4f}"
        rows.append(row)
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def test_evaluate_external_end_to_end(tmp_path) -> None:
    csv_path = tmp_path / "ext.csv"
    _write_external_csv(csv_path)
    report = evaluate_external(csv_path, classes=CLASSES, n_boot=100, seed=0)
    assert report["n_samples"] == 40
    assert "quadratic_weighted_kappa" in report["grading"]
    assert "human_ceiling" in report
    assert report["human_ceiling"]["n_raters"] == 2
    assert set(report["subgroups_by_site"]) == {"foot", "pressure_injury_site"}
    assert "model_vs_ceiling" in report


def test_evaluate_external_with_conformal(tmp_path) -> None:
    from training.conformal import ConformalCalibrator

    csv_path = tmp_path / "ext.csv"
    _write_external_csv(csv_path)
    # Fit a calibrator on independent synthetic probabilities.
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(500, len(CLASSES)))
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    labels = np.array([rng.choice(len(CLASSES), p=probs[i]) for i in range(500)])
    calib = ConformalCalibrator.fit(
        probs, labels, alpha=0.1, class_names=CLASSES,
        defer_classes=["stage_3", "stage_4", "unstageable"])
    cpath = calib.to_json(tmp_path / "conformal.json")
    report = evaluate_external(csv_path, classes=CLASSES, calibrator_path=cpath, n_boot=100)
    assert "conformal" in report
    assert 0.0 <= report["conformal"]["empirical_coverage"] <= 1.0
    assert report["conformal"]["mean_set_size"] >= 1.0

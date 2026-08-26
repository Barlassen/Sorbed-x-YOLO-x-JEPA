"""Split-conformal (APS) prediction sets for the stage output.

Verifies the finite-sample marginal-coverage guarantee empirically, the
depth-abstention logic, and JSON round-trip between the training-side calibrator
and the deployed-path gate.
"""

from __future__ import annotations

import numpy as np
from training.conformal import (
    ConformalCalibrator,
    calibrate_threshold,
    prediction_set,
    spans_defer_classes,
)


def _synthetic_probs(n: int, num_classes: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    """A calibrated-ish classifier: labels drawn from the softmax it emits."""
    rng = np.random.default_rng(seed)
    logits = rng.normal(size=(n, num_classes))
    logits[np.arange(n), rng.integers(0, num_classes, size=n)] += 1.5  # some signal
    exp = np.exp(logits - logits.max(axis=1, keepdims=True))
    probs = exp / exp.sum(axis=1, keepdims=True)
    # Draw the true label from each row's own distribution -> well-calibrated.
    labels = np.array([rng.choice(num_classes, p=probs[i]) for i in range(n)])
    return probs, labels


def test_marginal_coverage_holds() -> None:
    probs, labels = _synthetic_probs(4000, 6, seed=0)
    cal_p, cal_y = probs[:2000], labels[:2000]
    test_p, test_y = probs[2000:], labels[2000:]
    names = [f"c{i}" for i in range(6)]
    alpha = 0.1
    calibrator = ConformalCalibrator.fit(cal_p, cal_y, alpha=alpha, class_names=names)
    coverage = calibrator.empirical_coverage(test_p, test_y)
    # Guarantee is >= 1 - alpha; allow a small finite-sample slack downward.
    assert coverage >= (1 - alpha) - 0.03


def test_lower_alpha_gives_larger_sets() -> None:
    probs, labels = _synthetic_probs(3000, 6, seed=1)
    names = [f"c{i}" for i in range(6)]
    strict = ConformalCalibrator.fit(probs, labels, alpha=0.01, class_names=names)
    loose = ConformalCalibrator.fit(probs, labels, alpha=0.2, class_names=names)
    strict_size = np.mean([len(strict.predict_set(p)) for p in probs])
    loose_size = np.mean([len(loose.predict_set(p)) for p in probs])
    assert strict_size >= loose_size


def test_prediction_set_never_empty() -> None:
    row = np.array([0.7, 0.2, 0.1])
    assert prediction_set(row, tau=0.0) == [0]  # argmax only
    assert prediction_set(row, tau=1.0) == [0, 1, 2]  # full set


def test_calibrate_threshold_bounds() -> None:
    scores = np.linspace(0, 1, 100)
    assert 0.0 <= calibrate_threshold(scores, 0.1) <= 1.0


def test_spans_defer_classes() -> None:
    defer = {3, 4, 5}
    assert spans_defer_classes([2, 3], defer) is True     # spans a defer class
    assert spans_defer_classes([0, 1, 2], defer) is False  # ambiguous but shallow
    assert spans_defer_classes([4], defer) is False        # singleton = confident


def test_json_roundtrip_to_runtime_gate(tmp_path) -> None:
    from sorbed.staging.conformal import ConformalStageGate

    probs, labels = _synthetic_probs(1500, 6, seed=2)
    names = ["evre1", "evre2", "evre3", "evre4", "unstageable", "dti"]
    calibrator = ConformalCalibrator.fit(
        probs, labels, alpha=0.1, class_names=names,
        defer_classes=["evre3", "evre4", "unstageable"])
    path = calibrator.to_json(tmp_path / "conformal.json")

    gate = ConformalStageGate.from_json(path)
    assert gate.tau == calibrator.tau
    # A confident row -> singleton -> no defer. A flat row -> wide set -> defer.
    confident = np.array([0.95, 0.02, 0.01, 0.01, 0.005, 0.005])
    flat = np.full(6, 1 / 6)
    assert gate.should_defer(confident) is False
    assert gate.should_defer(flat) is True
    # Training-side and runtime-side sets agree on the same row.
    assert set(gate.prediction_set(flat)) == set(calibrator.predict_named(flat))

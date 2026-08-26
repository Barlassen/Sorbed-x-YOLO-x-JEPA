#!/usr/bin/env python3
"""Split-conformal prediction sets for the pressure-injury stage.

A single softmax argmax hides what the model does not know. Conformal prediction
turns calibrated stage probabilities into a *set* of stages with a
distribution-free, finite-sample coverage guarantee: at level ``alpha`` the true
stage is in the set at least ``1 - alpha`` of the time, no matter the model or
the data distribution (Vovk; Romano, Sesia & Candès 2020, "APS"). That is
exactly the honesty a graded clinical output needs when human inter-rater
agreement itself is only κ≈0.57.

The method here is **Adaptive Prediction Sets (APS)**:

* nonconformity score of a labelled calibration example = the cumulative sum of
  class probabilities sorted high→low, up to and including the true class;
* the threshold ``tau`` is the ``ceil((n+1)(1-alpha))/n`` empirical quantile of
  those scores;
* the prediction set for a new example is the smallest set of top classes whose
  cumulative probability first reaches ``tau``.

Depth-driven abstention falls out of this for free: if the guaranteed-coverage
set spans stages that hinge on information a photo cannot see (Evre 3 vs Evre 4
vs Unstageable), the system should defer rather than commit — see
:func:`spans_defer_classes`.

numpy only (a Sorbed core dependency); no network, no learned parameters beyond
the single scalar ``tau``. Serialises to JSON so the deployed CPU path can load
it without any training stack.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def aps_scores(probs: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """APS nonconformity scores for labelled examples.

    ``probs`` is ``(n, C)`` (rows need not be perfectly normalised; they are
    renormalised defensively). ``labels`` is ``(n,)`` of true class indices. The
    score of row ``i`` is the cumulative probability mass of all classes at least
    as likely as the true class, i.e. how much of the sorted distribution you must
    accept before the truth is included. Lower is more confident-and-correct.
    """
    probs = np.asarray(probs, dtype=np.float64)
    labels = np.asarray(labels, dtype=np.int64)
    if probs.ndim != 2:
        raise ValueError("probs must be (n, C)")
    if labels.shape[0] != probs.shape[0]:
        raise ValueError("probs and labels disagree on n")
    probs = probs / probs.sum(axis=1, keepdims=True).clip(min=1e-12)
    order = np.argsort(-probs, axis=1)  # high -> low per row
    sorted_probs = np.take_along_axis(probs, order, axis=1)
    cumulative = np.cumsum(sorted_probs, axis=1)
    # Rank (position in the sorted order) of the true label per row.
    rank_of_true = np.argmax(order == labels[:, None], axis=1)
    return cumulative[np.arange(probs.shape[0]), rank_of_true]


def calibrate_threshold(scores: np.ndarray, alpha: float) -> float:
    """Finite-sample APS threshold ``tau`` from calibration scores.

    Uses the conformal quantile level ``ceil((n+1)(1-alpha))/n`` so the marginal
    coverage guarantee ``>= 1 - alpha`` holds for exchangeable data. Clamped to
    ``[0, 1]``; with too few points for the level, ``tau = 1`` (the full set,
    trivially covering).
    """
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    scores = np.asarray(scores, dtype=np.float64)
    n = scores.shape[0]
    if n == 0:
        raise ValueError("need at least one calibration score")
    level = np.ceil((n + 1) * (1.0 - alpha)) / n
    if level >= 1.0:
        return 1.0
    return float(np.clip(np.quantile(scores, level, method="higher"), 0.0, 1.0))


def prediction_set(prob_row: np.ndarray, tau: float) -> list[int]:
    """Smallest top-probability class set whose cumulative mass reaches ``tau``.

    Always returns at least the argmax class, so the set is never empty.
    """
    prob_row = np.asarray(prob_row, dtype=np.float64)
    prob_row = prob_row / prob_row.sum().clip(min=1e-12)
    order = np.argsort(-prob_row)
    cumulative = np.cumsum(prob_row[order])
    # Include classes up to the first index where cumulative >= tau.
    reached = int(np.searchsorted(cumulative, tau, side="left")) + 1
    reached = max(1, min(reached, prob_row.shape[0]))
    return sorted(int(order[i]) for i in range(reached))


def spans_defer_classes(pred_set: list[int], defer_indices: set[int]) -> bool:
    """True when a set covering ``>= 2`` classes includes a defer-class.

    A singleton set is a confident call and never defers. A larger set that
    reaches into the depth-ambiguous stages (Evre 3 / Evre 4 / Unstageable) is
    exactly the case a 2D photo cannot resolve, so the caller should abstain.
    """
    if len(pred_set) < 2:
        return False
    return bool(set(pred_set) & defer_indices)


@dataclass(frozen=True)
class ConformalCalibrator:
    """A fitted APS calibrator: one threshold plus its metadata."""

    tau: float
    alpha: float
    num_classes: int
    class_names: list[str]
    calibration_size: int
    defer_classes: list[str]

    @classmethod
    def fit(
        cls,
        probs: np.ndarray,
        labels: np.ndarray,
        *,
        alpha: float,
        class_names: list[str],
        defer_classes: list[str] | None = None,
    ) -> ConformalCalibrator:
        """Calibrate on held-out ``(probs, labels)`` at miscoverage ``alpha``."""
        probs = np.asarray(probs, dtype=np.float64)
        scores = aps_scores(probs, labels)
        tau = calibrate_threshold(scores, alpha)
        defer = defer_classes or []
        unknown = set(defer) - set(class_names)
        if unknown:
            raise ValueError(f"defer_classes not in class_names: {sorted(unknown)}")
        return cls(
            tau=tau, alpha=float(alpha), num_classes=probs.shape[1],
            class_names=list(class_names), calibration_size=int(probs.shape[0]),
            defer_classes=list(defer),
        )

    def predict_set(self, prob_row: np.ndarray) -> list[int]:
        """Prediction set (class indices) for one probability row."""
        return prediction_set(prob_row, self.tau)

    def predict_named(self, prob_row: np.ndarray) -> list[str]:
        """Prediction set as class names."""
        return [self.class_names[i] for i in self.predict_set(prob_row)]

    def should_defer(self, prob_row: np.ndarray) -> bool:
        """Abstain when the coverage set spans a depth-ambiguous stage."""
        defer_idx = {self.class_names.index(c) for c in self.defer_classes}
        return spans_defer_classes(self.predict_set(prob_row), defer_idx)

    def empirical_coverage(self, probs: np.ndarray, labels: np.ndarray) -> float:
        """Fraction of ``labels`` contained in their prediction sets (a check)."""
        probs = np.asarray(probs, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int64)
        hits = sum(
            int(labels[i]) in self.predict_set(probs[i]) for i in range(probs.shape[0])
        )
        return hits / max(1, probs.shape[0])

    def to_json(self, path: Path) -> Path:
        path = Path(path).expanduser()
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "method": "aps",
            "tau": self.tau,
            "alpha": self.alpha,
            "num_classes": self.num_classes,
            "class_names": self.class_names,
            "calibration_size": self.calibration_size,
            "defer_classes": self.defer_classes,
        }
        with path.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False)
        return path

    @classmethod
    def from_json(cls, path: Path) -> ConformalCalibrator:
        with Path(path).expanduser().open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls(
            tau=float(payload["tau"]), alpha=float(payload["alpha"]),
            num_classes=int(payload["num_classes"]),
            class_names=list(payload["class_names"]),
            calibration_size=int(payload.get("calibration_size", 0)),
            defer_classes=list(payload.get("defer_classes", [])),
        )

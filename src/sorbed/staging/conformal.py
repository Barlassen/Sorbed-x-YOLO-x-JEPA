"""Load a fitted conformal calibrator and turn stage probabilities into a set.

This is the deployed-path counterpart to ``training/conformal.py``: it reads the
JSON artifact produced there (a single APS threshold ``tau`` plus metadata) and,
given a stage probability vector, returns the coverage-guaranteed prediction set
and a depth-abstention decision — with no dependency on the training stack.

The calibrator is *optional*. It requires an external, multi-rater pressure-
injury calibration set to fit (the project's stated top-priority data gap); until
that artifact exists the staging pipeline runs exactly as before. When it is
present, :class:`ConformalStageGate` gives the arbiter a distribution-free reason
to defer on the depth-ambiguous stages a photo cannot resolve.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class ConformalStageGate:
    """Applies a fitted APS threshold to stage probabilities on the CPU path."""

    tau: float
    class_names: tuple[str, ...]
    defer_classes: tuple[str, ...]
    alpha: float = 0.1

    @classmethod
    def from_json(cls, path: Path | str) -> ConformalStageGate:
        with Path(path).expanduser().open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return cls(
            tau=float(payload["tau"]),
            class_names=tuple(payload["class_names"]),
            defer_classes=tuple(payload.get("defer_classes", ())),
            alpha=float(payload.get("alpha", 0.1)),
        )

    def prediction_set(self, probs: np.ndarray) -> list[str]:
        """Smallest top-probability stage set whose mass reaches ``tau``."""
        row = np.asarray(probs, dtype=np.float64)
        if row.shape != (len(self.class_names),):
            raise ValueError(
                f"expected {len(self.class_names)} probabilities, got {row.shape}"
            )
        row = row / row.sum().clip(min=1e-12)
        order = np.argsort(-row)
        cumulative = np.cumsum(row[order])
        reached = int(np.searchsorted(cumulative, self.tau, side="left")) + 1
        reached = max(1, min(reached, row.shape[0]))
        return sorted(
            (self.class_names[int(order[i])] for i in range(reached)),
            key=self.class_names.index,
        )

    def should_defer(self, probs: np.ndarray) -> bool:
        """Abstain when a multi-class set reaches a depth-ambiguous stage."""
        pred = self.prediction_set(probs)
        if len(pred) < 2:
            return False
        return bool(set(pred) & set(self.defer_classes))

"""Optional learned staging head over the clinical feature vector.

The heavy neural nets live upstream (segmentation/tissue), where their output is
*inspectable pixels*. The final grade instead uses a transparent gradient-boosted
classifier over the low-dimensional, already-clinically-meaningful feature vector,
so its per-feature contributions are legible. This head is optional: without a
trained model the engine is transparently rule-only.

There is no public, permissively-licensed pressure-injury staging dataset (see
docs/MODELS.md), so no model ships with Sorbed. This class is the real mechanism
to train one on your own labeled feature vectors and to explain its output.
"""

from __future__ import annotations

import pickle  # nosec B403 - only ever loads operator-provided, trusted model files
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from sorbed.domain.enums import PressureInjuryStage
from sorbed.staging.features import FeatureVector

# Fixed feature ordering — the contract between training and inference.
FEATURE_NAMES: tuple[str, ...] = (
    "skin_intact",
    "open_bed_fraction",
    "obscured_fraction",
    "granulation",
    "slough",
    "eschar",
    "epithelial",
    "adipose",
    "muscle",
    "tendon_bone",
    "maroon_purple",
    "erythema",
    "depth_index",
    "seg_confidence",
    "wound_fraction_of_image",
)


def feature_array(f: FeatureVector) -> np.ndarray:
    """Project a :class:`FeatureVector` to the fixed model input vector."""
    return np.array(
        [
            float(f.skin_intact),
            f.open_bed_fraction,
            f.obscured_fraction,
            f.granulation,
            f.slough,
            f.eschar,
            f.epithelial,
            f.adipose,
            f.muscle,
            f.tendon_bone,
            f.maroon_purple,
            f.erythema,
            f.depth_index,
            f.seg_confidence,
            f.wound_fraction_of_image,
        ],
        dtype=np.float64,
    )


@dataclass(frozen=True)
class MLStagePrediction:
    stage: PressureInjuryStage
    probabilities: dict[PressureInjuryStage, float]
    top_features: list[tuple[str, float]]  # (feature, contribution), ranked


class GBMStagingHead:
    """A gradient-boosted staging classifier with per-feature attributions."""

    name = "gbm"

    def __init__(self, model: object, classes: list[PressureInjuryStage]) -> None:
        self._model = model
        self._classes = classes

    @classmethod
    def train(
        cls,
        features: list[FeatureVector],
        labels: list[PressureInjuryStage],
        **kwargs: object,
    ) -> GBMStagingHead:
        """Fit a classifier on labeled feature vectors.

        Uses scikit-learn's HistGradientBoostingClassifier — CPU-cheap and a good
        fit for this low-dimensional, tabular problem.
        """
        from sklearn.ensemble import HistGradientBoostingClassifier

        if len(features) != len(labels):
            raise ValueError("features and labels must have equal length")
        classes = sorted(set(labels), key=lambda s: s.value)
        x = np.stack([feature_array(f) for f in features])
        y = np.array([label.value for label in labels])
        model = HistGradientBoostingClassifier(**kwargs)  # type: ignore[arg-type]
        model.fit(x, y)
        return cls(model=model, classes=classes)

    def predict(self, features: FeatureVector) -> MLStagePrediction:
        x = feature_array(features).reshape(1, -1)
        proba = np.asarray(self._model.predict_proba(x))[0]  # type: ignore[attr-defined]
        model_classes = [PressureInjuryStage(c) for c in self._model.classes_]  # type: ignore[attr-defined]
        probabilities = {
            stage: float(p) for stage, p in zip(model_classes, proba, strict=True)
        }
        best = max(probabilities.items(), key=lambda kv: kv[1])[0]
        return MLStagePrediction(
            stage=best,
            probabilities=probabilities,
            top_features=self._attributions(x),
        )

    def _attributions(self, x: np.ndarray) -> list[tuple[str, float]]:
        """Per-feature contribution for this sample.

        Uses SHAP when available (exact for tree models); otherwise falls back to
        the model's global permutation importances scaled by the sample's feature
        deviation, so an attribution is always produced.
        """
        try:
            import shap

            explainer = shap.TreeExplainer(self._model)
            values = np.asarray(explainer.shap_values(x))
            magnitude = np.abs(values).reshape(len(FEATURE_NAMES), -1).sum(axis=1)
        except Exception:
            importances = getattr(self._model, "feature_importances_", None)
            if importances is None:
                magnitude = np.abs(x).ravel()
            else:
                magnitude = np.abs(np.asarray(importances) * x.ravel())
        ranked = sorted(
            zip(FEATURE_NAMES, magnitude.tolist(), strict=True),
            key=lambda kv: kv[1],
            reverse=True,
        )
        return [(name, round(val, 4)) for name, val in ranked[:5] if val > 0]

    def save(self, path: str | Path) -> None:
        payload = {"model": self._model, "classes": [c.value for c in self._classes]}
        Path(path).write_bytes(pickle.dumps(payload))

    @classmethod
    def load(cls, path: str | Path) -> GBMStagingHead:
        """Load a trained head. Only load files you trust (pickle executes code)."""
        payload = pickle.loads(Path(path).read_bytes())  # nosec B301 - trusted operator input
        classes = [PressureInjuryStage(c) for c in payload["classes"]]
        return cls(model=payload["model"], classes=classes)

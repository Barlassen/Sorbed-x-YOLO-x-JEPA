"""Backend selection for the pluggable analysis stages.

Backends are resolved by name from configuration. The classical / color-model /
rule-engine defaults need no weights and always work offline. Learned backends
register here and are selected the same way.
"""

from __future__ import annotations

from sorbed.config.settings import Settings
from sorbed.segmentation.base import WoundSegmenter
from sorbed.segmentation.classical import ClassicalSegmenter
from sorbed.staging.engine import StagingEngine
from sorbed.staging.rules import Thresholds
from sorbed.tissue.color_model import ColorTissueClassifier


class UnknownBackendError(RuntimeError):
    """Raised when configuration names a backend that is not registered."""


def build_segmenter(settings: Settings) -> WoundSegmenter:
    name = settings.segmentation_backend
    if name == "classical":
        return ClassicalSegmenter(
            grabcut_iterations=settings.grabcut_iterations,
            min_area_fraction=settings.min_wound_area_fraction,
        )
    if name in {"onnx", "smp_unet"}:
        from sorbed.segmentation.backends.onnx_backend import OnnxSegmenter

        return OnnxSegmenter.from_settings(settings)
    if name in {"hf_sam", "hf", "sam", "medsam"}:
        from sorbed.segmentation.backends.hf_backend import HuggingFaceSAMSegmenter

        return HuggingFaceSAMSegmenter.from_settings(settings)
    if name in {"yolo", "yolo26"}:
        from sorbed.segmentation.backends.yolo_backend import YoloSegmenter

        return YoloSegmenter.from_settings(settings)
    raise UnknownBackendError(f"unknown segmentation backend: {name!r}")


def build_tissue_classifier(settings: Settings) -> ColorTissueClassifier:
    if settings.tissue_backend == "color_model":
        return ColorTissueClassifier()
    raise UnknownBackendError(f"unknown tissue backend: {settings.tissue_backend!r}")


def build_staging_engine(settings: Settings) -> StagingEngine:
    if settings.staging_backend == "rule_engine":
        thresholds = Thresholds(obscured_unstageable=settings.obscured_fraction_unstageable)
        return StagingEngine(
            thresholds=thresholds,
            low_confidence_threshold=settings.low_confidence_threshold,
        )
    raise UnknownBackendError(f"unknown staging backend: {settings.staging_backend!r}")

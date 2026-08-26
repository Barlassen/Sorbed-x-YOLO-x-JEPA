"""Ultralytics YOLO26 wound-segmentation backend.

This backend runs a YOLO26 instance-segmentation model (``yolo26*-seg.pt``) via
the ``ultralytics`` package and turns its detected instance masks into a single
boolean wound mask plus a detection-derived confidence. It implements the same
:class:`~sorbed.segmentation.base.WoundSegmenter` protocol as the classical and
ONNX backends, so it is swappable via configuration
(``SORBED_SEGMENTATION_BACKEND=yolo``).

The pipeline in :func:`sorbed.pipeline.backends.build_segmenter` instantiates this
class via :meth:`YoloSegmenter.from_settings`. The model id/path is read from the
``SORBED_YOLO_SEG_MODEL`` environment variable and defaults to ``yolo26n-seg.pt``,
which ``ultralytics`` downloads on first use.

Prototype note: unlike the ONNX backend, weights are fetched by ``ultralytics``
rather than sha256-verified through Sorbed's registry. This is a research seam,
not the governed production path — see docs/MODELS.md on provenance.
"""

from __future__ import annotations

import os
from functools import lru_cache

import cv2
import numpy as np
from scipy import ndimage

from sorbed.config.settings import Settings
from sorbed.imaging import RasterImage
from sorbed.segmentation.base import SegmentationResult

_BACKEND_NAME = "yolo"
_ENV_MODEL = "SORBED_YOLO_SEG_MODEL"
_DEFAULT_MODEL = "yolo26n-seg.pt"


@lru_cache(maxsize=4)
def _load_model(model_id: str):
    """Load (and cache) an Ultralytics YOLO model by id/path.

    ``ultralytics`` is imported lazily so the weight-free classical core never
    depends on torch being installed.
    """
    from ultralytics import YOLO

    return YOLO(model_id)


class YoloSegmenter:
    """Run a YOLO26 instance-segmentation model and return a boolean wound mask."""

    name = _BACKEND_NAME

    def __init__(
        self,
        model_id: str = _DEFAULT_MODEL,
        *,
        min_area_fraction: float = 0.0015,
        confidence_floor: float = 0.25,
    ) -> None:
        self._model_id = model_id
        self._min_area_fraction = float(min_area_fraction)
        self._confidence_floor = float(confidence_floor)

    @classmethod
    def from_settings(cls, settings: Settings) -> YoloSegmenter:
        model_id = os.environ.get(_ENV_MODEL, "").strip() or _DEFAULT_MODEL
        return cls(
            model_id,
            min_area_fraction=settings.min_wound_area_fraction,
        )

    def segment(self, image: RasterImage) -> SegmentationResult:
        orig_h, orig_w = image.shape_hw
        model = _load_model(self._model_id)

        # Ultralytics expects an 8-bit BGR array when handed a numpy image.
        bgr = cv2.cvtColor(image.to_uint8_rgb(), cv2.COLOR_RGB2BGR)
        result = model.predict(bgr, conf=self._confidence_floor, verbose=False)[0]

        mask, confidence = self._combine_instances(result, orig_h, orig_w)
        mask = mask & self._valid_mask(image)
        mask = self._largest_component(mask)

        return SegmentationResult(
            wound_mask=mask.astype(bool),
            confidence=float(confidence),
            backend=_BACKEND_NAME,
            weights_sha256=None,
        )

    def _combine_instances(
        self, result: object, orig_h: int, orig_w: int
    ) -> tuple[np.ndarray, float]:
        """Union all detected instance masks into one wound mask.

        The wound is a single region, but a detector may return several instances
        (or none). We take their union, resize to the original resolution, and
        derive confidence from the mean detection score — never a constant.
        """
        empty = np.zeros((orig_h, orig_w), dtype=bool)
        masks = getattr(result, "masks", None)
        if masks is None or masks.data is None or len(masks.data) == 0:
            return empty, 0.0

        # (N, h, w) float in [0,1] at the model's mask resolution.
        data = masks.data.cpu().numpy()
        union = data.max(axis=0)  # (h, w)
        union = cv2.resize(
            union.astype(np.float32), (orig_w, orig_h), interpolation=cv2.INTER_LINEAR
        )
        wound = union >= 0.5

        confidence = 0.0
        boxes = getattr(result, "boxes", None)
        if boxes is not None and boxes.conf is not None and len(boxes.conf) > 0:
            confidence = float(boxes.conf.cpu().numpy().mean())
        return wound, max(0.0, min(1.0, confidence))

    @staticmethod
    def _valid_mask(image: RasterImage) -> np.ndarray:
        """Opaque pixels are eligible to be wound; transparent ones are excluded."""
        if image.alpha is None:
            return np.ones(image.shape_hw, dtype=bool)
        return image.alpha > 0.5

    def _largest_component(self, mask: np.ndarray) -> np.ndarray:
        if not mask.any():
            return mask
        if mask.sum() < self._min_area_fraction * mask.size:
            return mask
        labeled, count = ndimage.label(mask)
        if count <= 1:
            return mask
        sizes = ndimage.sum(np.ones_like(labeled), labeled, index=range(1, count + 1))
        keep = int(np.argmax(sizes)) + 1
        return labeled == keep

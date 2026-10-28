"""Ultralytics YOLO26 wound-segmentation backend.

This backend runs a YOLO26 instance-segmentation model (``yolo26*-seg.pt``) via
the ``ultralytics`` package and turns its detected instance masks into a single
boolean wound mask plus a detection-derived confidence. It implements the same
:class:`~sorbed.segmentation.base.WoundSegmenter` protocol as the classical and
ONNX backends, so it is swappable via configuration
(``SORBED_SEGMENTATION_BACKEND=yolo``).

The pipeline in :func:`sorbed.pipeline.backends.build_segmenter` instantiates this
class via :meth:`YoloSegmenter.from_settings`. The model id/path comes from
``settings.yolo_seg_model`` (``SORBED_YOLO_SEG_MODEL``) and defaults to
``yolo26n-seg.pt``, which ``ultralytics`` downloads on first use. An optional
fallback model runs only when the main one finds no wound ("old-first": the
FUSeg-trained model first, the combined model on close-ups it misses). Run with
confidence 0.01 and without the largest-component filter, this is exactly the
mask the staging thresholds were tuned on (HANDOFF, staging step 2).

Prototype note: unlike the ONNX backend, weights are fetched by ``ultralytics``
rather than sha256-verified through Sorbed's registry. This is a research seam,
not the governed production path — see docs/MODELS.md on provenance.
"""

from __future__ import annotations

from functools import lru_cache

import cv2
import numpy as np
from scipy import ndimage

from sorbed.config.settings import Settings
from sorbed.imaging import RasterImage
from sorbed.segmentation.base import SegmentationResult

_BACKEND_NAME = "yolo"
_DEFAULT_MODEL = "yolo26n-seg.pt"
_MAX_SIDE = 1280


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
        fallback_model_id: str | None = None,
        min_area_fraction: float = 0.0015,
        confidence_floor: float = 0.25,
        keep_largest: bool = True,
    ) -> None:
        self._model_id = model_id
        self._fallback_model_id = fallback_model_id
        self._min_area_fraction = float(min_area_fraction)
        self._confidence_floor = float(confidence_floor)
        self._keep_largest = keep_largest

    @classmethod
    def from_settings(cls, settings: Settings) -> YoloSegmenter:
        return cls(
            settings.yolo_seg_model.strip() or _DEFAULT_MODEL,
            fallback_model_id=settings.yolo_seg_fallback_model.strip() or None,
            min_area_fraction=settings.min_wound_area_fraction,
            confidence_floor=settings.yolo_seg_confidence,
            keep_largest=settings.yolo_seg_keep_largest,
        )

    def segment(self, image: RasterImage) -> SegmentationResult:
        orig_h, orig_w = image.shape_hw
        # Ultralytics expects an 8-bit BGR array when handed a numpy image. Full-size
        # instance masks cost N x H x W memory, so very large photos run downscaled.
        bgr = cv2.cvtColor(image.to_uint8_rgb(), cv2.COLOR_RGB2BGR)
        scale = _MAX_SIDE / max(orig_h, orig_w)
        if scale < 1.0:
            size = (max(1, round(orig_w * scale)), max(1, round(orig_h * scale)))
            bgr = cv2.resize(bgr, size, interpolation=cv2.INTER_AREA)

        mask, confidence = self._predict(self._model_id, bgr)
        if not mask.any() and self._fallback_model_id:
            mask, confidence = self._predict(self._fallback_model_id, bgr)
        if mask.shape != (orig_h, orig_w):
            mask = cv2.resize(mask.astype(np.uint8), (orig_w, orig_h), interpolation=cv2.INTER_NEAREST) > 0
        mask = mask & self._valid_mask(image)
        if self._keep_largest:
            mask = self._largest_component(mask)

        return SegmentationResult(
            wound_mask=mask.astype(bool),
            confidence=float(confidence),
            backend=_BACKEND_NAME,
            weights_sha256=None,
        )

    def _predict(self, model_id: str, bgr: np.ndarray) -> tuple[np.ndarray, float]:
        """Union of all detected instance masks, and how sure the detector is of it.

        The wound is a single region, but a detector may return several instances
        (or none). Confidence is the detection score behind each mask pixel (the
        best covering instance), averaged over the mask — never a constant.
        """
        h, w = bgr.shape[:2]
        empty = np.zeros((h, w), dtype=bool)
        result = _load_model(model_id).predict(
            bgr, conf=self._confidence_floor, verbose=False, retina_masks=True
        )[0]
        masks = getattr(result, "masks", None)
        if masks is None or masks.data is None or len(masks.data) == 0:
            return empty, 0.0

        # (N, H, W) in [0,1]; retina masks are already at the input resolution.
        data = masks.data.cpu().numpy()
        if data.shape[1:] != (h, w):
            data = np.stack([cv2.resize(m.astype(np.float32), (w, h), interpolation=cv2.INTER_NEAREST)
                             for m in data])
        instances = data > 0.5
        wound = instances.any(axis=0)
        if not wound.any():
            return wound, 0.0
        scores = result.boxes.conf.cpu().numpy().astype(np.float32)
        per_pixel = (instances * scores[:, None, None]).max(axis=0)
        return wound, max(0.0, min(1.0, float(per_pixel[wound].mean())))

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

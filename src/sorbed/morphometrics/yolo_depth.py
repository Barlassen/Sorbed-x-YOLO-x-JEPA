"""A YOLO26 monocular-depth cue — a learned counterpart to the shading proxy.

The classical :func:`sorbed.morphometrics.depth_proxy.shading_depth_proxy` reads
interior-vs-rim darkening as a weak depth hint. This module produces the same
:class:`~sorbed.domain.metrics.DepthProxy` contract — a monotone index in [0, 1]
— but derives it from a YOLO26 monocular-depth model (``yolo26*-depth.pt``), which
predicts a per-pixel depth map from one RGB image.

It is still **not a physical measurement**: monocular metric depth from an
uncalibrated ward photo is unreliable, so ``is_physical_measurement`` stays
``False`` and the value feeds the staging engine only as weak, flagged evidence
(the ``depth_cue`` in the Stage-3 rule). What it buys over the shading proxy is a
learned cue that responds to scene geometry rather than raw pixel brightness.

The model id/path is read from ``SORBED_YOLO_DEPTH_MODEL`` and defaults to
``yolo26n-depth.pt``, which ``ultralytics`` downloads on first use. ``ultralytics``
is imported lazily so the weight-free core never depends on torch.
"""

from __future__ import annotations

import os
from functools import lru_cache

import cv2
import numpy as np
from scipy import ndimage

from sorbed.domain.metrics import DepthProxy

_METHOD = "yolo26_depth_v1"
_ENV_MODEL = "SORBED_YOLO_DEPTH_MODEL"
_DEFAULT_MODEL = "yolo26n-depth.pt"


@lru_cache(maxsize=4)
def _load_model(model_id: str):
    from ultralytics import YOLO

    return YOLO(model_id)


def _predict_depth(rgb: np.ndarray, model_id: str) -> np.ndarray:
    """Return a (H, W) float depth map (in the model's metric units) for ``rgb``.

    ``rgb`` is float32 [0, 1] (H, W, 3). Ultralytics wants an 8-bit BGR array.
    """
    model = _load_model(model_id)
    bgr = cv2.cvtColor(
        (np.clip(rgb, 0.0, 1.0) * 255.0).astype(np.uint8), cv2.COLOR_RGB2BGR
    )
    result = model.predict(bgr, verbose=False)[0]
    depth = result.depth.data.cpu().numpy().astype(np.float32)
    if depth.shape != rgb.shape[:2]:
        depth = cv2.resize(
            depth, (rgb.shape[1], rgb.shape[0]), interpolation=cv2.INTER_LINEAR
        )
    return depth


def yolo_depth_proxy(
    rgb: np.ndarray, wound_mask: np.ndarray, *, model_id: str | None = None
) -> DepthProxy | None:
    """Interior-vs-rim depth contrast of the wound, mapped to [0, 1].

    A deeper crater sits farther from the camera, so the wound interior reads as a
    larger depth than its rim. We take that gap and normalise it into the unitless
    [0, 1] index the staging rule expects. Returns ``None`` when there is no wound
    or the geometry is too thin to separate rim from interior (mirroring the
    classical proxy's contract), so callers can fall back cleanly.
    """
    if not wound_mask.any():
        return None

    model_id = model_id or os.environ.get(_ENV_MODEL, "").strip() or _DEFAULT_MODEL
    depth = _predict_depth(rgb, model_id)

    distance = ndimage.distance_transform_edt(wound_mask)
    if distance.max() <= 0:
        return None
    rim = wound_mask & (distance <= max(1.0, 0.15 * distance.max()))
    interior = wound_mask & (distance >= 0.6 * distance.max())
    if not rim.any() or not interior.any():
        return None

    rim_d = float(depth[rim].mean())
    interior_d = float(depth[interior].mean())
    # Positive when the interior is farther (deeper) than the rim. Normalise by the
    # rim depth so the index is scale-relative, then clip into [0, 1].
    index = (interior_d - rim_d) / (abs(rim_d) + 1e-6)
    index = float(np.clip(index, 0.0, 1.0))
    return DepthProxy(
        relative_depth_index=round(index, 4),
        method=_METHOD,
        is_physical_measurement=False,
    )

"""Synthetic-but-real wound images with known ground truth.

These are genuine rasters (real pixels, encodable to any format) whose geometry
and tissue regions are known *by construction*. Tests assert that the pipeline
*recovers* those known values within tolerance — a real computation validated
against real geometry, never a hand-typed expected result.
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import numpy as np
from PIL import Image

SKIN_TONE = (0.80, 0.62, 0.52)
GRANULATION = (0.72, 0.20, 0.18)
SLOUGH = (0.86, 0.77, 0.34)
ESCHAR = (0.15, 0.12, 0.11)
MAROON = (0.42, 0.14, 0.22)


@dataclass(frozen=True)
class SynthWound:
    rgb: np.ndarray  # float32 (H, W, 3) in [0, 1]
    wound_mask: np.ndarray  # bool (H, W), the true wound region
    ellipse_axes_px: tuple[float, float]  # (semi-major*2, semi-minor*2) full lengths

    @property
    def true_area_px(self) -> int:
        return int(self.wound_mask.sum())

    def to_png_bytes(self) -> bytes:
        return _encode(self.rgb, "PNG")

    def to_jpeg_bytes(self) -> bytes:
        return _encode(self.rgb, "JPEG")


def make_wound(
    *,
    h: int = 220,
    w: int = 300,
    center: tuple[int, int] | None = None,
    axes: tuple[int, int] = (80, 55),
    fill: tuple[float, float, float] = GRANULATION,
    inner: tuple[tuple[int, int], tuple[float, float, float]] | None = None,
    seed: int = 0,
) -> SynthWound:
    """Render an elliptical wound of known size on a skin-tone background."""
    rng = np.random.default_rng(seed)
    img = np.full((h, w, 3), SKIN_TONE, dtype=np.float32)
    img += rng.normal(0.0, 0.015, (h, w, 3)).astype(np.float32)
    cx, cy = center or (w // 2, h // 2)
    yy, xx = np.ogrid[:h, :w]
    ell = ((xx - cx) ** 2) / axes[0] ** 2 + ((yy - cy) ** 2) / axes[1] ** 2 <= 1
    img[ell] = fill
    if inner is not None:
        (iax, icol) = inner
        inner_ell = ((xx - cx) ** 2) / iax[0] ** 2 + ((yy - cy) ** 2) / iax[1] ** 2 <= 1
        img[inner_ell] = icol
    img = np.clip(img, 0.0, 1.0).astype(np.float32)
    return SynthWound(rgb=img, wound_mask=ell, ellipse_axes_px=(2.0 * axes[0], 2.0 * axes[1]))


def blank_image(h: int = 200, w: int = 260, value: float = 0.5) -> np.ndarray:
    """A featureless gray image (no wound) for safety/abstention tests."""
    return np.full((h, w, 3), value, dtype=np.float32)


def _encode(rgb: np.ndarray, fmt: str) -> bytes:
    buf = io.BytesIO()
    Image.fromarray((np.clip(rgb, 0, 1) * 255).astype(np.uint8)).save(buf, format=fmt)
    return buf.getvalue()

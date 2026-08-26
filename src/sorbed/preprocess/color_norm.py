"""Illumination / white-balance normalization.

Phone photos of the same wound vary widely in white balance and lighting, which
shifts the very colors the tissue classifier depends on. Gray-world normalization
removes a global color cast by forcing the per-channel means to a common gray,
stabilizing tissue color across images without inventing detail.
"""

from __future__ import annotations

import numpy as np


def gray_world_normalize(rgb: np.ndarray, valid: np.ndarray | None = None) -> np.ndarray:
    """Return a white-balanced copy of ``rgb`` (float32, ``[0, 1]``).

    Uses the gray-world assumption: the average scene color is achromatic. Only
    ``valid`` pixels (e.g. opaque ones) inform the estimate; all pixels are
    scaled by the same factors so no spatial detail is fabricated.
    """
    sample = rgb[valid] if valid is not None and valid.any() else rgb.reshape(-1, 3)
    means = sample.reshape(-1, 3).mean(axis=0)
    gray = float(means.mean())
    scale = np.where(means > 1e-4, gray / means, 1.0).astype(np.float32)
    out = np.clip(rgb * scale, 0.0, 1.0).astype(np.float32)
    return out

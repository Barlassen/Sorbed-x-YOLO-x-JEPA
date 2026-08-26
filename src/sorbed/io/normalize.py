"""Convert decoder outputs to Sorbed's canonical pixel form.

Canonical form is float32 ``(H, W, 3)`` RGB in ``[0, 1]``, sRGB, EXIF-oriented.
These helpers are shared by every decoder so all formats converge identically.
"""

from __future__ import annotations

import numpy as np


def to_float01(array: np.ndarray) -> np.ndarray:
    """Scale an integer image to float32 ``[0, 1]`` by its dtype's max.

    Floating inputs are assumed already in ``[0, 1]`` and only clipped.
    """
    if np.issubdtype(array.dtype, np.floating):
        return np.clip(array.astype(np.float32), 0.0, 1.0)
    info = np.iinfo(array.dtype)
    scale = float(info.max - info.min)
    shifted = array.astype(np.float32) - float(info.min)
    return np.clip(shifted / scale, 0.0, 1.0) if scale > 0 else shifted


def split_alpha(rgb_like: np.ndarray) -> tuple[np.ndarray, np.ndarray | None]:
    """Split a possibly-RGBA float array into RGB and an optional alpha plane."""
    if rgb_like.ndim == 2:
        rgb = np.repeat(rgb_like[:, :, None], 3, axis=2)
        return rgb, None
    if rgb_like.shape[2] == 1:
        rgb = np.repeat(rgb_like, 3, axis=2)
        return rgb, None
    if rgb_like.shape[2] == 3:
        return rgb_like, None
    if rgb_like.shape[2] == 4:
        return rgb_like[:, :, :3], rgb_like[:, :, 3]
    raise ValueError(f"unsupported channel count: {rgb_like.shape[2]}")


def composite_over_gray(rgb: np.ndarray, alpha: np.ndarray | None, gray: float = 0.5) -> np.ndarray:
    """Composite transparent pixels over a neutral gray so color stats are stable."""
    if alpha is None:
        return rgb
    a = alpha[:, :, None]
    return rgb * a + gray * (1.0 - a)


def ensure_canonical(rgb: np.ndarray) -> np.ndarray:
    """Return a contiguous float32 ``(H, W, 3)`` array in ``[0, 1]``."""
    if rgb.dtype != np.float32:
        rgb = rgb.astype(np.float32)
    rgb = np.ascontiguousarray(np.clip(rgb, 0.0, 1.0))
    if rgb.ndim != 3 or rgb.shape[2] != 3:
        raise ValueError(f"expected (H, W, 3), got {rgb.shape}")
    return rgb

"""Periwound skin analysis: erythema and maceration in the ring around the wound.

Erythema is measured as a* (redness) elevation of the periwound ring relative to
a healthy-skin reference sampled at the image border. Maceration (white, soggy
skin) shows as a bright, low-chroma ring.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage import color

from sorbed.domain.metrics import PeriwoundFindings


def analyze_periwound(rgb: np.ndarray, wound_mask: np.ndarray) -> PeriwoundFindings | None:
    """Analyze the skin ring immediately surrounding the wound."""
    if not wound_mask.any():
        return None
    h, w = wound_mask.shape
    ring_width = max(2, int(0.03 * np.sqrt(h * w)))
    dilated = ndimage.binary_dilation(wound_mask, iterations=ring_width)
    ring = dilated & ~wound_mask
    if not ring.any():
        return None

    lab = color.rgb2lab(rgb)
    a = lab[..., 1]
    L = lab[..., 0]
    b = lab[..., 2]

    reference = _border_reference(a, wound_mask)
    erythema_index = float(a[ring].mean() - reference)

    ring_L = L[ring]
    ring_chroma = np.sqrt(a[ring] ** 2 + b[ring] ** 2)
    maceration_fraction = float(((ring_L > 70) & (ring_chroma < 12)).mean())

    return PeriwoundFindings(
        erythema_index=round(erythema_index, 3),
        maceration_suspected=maceration_fraction > 0.25,
        analyzed_ring_px=int(ring.sum()),
    )


def _border_reference(a_channel: np.ndarray, wound_mask: np.ndarray) -> float:
    h, w = a_channel.shape
    bh, bw = max(1, int(0.06 * h)), max(1, int(0.06 * w))
    border = np.zeros((h, w), dtype=bool)
    border[:bh, :] = border[-bh:, :] = True
    border[:, :bw] = border[:, -bw:] = True
    border &= ~wound_mask
    if not border.any():
        return float(a_channel.mean())
    return float(a_channel[border].mean())

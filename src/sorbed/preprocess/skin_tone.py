"""Skin-tone estimation for equity-aware confidence and warnings.

Early-stage and deep-tissue injuries are defined largely by color change, which
is substantially harder to see on darker skin — a documented source of later,
higher-stage diagnosis. We estimate the patient's skin tone with the Individual
Typology Angle (ITA), a standard colorimetric measure, and coarsen it to two
bands so the staging engine can lower confidence and raise an explicit warning
rather than risk a false negative.

The estimate samples *real skin*: the periwound ring when a wound mask is known,
otherwise the image border, and in both cases it discards non-skin pixels —
near-black padding/background and blown-out glare — before computing ITA. It
abstains (``UNKNOWN``) only when too little skin-like signal remains.
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage
from skimage import color

from sorbed.domain.enums import SkinToneBand

# ITA (degrees) below this threshold indicates darker skin (tan/brown/dark).
_ITA_DARK_THRESHOLD = 28.0
_MIN_SKIN_PIXELS = 60


def estimate_skin_tone(rgb: np.ndarray, wound_mask: np.ndarray | None = None) -> SkinToneBand:
    lab = color.rgb2lab(rgb)
    L, a, b = lab[..., 0], lab[..., 1], lab[..., 2]

    # A pixel reads as skin if it is neither padding/background (very dark) nor
    # glare (very bright) and carries the warm chroma of skin.
    skin_like = (L > 20) & (L < 92) & (a > 3) & (b > 3)

    region = _sampling_region(rgb.shape[:2], wound_mask)
    candidate = region & skin_like
    if int(candidate.sum()) < _MIN_SKIN_PIXELS:
        # Fall back to any skin-like pixel outside the wound.
        candidate = skin_like.copy()
        if wound_mask is not None:
            candidate &= ~wound_mask
    if int(candidate.sum()) < _MIN_SKIN_PIXELS:
        return SkinToneBand.UNKNOWN

    L_med = float(np.median(L[candidate]))
    b_med = float(np.median(b[candidate]))
    ita = np.degrees(np.arctan2(L_med - 50.0, b_med))
    return SkinToneBand.IV_VI if ita < _ITA_DARK_THRESHOLD else SkinToneBand.I_III


def _sampling_region(shape: tuple[int, int], wound_mask: np.ndarray | None) -> np.ndarray:
    """The periwound ring when a wound is known, else the image border."""
    h, w = shape
    if wound_mask is not None and wound_mask.any():
        width = max(6, int(0.05 * np.sqrt(h * w)))
        # Offset annulus: skip the immediate periwound (often erythematous, which
        # would bias the tone darker) and sample skin a little further out.
        inner = ndimage.binary_dilation(wound_mask, iterations=max(2, width // 2))
        outer = ndimage.binary_dilation(wound_mask, iterations=width * 2)
        ring = outer & ~inner
        if ring.any():
            return ring
    bh, bw = max(1, int(0.08 * h)), max(1, int(0.08 * w))
    border = np.zeros((h, w), dtype=bool)
    border[:bh, :] = border[-bh:, :] = True
    border[:, :bw] = border[:, -bw:] = True
    if wound_mask is not None:
        border &= ~wound_mask
    return border

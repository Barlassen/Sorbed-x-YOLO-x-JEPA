"""Mask and tissue-overlay rendering."""

from __future__ import annotations

import numpy as np
from PIL import Image

from sorbed.tissue.color_model import TISSUE_ORDER
from sorbed.visualize.palette import tissue_color


def render_mask(wound_mask: np.ndarray) -> Image.Image:
    """A binary wound mask as a white-on-black PNG image."""
    arr = np.where(wound_mask, 255, 0).astype(np.uint8)
    return Image.fromarray(arr, mode="L")


def tissue_color_map(label_map: np.ndarray) -> np.ndarray:
    """Map a tissue label map to an RGB image using the shared palette."""
    h, w = label_map.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, cls in enumerate(TISSUE_ORDER):
        rgb[label_map == idx] = tissue_color(cls)
    return rgb


def render_tissue_overlay(
    display_rgb_u8: np.ndarray,
    label_map: np.ndarray,
    wound_mask: np.ndarray,
    *,
    alpha: float = 0.45,
) -> Image.Image:
    """Blend tissue colors over the wound region of the display image."""
    base = display_rgb_u8.astype(np.float32)
    colors = tissue_color_map(label_map).astype(np.float32)
    blend = base.copy()
    m = wound_mask
    blend[m] = (1.0 - alpha) * base[m] + alpha * colors[m]
    return Image.fromarray(np.clip(blend, 0, 255).astype(np.uint8), mode="RGB")


def render_schematic(
    label_map: np.ndarray,
    wound_mask: np.ndarray,
    *,
    background: tuple[int, int, int] = (236, 237, 240),
) -> Image.Image:
    """A clean, synthetic diagram of the wound: its shape filled with solid tissue
    colors on a plain background, independent of the original photo."""
    h, w = wound_mask.shape
    canvas = np.empty((h, w, 3), dtype=np.uint8)
    canvas[:] = background
    colors = tissue_color_map(label_map)
    canvas[wound_mask] = colors[wound_mask]
    return Image.fromarray(canvas, mode="RGB")

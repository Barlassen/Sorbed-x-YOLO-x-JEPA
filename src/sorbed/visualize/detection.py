"""Detection-style annotated output.

Draws a colored bounding box around the wound with a ``stage  confidence`` label
tab, in the style of object-detection wound papers (e.g. the mobile YOLOv8
pressure-injury detector, Appl. Sci. 2024, 14(16):7124). The box, class, and
confidence are the analysis's own computed values — this is a rendering, not a
separate inference.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from sorbed.domain.analysis import WoundAnalysis
from sorbed.visualize.palette import stage_color


def render_detection(
    analysis: WoundAnalysis,
    display_rgb_u8: np.ndarray,
    *,
    line_width: int = 4,
) -> Image.Image:
    """Draw the wound bounding box + stage/confidence label over the photo."""
    image = Image.fromarray(display_rgb_u8.copy(), mode="RGB").convert("RGB")
    draw = ImageDraw.Draw(image)
    geom = analysis.metrics.geometry
    d = analysis.decision
    color = stage_color(d.stage)

    if geom.area_px <= 0:
        _banner(draw, image.width, "No wound localized", (158, 158, 158))
        return image

    minr, minc, maxr, maxc = geom.bbox_px
    draw.rectangle([minc, minr, maxc, maxr], outline=color, width=line_width)

    label = _label(analysis)
    font = _font(16)
    tw, th = _text_size(draw, label, font)
    pad = 5
    ty = max(0, minr - th - 2 * pad)
    draw.rectangle([minc, ty, minc + tw + 2 * pad, ty + th + 2 * pad], fill=color)
    draw.text((minc + pad, ty + pad), label, fill=_ink(color), font=font)
    return image


def _label(analysis: WoundAnalysis) -> str:
    # Localization only — a neutral "wound" tab. The grade and confidence live in
    # exactly one place (the report hero); burning them onto the photo as well
    # over-emphasises an uncertain, depth-dependent call.
    return "yara · wound"


def _banner(draw: ImageDraw.ImageDraw, width: int, text: str, color: tuple[int, int, int]) -> None:
    font = _font(16)
    draw.rectangle([0, 0, width, 26], fill=color)
    draw.text((6, 4), text, fill=_ink(color), font=font)


def _ink(bg: tuple[int, int, int]) -> tuple[int, int, int]:
    """Black or white label text, whichever contrasts with the box color."""
    luminance = 0.299 * bg[0] + 0.587 * bg[1] + 0.114 * bg[2]
    return (0, 0, 0) if luminance > 140 else (255, 255, 255)


def _font(size: int) -> ImageFont.ImageFont:
    for name in ("DejaVuSans-Bold.ttf", "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _text_size(draw: ImageDraw.ImageDraw, text: str, font: ImageFont.ImageFont) -> tuple[int, int]:
    left, top, right, bottom = draw.textbbox((0, 0), text, font=font)
    return right - left, bottom - top

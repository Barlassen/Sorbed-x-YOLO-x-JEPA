"""The annotated clinical guide image.

Composites the tissue overlay with the wound outline, length/width measurement
lines, a scale bar (when calibrated), a legend of the tissue present, and a
header stating the provisional grade and the decision-support disclaimer. Every
label is drawn from the analysis — no value is invented here.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from skimage import measure

from sorbed.domain.analysis import WoundAnalysis
from sorbed.domain.enums import TissueClass
from sorbed.visualize.overlay import render_schematic, render_tissue_overlay
from sorbed.visualize.palette import (
    CONTOUR_COLOR,
    LENGTH_COLOR,
    SCALEBAR_COLOR,
    WIDTH_COLOR,
    tissue_color,
)

_HEADER_H = 46
_LEGEND_ROW = 22
_PAD = 12


def render_guide(
    analysis: WoundAnalysis,
    display_rgb_u8: np.ndarray,
    wound_mask: np.ndarray,
    label_map: np.ndarray,
) -> Image.Image:
    """Annotated guide drawn over the original photo (tissue overlay + markup)."""
    base = render_tissue_overlay(display_rgb_u8, label_map, wound_mask, alpha=0.5).convert("RGB")
    return _compose(analysis, base, wound_mask)


def render_schematic_guide(
    analysis: WoundAnalysis,
    wound_mask: np.ndarray,
    label_map: np.ndarray,
) -> Image.Image:
    """Annotated guide drawn over a clean synthetic diagram (no photo)."""
    base = render_schematic(label_map, wound_mask).convert("RGB")
    return _compose(analysis, base, wound_mask)


def _compose(
    analysis: WoundAnalysis,
    base: Image.Image,
    wound_mask: np.ndarray,
) -> Image.Image:
    """Draw contour, measurements, scale bar, header, and legend onto ``base``."""
    draw = ImageDraw.Draw(base)
    _draw_contour(draw, wound_mask)
    _draw_measurements(draw, analysis, wound_mask)
    _draw_scalebar(draw, analysis, base.size)

    present = _present_tissues(analysis)
    legend_h = _PAD + len(present) * _LEGEND_ROW + _PAD if present else 0
    canvas = Image.new("RGB", (base.width, _HEADER_H + base.height + legend_h), (18, 18, 20))
    canvas.paste(base, (0, _HEADER_H))
    cdraw = ImageDraw.Draw(canvas)
    _draw_header(cdraw, analysis, base.width)
    if present:
        _draw_legend(cdraw, present, y0=_HEADER_H + base.height + _PAD)
    return canvas


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    for name in (
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
        "DejaVuSans.ttf",
    ):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_header(draw: ImageDraw.ImageDraw, analysis: WoundAnalysis, width: int) -> None:
    d = analysis.decision
    stage = d.stage.value.replace("_", " ").title()
    verb = "withheld" if d.abstained else f"{d.confidence * 100:.0f}% conf."
    draw.rectangle([0, 0, width, _HEADER_H], fill=(28, 28, 32))
    draw.text((_PAD, 6), f"Provisional grade: {stage}  ({verb})", font=_font(18, True),
              fill=(255, 255, 255))
    draw.text((_PAD, 28), "Decision support — NOT a diagnosis. Clinician review required.",
              font=_font(12), fill=(200, 170, 90))


def _draw_contour(draw: ImageDraw.ImageDraw, wound_mask: np.ndarray) -> None:
    if not wound_mask.any():
        return
    for contour in measure.find_contours(wound_mask.astype(float), 0.5):
        pts = [(float(x), float(y)) for y, x in contour]
        if len(pts) >= 2:
            draw.line(pts, fill=CONTOUR_COLOR, width=2)


def _draw_measurements(
    draw: ImageDraw.ImageDraw, analysis: WoundAnalysis, wound_mask: np.ndarray
) -> None:
    geom = analysis.metrics.geometry
    if geom.area_px <= 0:
        return
    minr, minc, maxr, maxc = geom.bbox_px
    cy, cx = geom.centroid_px
    # Length (head-to-toe, vertical) and width (horizontal) through the centroid.
    draw.line([(cx, minr), (cx, maxr)], fill=LENGTH_COLOR, width=2)
    draw.line([(minc, cy), (maxc, cy)], fill=WIDTH_COLOR, width=2)

    if geom.length_mm is not None and geom.width_mm is not None:
        length_label = f"L {geom.length_mm:.0f} mm"
        width_label = f"W {geom.width_mm:.0f} mm"
    else:
        length_label = f"L {geom.length_px:.0f} px"
        width_label = f"W {geom.width_px:.0f} px"
    draw.text((cx + 4, minr + 2), length_label, font=_font(13, True), fill=LENGTH_COLOR)
    draw.text((minc + 2, cy + 4), width_label, font=_font(13, True), fill=WIDTH_COLOR)


def _draw_scalebar(
    draw: ImageDraw.ImageDraw, analysis: WoundAnalysis, size: tuple[int, int]
) -> None:
    width, height = size
    x0, y0 = _PAD, height - _PAD - 8
    mm_per_px = analysis.calibration.mm_per_px
    if mm_per_px is None:
        draw.text((x0, y0 - 6), "no scale reference", font=_font(12), fill=SCALEBAR_COLOR)
        return
    bar_px = 10.0 / mm_per_px  # 1 cm
    if bar_px > width * 0.6:  # 1 cm too wide; use 1 mm
        bar_px, label = 1.0 / mm_per_px, "1 mm"
    else:
        label = "1 cm"
    draw.rectangle([x0, y0, x0 + bar_px, y0 + 6], fill=SCALEBAR_COLOR)
    draw.text((x0, y0 - 16), label, font=_font(12, True), fill=SCALEBAR_COLOR)


def _present_tissues(analysis: WoundAnalysis) -> list[tuple[TissueClass, float]]:
    fr = analysis.metrics.tissue.fractions
    items = [
        (c, f) for c, f in fr.items() if f >= 0.02 and c is not TissueClass.BACKGROUND
    ]
    return sorted(items, key=lambda kv: kv[1], reverse=True)


def _draw_legend(
    draw: ImageDraw.ImageDraw, present: list[tuple[TissueClass, float]], y0: int
) -> None:
    for i, (cls, frac) in enumerate(present):
        y = y0 + i * _LEGEND_ROW
        draw.rectangle([_PAD, y, _PAD + 16, y + 14], fill=tissue_color(cls))
        label = f"{cls.value.replace('_', ' ').title()}  {frac * 100:.0f}%"
        draw.text((_PAD + 24, y), label, font=_font(13), fill=(235, 235, 235))

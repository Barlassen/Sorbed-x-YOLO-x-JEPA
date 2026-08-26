"""A single, comprehensive analysis figure with a clean, minimal UI.

Combines the input, wound mask, tissue match, a shading-based depth heatmap, and
the detection box in one panel row, then lays out the statistics: size and shape,
a tissue-composition bar, PUSH / DESIGN-R sub-scores, depth and periwound cues,
provenance, and the top evidence. Typography is Inter (vendored) and section /
panel headers carry small line icons. Everything shown is computed by the
pipeline; the depth map is a relative shading cue, not a measurement.
"""

from __future__ import annotations

import textwrap

import cv2
import numpy as np
from PIL import Image, ImageDraw

from sorbed.domain.analysis import WoundAnalysis
from sorbed.domain.enums import TissueClass
from sorbed.visualize.detection import render_detection
from sorbed.visualize.icons import draw_icon
from sorbed.visualize.overlay import render_mask, render_tissue_overlay
from sorbed.visualize.palette import stage_color, tissue_color
from sorbed.visualize.typography import font

# Minimal light theme.
_BG = (250, 250, 251)
_INK = (28, 30, 36)
_MUTED = (140, 145, 156)
_FAINT = (176, 180, 190)
_HAIR = (229, 231, 236)
_TILE = (243, 244, 246)

_MARGIN = 32
_PANEL = 196
_GAP = 20
_HEADER = 88
# (caption, icon)
_PANELS = (
    ("Input", "image"),
    ("Mask", "mask"),
    ("Tissue", "layers"),
    ("Depth", "depth"),
    ("Detection", "target"),
)


def render_depth_overlay(
    rgb_u8: np.ndarray, depth_field: np.ndarray, wound_mask: np.ndarray, *, alpha: float = 0.7
) -> Image.Image:
    """Colorized relative-depth heatmap blended over the wound region."""
    d8 = (np.clip(depth_field, 0.0, 1.0) * 255).astype(np.uint8)
    heat = cv2.cvtColor(cv2.applyColorMap(d8, cv2.COLORMAP_TURBO), cv2.COLOR_BGR2RGB)
    out = rgb_u8.astype(np.float32)
    m = wound_mask
    out[m] = (1.0 - alpha) * rgb_u8[m] + alpha * heat[m]
    return Image.fromarray(np.clip(out, 0, 255).astype(np.uint8), mode="RGB")


def render_dashboard(
    analysis: WoundAnalysis,
    display_rgb_u8: np.ndarray,
    wound_mask: np.ndarray,
    tissue_label_map: np.ndarray,
    depth_field: np.ndarray,
) -> Image.Image:
    """Build the analysis dashboard image."""
    images = [
        Image.fromarray(display_rgb_u8, mode="RGB"),
        render_mask(wound_mask).convert("RGB"),
        render_tissue_overlay(display_rgb_u8, tissue_label_map, wound_mask),
        render_depth_overlay(display_rgb_u8, depth_field, wound_mask),
        render_detection(analysis, display_rgb_u8),
    ]
    cols = len(images)
    width = 2 * _MARGIN + cols * _PANEL + (cols - 1) * _GAP
    height = _HEADER + _PANEL + 44 + 322
    canvas = Image.new("RGB", (width, height), _BG)
    draw = ImageDraw.Draw(canvas)

    _header(draw, analysis, width)
    y_panels = _HEADER + 18
    for i, ((label, icon), img) in enumerate(zip(_PANELS, images, strict=True)):
        x = _MARGIN + i * (_PANEL + _GAP)
        _panel(canvas, draw, img, label, icon, x, y_panels)

    _stats(draw, analysis, _MARGIN, y_panels + _PANEL + 46, width)
    return canvas


def _tw(draw: ImageDraw.ImageDraw, text: str, f: object) -> int:
    left, _, right, _ = draw.textbbox((0, 0), text, font=f)
    return right - left


def _header(draw: ImageDraw.ImageDraw, analysis: WoundAnalysis, width: int) -> None:
    d = analysis.decision
    color = stage_color(d.stage)

    draw.rounded_rectangle([_MARGIN, 26, _MARGIN + 34, 60], radius=9, fill=color)
    draw_icon(draw, "spark", _MARGIN + 8, 34, 18, (255, 255, 255), width=2)
    draw.text((_MARGIN + 46, 24), "Sorbed", font=font(24, "Bold"), fill=_INK)
    draw.text((_MARGIN + 47, 54), "pressure-injury analysis", font=font(12), fill=_MUTED)

    stage = d.stage.value.replace("_", " ").title()
    _pill(draw, stage, color, width - _MARGIN, 24)

    # Confidence meter.
    conf = 0.0 if d.abstained else d.confidence
    label = "review" if d.abstained else f"{d.confidence * 100:.0f}% confidence"
    mw, mx1 = 190, width - _MARGIN
    mx0 = mx1 - mw
    my = 62
    draw.rounded_rectangle([mx0, my, mx1, my + 8], radius=4, fill=_TILE)
    if conf > 0:
        draw.rounded_rectangle([mx0, my, mx0 + int(mw * conf), my + 8], radius=4, fill=color)
    draw.text((mx0 - 8 - _tw(draw, label, font(11, "Medium")), my - 3), label,
              font=font(11, "Medium"), fill=_MUTED)

    draw.line([(_MARGIN, _HEADER), (width - _MARGIN, _HEADER)], fill=_HAIR, width=1)


def _pill(
    draw: ImageDraw.ImageDraw, text: str, color: tuple[int, int, int], right: int, top: int
) -> None:
    f = font(15, "SemiBold")
    tw = _tw(draw, text, f)
    pad_x, h, dot = 30, 30, 8
    x0 = right - tw - pad_x - 20
    draw.rounded_rectangle([x0, top, right, top + h], radius=h // 2, fill=color)
    lum = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
    ink = (0, 0, 0) if lum > 150 else (255, 255, 255)
    draw.ellipse([x0 + 12, top + h / 2 - dot / 2, x0 + 12 + dot, top + h / 2 + dot / 2], fill=ink)
    draw.text((x0 + 26, top + 6), text, font=f, fill=ink)


def _fit(img: Image.Image, box: int) -> Image.Image:
    img = img.convert("RGB")
    scale = min(box / img.width, box / img.height)
    resized = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))))
    tile = Image.new("RGB", (box, box), _TILE)
    tile.paste(resized, ((box - resized.width) // 2, (box - resized.height) // 2))
    return tile


def _panel(
    canvas: Image.Image, draw: ImageDraw.ImageDraw, img: Image.Image, label: str, icon: str,
    x: int, y: int,
) -> None:
    canvas.paste(_fit(img, _PANEL), (x, y))
    draw.rectangle([x, y, x + _PANEL - 1, y + _PANEL - 1], outline=_HAIR, width=1)
    draw_icon(draw, icon, x, y + _PANEL + 9, 15, _FAINT, width=2)
    draw.text((x + 21, y + _PANEL + 10), label, font=font(12, "SemiBold"), fill=_MUTED)


def _section(draw: ImageDraw.ImageDraw, icon: str, title: str, x: int, y: int) -> None:
    draw_icon(draw, icon, x, y, 15, _FAINT, width=2)
    draw.text((x + 21, y + 1), title, font=font(11, "SemiBold"), fill=_FAINT)


def _stats(
    draw: ImageDraw.ImageDraw, analysis: WoundAnalysis, x0: int, y0: int, width: int
) -> None:
    g = analysis.metrics.geometry
    gutter = 36
    col_w = (width - 2 * _MARGIN - 2 * gutter) // 3
    cx = [x0, x0 + col_w + gutter, x0 + 2 * (col_w + gutter)]

    mm = analysis.calibration.mm_per_px
    size_rows = (
        [
            ("Area", f"{g.area_cm2:.2f} cm²"),
            ("Length × width", f"{g.length_mm:.0f} × {g.width_mm:.0f} mm"),
            ("Perimeter", f"{g.perimeter_px * mm:.0f} mm"),
        ]
        if g.area_cm2 is not None and mm
        else [
            ("Area", f"{g.area_px:.0f} px"),
            ("Length × width", f"{g.length_px:.0f} × {g.width_px:.0f} px"),
            ("Perimeter", f"{g.perimeter_px:.0f} px"),
        ]
    )
    size_rows += [
        ("Circularity", f"{g.circularity:.2f}"),
        ("Wound / image", f"{g.wound_fraction_of_image * 100:.1f}%"),
        ("Calibration", analysis.calibration.status.value.replace("_", " ")),
        ("Skin tone", _tone_label(analysis)),
    ]
    _section(draw, "ruler", "MEASUREMENTS", cx[0], y0)
    _rows(draw, size_rows, cx[0], y0 + 26, col_w)

    _section(draw, "droplet", "TISSUE", cx[1], y0)
    _tissue_strip(draw, analysis, cx[1], y0 + 30, col_w)

    hs, dp, pw = (
        analysis.metrics.healing_scores,
        analysis.metrics.depth_proxy,
        analysis.metrics.periwound,
    )
    score_rows: list[tuple[str, str]] = []
    if hs and hs.push_partial_total is not None:
        score_rows.append(("PUSH (partial)", str(hs.push_partial_total)))
    if hs and hs.design_r_size_subscore is not None:
        score_rows.append(("DESIGN-R size", str(hs.design_r_size_subscore)))
    if hs and hs.granulation_percent is not None:
        score_rows.append(("Granulation", f"{hs.granulation_percent:.0f}%"))
    if dp:
        score_rows.append(("Depth (relative)", f"{dp.relative_depth_index:.2f}"))
    if pw and pw.erythema_index is not None:
        score_rows.append(("Periwound erythema", f"{pw.erythema_index:+.1f} a*"))
    score_rows.append(("Segmenter", analysis.provenance.segmentation_backend))
    _section(draw, "gauge", "SCORES & CUES", cx[2], y0)
    _rows(draw, score_rows, cx[2], y0 + 26, col_w)

    _evidence(draw, analysis, cx[0], y0 + 200, width - 2 * _MARGIN)


def _tone_label(analysis: WoundAnalysis) -> str:
    return {
        "fitzpatrick_i_iii": "Fitzpatrick I–III",
        "fitzpatrick_iv_vi": "Fitzpatrick IV–VI",
    }.get(analysis.skin_tone_band.value, "unknown")


def _rows(
    draw: ImageDraw.ImageDraw, rows: list[tuple[str, str]], x: int, y: int, w: int
) -> None:
    for label, value in rows:
        draw.text((x, y), label, font=font(12), fill=_MUTED)
        draw.text((x + w - _tw(draw, value, font(12, "SemiBold")), y), value,
                  font=font(12, "SemiBold"), fill=_INK)
        y += 22


def _tissue_strip(
    draw: ImageDraw.ImageDraw, analysis: WoundAnalysis, x: int, y: int, w: int
) -> None:
    fr = [
        (c, f)
        for c, f in sorted(analysis.metrics.tissue.fractions.items(), key=lambda kv: -kv[1])
        if f > 0.005 and c is not TissueClass.BACKGROUND
    ]
    if not fr:
        return
    h = 16
    draw.rounded_rectangle([x, y, x + w, y + h], radius=h // 2, fill=_TILE)
    cxp = x
    for i, (c, f) in enumerate(fr):
        end = x + w if i == len(fr) - 1 else min(x + w, cxp + max(2, int(w * f)))
        draw.rectangle([cxp, y, end, y + h], fill=tissue_color(c))
        cxp = end
    yy = y + h + 12
    for c, f in fr:
        draw.ellipse([x, yy + 2, x + 10, yy + 12], fill=tissue_color(c))
        draw.text((x + 18, yy), c.value.replace("_", " ").title(), font=font(12), fill=_INK)
        pct = f"{f * 100:.0f}%"
        draw.text((x + w - _tw(draw, pct, font(12, "SemiBold")), yy), pct,
                  font=font(12, "SemiBold"), fill=_MUTED)
        yy += 19


def _evidence(draw: ImageDraw.ImageDraw, analysis: WoundAnalysis, x: int, y: int, w: int) -> None:
    draw.line([(x, y - 14), (x + w, y - 14)], fill=_HAIR, width=1)
    _section(draw, "spark", "WHY THIS GRADE", x, y)
    yy = y + 24
    char_w = max(60, int(w / 6.6))
    for ev in analysis.decision.evidence[:2]:
        for line in textwrap.wrap(ev.description, width=char_w)[:2]:
            draw.text((x + 21, yy), line, font=font(12), fill=_INK)
            yy += 17
    for c in analysis.decision.caveats[:2]:
        col = {"info": (70, 120, 190), "warning": (185, 130, 40), "critical": (200, 70, 70)}.get(
            c.severity.value, _MUTED
        )
        draw.ellipse([x + 2, yy + 4, x + 10, yy + 12], fill=col)
        for line in textwrap.wrap(c.message, width=char_w)[:2]:
            draw.text((x + 21, yy), line, font=font(12), fill=col)
            yy += 17

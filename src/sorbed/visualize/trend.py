"""Longitudinal healing-trend dashboard.

Shows the same wound across visits: per-visit thumbnails, a wound-area line chart
with the projected-closure line, tissue composition over time, and the validated
healing statistics (percent area reduction, healing velocity, 4-week PAR, PUSH
trend). Every value comes from :func:`sorbed.trend.compute.compute_trend`.
"""

from __future__ import annotations

import textwrap

from PIL import Image, ImageDraw

from sorbed.domain.enums import TissueClass
from sorbed.trend.models import HealingTrend
from sorbed.visualize.icons import draw_icon
from sorbed.visualize.palette import tissue_color
from sorbed.visualize.plotting import line_chart, stacked_bars
from sorbed.visualize.typography import font

_BG = (250, 250, 251)
_INK = (28, 30, 36)
_MUTED = (140, 145, 156)
_FAINT = (176, 180, 190)
_HAIR = (229, 231, 236)
_TILE = (243, 244, 246)

_TRAJECTORY_COLOR = {
    "healing": (0, 178, 80),
    "stalled": (230, 160, 30),
    "deteriorating": (213, 40, 40),
    "indeterminate": (150, 152, 158),
}
_TISSUE_ORDER = (
    TissueClass.GRANULATION, TissueClass.SLOUGH, TissueClass.ESCHAR,
    TissueClass.EPITHELIAL, TissueClass.INTACT_SKIN,
)
_MARGIN = 32
_THUMB = 150


def render_trend_dashboard(trend: HealingTrend, thumbnails: list[Image.Image]) -> Image.Image:
    """Render the healing-trend figure for a series of visits."""
    n = len(trend.points)
    width = max(1120, _MARGIN * 2 + n * (_THUMB + 16))
    height = 108 + _THUMB + 64 + 210 + 26 + 140
    canvas = Image.new("RGB", (width, height), _BG)
    draw = ImageDraw.Draw(canvas)

    _header(draw, trend, width)
    y = 108
    _thumbnails(canvas, draw, trend, thumbnails, y)
    y += _THUMB + 64

    chart_w = (width - _MARGIN * 2 - 24) // 2
    area_chart = _area_chart(trend, (chart_w, 210))
    tissue_chart = _tissue_chart(trend, (chart_w, 210))
    canvas.paste(area_chart, (_MARGIN, y))
    canvas.paste(tissue_chart, (_MARGIN + chart_w + 24, y))
    _chart_caption(draw, "Wound area over time", _MARGIN, y - 4)
    _chart_caption(draw, "Tissue composition over time", _MARGIN + chart_w + 24, y - 4)
    y += 210 + 26

    _stats(draw, trend, _MARGIN, y, width)
    return canvas


def _tw(draw: ImageDraw.ImageDraw, text: str, f: object) -> int:
    left, _, right, _ = draw.textbbox((0, 0), text, font=f)
    return right - left


def _header(draw: ImageDraw.ImageDraw, trend: HealingTrend, width: int) -> None:
    color = _TRAJECTORY_COLOR.get(trend.trajectory, _MUTED)
    draw.rounded_rectangle([_MARGIN, 26, _MARGIN + 34, 60], radius=9, fill=color)
    draw_icon(draw, "gauge", _MARGIN + 8, 34, 18, (255, 255, 255), width=2)
    draw.text((_MARGIN + 46, 24), "Healing trend", font=font(23, "Bold"), fill=_INK)
    sub = f"{len(trend.points)} visits"
    if trend.patient_ref:
        sub = f"{trend.patient_ref} · {sub}"
    draw.text((_MARGIN + 47, 54), sub, font=font(12), fill=_MUTED)

    label = trend.trajectory.title()
    _pill(draw, label, color, width - _MARGIN, 24)
    par = f"{trend.percent_area_reduction:+.0f}% area"
    draw.text((width - _MARGIN - _tw(draw, par, font(12, "Medium")), 60), par,
              font=font(12, "Medium"), fill=_MUTED)
    draw.line([(_MARGIN, 88), (width - _MARGIN, 88)], fill=_HAIR, width=1)


def _pill(draw: ImageDraw.ImageDraw, text: str, color, right: int, top: int) -> None:
    f = font(15, "SemiBold")
    tw = _tw(draw, text, f)
    h, dot = 30, 8
    x0 = right - tw - 46
    draw.rounded_rectangle([x0, top, right, top + h], radius=h // 2, fill=color)
    lum = 0.299 * color[0] + 0.587 * color[1] + 0.114 * color[2]
    ink = (0, 0, 0) if lum > 150 else (255, 255, 255)
    draw.ellipse([x0 + 12, top + h / 2 - dot / 2, x0 + 12 + dot, top + h / 2 + dot / 2], fill=ink)
    draw.text((x0 + 26, top + 6), text, font=f, fill=ink)


def _thumbnails(
    canvas: Image.Image, draw: ImageDraw.ImageDraw, trend: HealingTrend,
    thumbs: list[Image.Image], y: int,
) -> None:
    for i, p in enumerate(trend.points):
        x = _MARGIN + i * (_THUMB + 16)
        if i < len(thumbs):
            canvas.paste(_fit(thumbs[i], _THUMB), (x, y))
        draw.rectangle([x, y, x + _THUMB - 1, y + _THUMB - 1], outline=_HAIR, width=1)
        size = (f"{p.area_cm2:.1f} cm²" if p.area_cm2 is not None else f"{p.area_px:.0f} px")
        draw.text((x + 1, y + _THUMB + 6), p.label, font=font(12, "SemiBold"), fill=_INK)
        draw.text((x + 1, y + _THUMB + 23), f"{p.stage.value.replace('_', ' ')} · {size}",
                  font=font(11), fill=_MUTED)


def _area_chart(trend: HealingTrend, size: tuple[int, int]) -> Image.Image:
    x = [p.day for p in trend.points]
    ys = [(p.area_cm2 if trend.is_calibrated else p.area_px) for p in trend.points]
    color = _TRAJECTORY_COLOR.get(trend.trajectory, _MUTED)
    projection = None
    if trend.projected_days_to_closure is not None and x:
        projection = (x[-1] + trend.projected_days_to_closure, 0.0, color)
    unit = "cm²" if trend.is_calibrated else "px"
    return line_chart(size, x, [(f"area ({unit})", ys, color)], projection=projection)


def _tissue_chart(trend: HealingTrend, size: tuple[int, int]) -> Image.Image:
    labels = [p.label.replace("Day ", "D") for p in trend.points]
    stacks = [
        [(tissue_color(c), p.fraction(c)) for c in _TISSUE_ORDER if p.fraction(c) > 0.005]
        for p in trend.points
    ]
    return stacked_bars(size, labels, stacks)


def _chart_caption(draw: ImageDraw.ImageDraw, text: str, x: int, y: int) -> None:
    draw.text((x, y - 18), text, font=font(11, "SemiBold"), fill=_FAINT)


def _stats(draw: ImageDraw.ImageDraw, trend: HealingTrend, x0: int, y0: int, width: int) -> None:
    draw.line([(x0, y0 - 8), (width - _MARGIN, y0 - 8)], fill=_HAIR, width=1)
    unit = "cm²" if trend.is_calibrated else "px"
    rate = trend.healing_rate_per_week
    rate_txt = "—" if rate is None else f"{rate:+.2f} {unit}/wk"
    par4 = trend.par_at_4_weeks
    par4_txt = "—" if par4 is None else f"{par4:.0f}%"
    if trend.likely_to_heal is not None:
        par4_txt += "  ·  " + ("on track" if trend.likely_to_heal else "below 40%")
    closure = ("—" if trend.projected_days_to_closure is None
               else f"~{trend.projected_days_to_closure:.0f} days")

    cards = [
        ("ruler", "Area reduction", f"{trend.percent_area_reduction:+.0f}%"),
        ("gauge", "Healing rate", rate_txt),
        ("spark", "4-week PAR", par4_txt),
        ("target", "Projected closure", closure),
        ("droplet", "PUSH trend", trend.push_trend or "—"),
    ]
    gap = 16
    cw = (width - _MARGIN * 2 - gap * (len(cards) - 1)) // len(cards)
    for i, (icon, label, value) in enumerate(cards):
        cx = x0 + i * (cw + gap)
        draw.rounded_rectangle([cx, y0, cx + cw, y0 + 74], radius=12, fill=(255, 255, 255),
                               outline=_HAIR, width=1)
        draw_icon(draw, icon, cx + 16, y0 + 15, 15, _FAINT, width=2)
        draw.text((cx + 38, y0 + 16), label, font=font(11, "Medium"), fill=_MUTED)
        draw.text((cx + 16, y0 + 38), value, font=font(18, "Bold"), fill=_INK)

    ny = y0 + 92
    draw_icon(draw, "spark", x0, ny + 1, 12, _FAINT, width=1)
    for note in trend.notes[:2]:
        for line in textwrap.wrap(note, width=max(70, int((width - 2 * _MARGIN) / 6.6)))[:2]:
            draw.text((x0 + 20, ny), line, font=font(11), fill=_MUTED)
            ny += 16


def _fit(img: Image.Image, box: int) -> Image.Image:
    img = img.convert("RGB")
    scale = min(box / img.width, box / img.height)
    resized = img.resize((max(1, int(img.width * scale)), max(1, int(img.height * scale))))
    tile = Image.new("RGB", (box, box), _TILE)
    tile.paste(resized, ((box - resized.width) // 2, (box - resized.height) // 2))
    return tile

"""Inline SVG charts for the reports.

Chromium renders SVG as crisp vectors in print, so charts stay sharp at any
zoom and inherit the document's Manrope type. Each builder returns a self-
contained ``<svg>`` string sized to sit inside a report card. Marks are thin,
grids recessive, endpoints directly labelled, and colour is used only to encode
meaning (a single accent for one series; the domain's semantic tissue colours
for composition).
"""

from __future__ import annotations

from collections.abc import Sequence

from sorbed.domain.enums import TissueClass
from sorbed.trend.models import HealingTrend
from sorbed.visualize.palette import TISSUE_COLORS

_INK = "#14161c"
_MUTED = "#8a93a0"
_GRID = "#e9ecf1"
_AXIS = "#d6dae1"

_TISSUE_ORDER = (
    TissueClass.GRANULATION,
    TissueClass.EPITHELIAL,
    TissueClass.SLOUGH,
    TissueClass.ESCHAR,
)
_TISSUE_LABEL = {
    TissueClass.GRANULATION: "Granülasyon",
    TissueClass.EPITHELIAL: "Epitel",
    TissueClass.SLOUGH: "Slough",
    TissueClass.ESCHAR: "Eskar",
}


def _hex(rgb: tuple[int, int, int]) -> str:
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def _fmt(v: float) -> str:
    if abs(v) >= 100:
        return f"{v:.0f}"
    if abs(v) >= 10:
        return f"{v:.0f}"
    if abs(v) >= 1:
        return f"{v:.1f}"
    return f"{v:.2g}"


def _svg(w: int, h: int, body: str, *, title: str = "") -> str:
    t = f"<title>{title}</title>" if title else ""
    return (
        f"<svg viewBox='0 0 {w} {h}' width='100%' role='img' "
        f"style='display:block;font-family:Manrope,sans-serif'>{t}{body}</svg>"
    )


# --------------------------------------------------------------------------- #
# Line chart (area / PUSH over time)
# --------------------------------------------------------------------------- #
def line_chart_svg(
    xs: Sequence[float],
    ys: Sequence[float],
    *,
    accent: str,
    unit: str = "",
    x_label: str = "gün · day",
    projection: tuple[float, float] | None = None,
    lower_better: bool = False,
    width: int = 340,
    height: int = 208,
) -> str:
    # Generous top padding leaves a clear band for the endpoint callout so it
    # never collides with the line or the y-axis ticks; the bottom band is sized
    # for x tick labels plus the axis caption on separate rows.
    pad_l, pad_r, pad_t, pad_b = 36, 46, 30, 34
    x0, y0, x1, y1 = pad_l, pad_t, width - pad_r, height - pad_b

    all_x = list(xs) + ([projection[0]] if projection else [])
    all_y = list(ys) + ([projection[1]] if projection else [])
    xmin, xmax = min(all_x), max(all_x)
    ymin, ymax = 0.0, (max(all_y) * 1.22 or 1.0)
    xspan = (xmax - xmin) or 1.0
    yspan = (ymax - ymin) or 1.0

    def px(x: float) -> float:
        return x0 + (x - xmin) / xspan * (x1 - x0)

    def py(y: float) -> float:
        return y1 - (y - ymin) / yspan * (y1 - y0)

    parts: list[str] = []
    for i in range(5):
        yy = ymin + yspan * i / 4
        gy = py(yy)
        parts.append(f"<line x1='{x0}' y1='{gy:.1f}' x2='{x1}' y2='{gy:.1f}' "
                     f"stroke='{_GRID}' stroke-width='1'/>")
        parts.append(f"<text x='{x0 - 7}' y='{gy + 3:.1f}' text-anchor='end' "
                     f"font-size='8' fill='{_MUTED}'>{_fmt(yy)}</text>")
    parts.append(f"<line x1='{x0}' y1='{y1}' x2='{x1}' y2='{y1}' stroke='{_AXIS}' stroke-width='1'/>")

    pts = [(px(x), py(y)) for x, y in zip(xs, ys, strict=True)]
    area_d = (f"M {pts[0][0]:.1f} {y1} " + " ".join(f"L {x:.1f} {y:.1f}" for x, y in pts)
              + f" L {pts[-1][0]:.1f} {y1} Z")
    line_d = "M " + " L ".join(f"{x:.1f} {y:.1f}" for x, y in pts)
    parts.append(f"<path d='{area_d}' fill='{accent}' opacity='0.10'/>")
    parts.append(f"<path d='{line_d}' fill='none' stroke='{accent}' stroke-width='2.2' "
                 "stroke-linejoin='round' stroke-linecap='round'/>")

    if projection:
        pxp, pyp = px(projection[0]), py(projection[1])
        parts.append(f"<line x1='{pts[-1][0]:.1f}' y1='{pts[-1][1]:.1f}' x2='{pxp:.1f}' y2='{pyp:.1f}' "
                     f"stroke='{accent}' stroke-width='1.6' stroke-dasharray='3 3' opacity='0.55'/>")
        parts.append(f"<circle cx='{pxp:.1f}' cy='{pyp:.1f}' r='2.6' fill='none' "
                     f"stroke='{accent}' stroke-width='1.4'/>")

    # x tick labels only at first and last visit to avoid crowding.
    for i, (x, y) in enumerate(pts):
        parts.append(f"<circle cx='{x:.1f}' cy='{y:.1f}' r='3.4' fill='{accent}' "
                     "stroke='white' stroke-width='1.4'/>")
        if i in (0, len(pts) - 1):
            anc = "start" if i == 0 else "end"
            xx = x + (2 if i == 0 else -2)
            parts.append(f"<text x='{xx:.1f}' y='{y1 + 15:.1f}' text-anchor='{anc}' "
                         f"font-size='8' fill='{_MUTED}'>{_fmt(list(xs)[i])}</text>")

    # Endpoint callout pinned to the top band — always clear of the plot.
    ex, ey = pts[-1]
    label = f"{_fmt(list(ys)[-1])} {unit}".strip()
    cw = 8 + len(label) * 5.6
    cx0 = min(ex - cw / 2, x1 - cw)
    cx0 = max(cx0, x0)
    cy0 = pad_t - 24
    parts.append(f"<line x1='{ex:.1f}' y1='{ey:.1f}' x2='{cx0 + cw / 2:.1f}' y2='{cy0 + 17:.1f}' "
                 f"stroke='{accent}' stroke-width='1' opacity='0.35'/>")
    parts.append(f"<rect x='{cx0:.1f}' y='{cy0:.1f}' width='{cw:.1f}' height='17' rx='6' "
                 f"fill='{accent}'/>")
    parts.append(f"<text x='{cx0 + cw / 2:.1f}' y='{cy0 + 12:.1f}' text-anchor='middle' "
                 f"font-size='9' font-weight='800' fill='#ffffff'>{label}</text>")
    parts.append(f"<text x='{x1}' y='{height - 5}' text-anchor='end' font-size='7.5' "
                 f"fill='{_MUTED}'>{x_label} →</text>")
    return _svg(width, height, "".join(parts))


# --------------------------------------------------------------------------- #
# Stacked tissue composition over visits
# --------------------------------------------------------------------------- #
def tissue_stack_svg(trend: HealingTrend, *, width: int = 330, height: int = 190) -> str:
    pad_l, pad_r, pad_t, pad_b = 26, 12, 12, 40
    x0, y0, x1, y1 = pad_l, pad_t, width - pad_r, height - pad_b
    n = len(trend.points)
    slot = (x1 - x0) / max(1, n)
    bw = min(46, slot * 0.6)
    parts: list[str] = []
    for i in range(5):  # y grid 0..100
        gy = y1 - (y1 - y0) * i / 4
        parts.append(f"<line x1='{x0}' y1='{gy:.1f}' x2='{x1}' y2='{gy:.1f}' stroke='{_GRID}' stroke-width='1'/>")
        parts.append(f"<text x='{x0 - 5}' y='{gy + 3:.1f}' text-anchor='end' font-size='7.5' "
                     f"fill='{_MUTED}'>{i * 25}</text>")
    for idx, p in enumerate(trend.points):
        cx = x0 + slot * (idx + 0.5)
        top = y1
        for tc in _TISSUE_ORDER:
            frac = p.fraction(tc)
            seg = (y1 - y0) * frac
            if seg <= 0.3:
                continue
            parts.append(
                f"<rect x='{cx - bw / 2:.1f}' y='{top - seg:.1f}' width='{bw:.1f}' height='{seg:.1f}' "
                f"fill='{_hex(TISSUE_COLORS[tc])}' stroke='white' stroke-width='1'/>"
            )
            top -= seg
        parts.append(f"<text x='{cx:.1f}' y='{y1 + 13:.1f}' text-anchor='middle' font-size='8' "
                     f"fill='{_MUTED}'>{p.label.replace('ziyaret', 'z.').replace('. ', '')}</text>")
    # legend
    lx, ly = x0, height - 12
    for tc in _TISSUE_ORDER:
        parts.append(f"<rect x='{lx}' y='{ly - 6}' width='8' height='8' rx='2' fill='{_hex(TISSUE_COLORS[tc])}'/>")
        parts.append(f"<text x='{lx + 11}' y='{ly + 1}' font-size='7.5' fill='{_INK}'>{_TISSUE_LABEL[tc]}</text>")
        lx += 26 + len(_TISSUE_LABEL[tc]) * 4.3
    return _svg(width, height, "".join(parts))


# --------------------------------------------------------------------------- #
# Semicircular healing gauge (percent area reduction)
# --------------------------------------------------------------------------- #
def gauge_svg(pct: float, *, accent: str, caption: str = "", width: int = 200, height: int = 128) -> str:
    import math

    cx, cy, r = width / 2, height - 22, 62
    clamped = max(-100.0, min(100.0, pct))
    frac = (clamped + 100) / 200  # -100..100 -> 0..1 sweep
    a0, a1 = math.pi, math.pi * (1 - frac)

    def arc_pt(a: float) -> tuple[float, float]:
        return cx + r * math.cos(a), cy - r * math.sin(a)

    sx, sy = arc_pt(math.pi)
    ex, ey = arc_pt(0)
    vx, vy = arc_pt(a1)
    large = 0
    parts = [
        f"<path d='M {sx:.1f} {sy:.1f} A {r} {r} 0 {large} 1 {ex:.1f} {ey:.1f}' "
        f"fill='none' stroke='{_GRID}' stroke-width='11' stroke-linecap='round'/>",
    ]
    # value arc from left to the value angle
    large_v = 1 if frac > 0.5 else 0
    parts.append(
        f"<path d='M {sx:.1f} {sy:.1f} A {r} {r} 0 {large_v} 1 {vx:.1f} {vy:.1f}' "
        f"fill='none' stroke='{accent}' stroke-width='11' stroke-linecap='round'/>"
    )
    parts.append(f"<text x='{cx:.1f}' y='{cy - 8:.1f}' text-anchor='middle' font-size='26' "
                 f"font-weight='800' fill='{_INK}'>{pct:+.0f}%</text>")
    if caption:
        parts.append(f"<text x='{cx:.1f}' y='{cy + 12:.1f}' text-anchor='middle' font-size='8.5' "
                     f"fill='{_MUTED}'>{caption}</text>")
    _ = (a0,)  # a0 retained for readability of the sweep definition
    return _svg(width, height, "".join(parts))


# --------------------------------------------------------------------------- #
# Horizontal probability bars (stage likelihood)
# --------------------------------------------------------------------------- #
def hbars_svg(rows: Sequence[tuple[str, float, str]], *, width: int = 330, row_h: int = 22) -> str:
    pad_l, pad_r, pad_t = 96, 34, 6
    height = pad_t * 2 + row_h * len(rows)
    x0, x1 = pad_l, width - pad_r
    parts: list[str] = []
    for i, (label, val, color) in enumerate(rows):
        cy = pad_t + row_h * i + row_h / 2
        w = max(2.0, (x1 - x0) * max(0.0, min(1.0, val)))
        parts.append(f"<text x='{pad_l - 8}' y='{cy + 3:.1f}' text-anchor='end' font-size='8.5' "
                     f"fill='{_INK}'>{label}</text>")
        parts.append(f"<rect x='{x0}' y='{cy - 6:.1f}' width='{x1 - x0}' height='12' rx='6' fill='{_GRID}'/>")
        parts.append(f"<rect x='{x0}' y='{cy - 6:.1f}' width='{w:.1f}' height='12' rx='6' fill='{color}'/>")
        parts.append(f"<text x='{x1 + 4}' y='{cy + 3:.1f}' font-size='8.5' font-weight='700' "
                     f"fill='{_MUTED}'>{val * 100:.0f}%</text>")
    return _svg(width, height, "".join(parts))

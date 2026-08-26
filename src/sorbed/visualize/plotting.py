"""Minimal charts drawn with Pillow, matching the dashboard's clean style.

Kept dependency-free (no matplotlib) so charts render anywhere the rest of the
package does, with the vendored Geist font and the shared palette.
"""

from __future__ import annotations

from PIL import Image, ImageDraw

from sorbed.visualize.typography import font

Color = tuple[int, int, int]

_INK = (28, 30, 36)
_MUTED = (140, 145, 156)
_HAIR = (229, 231, 236)
_TILE = (243, 244, 246)
_BG = (255, 255, 255)


def line_chart(
    size: tuple[int, int],
    x: list[float],
    series: list[tuple[str, list[float | None], Color]],
    *,
    title: str = "",
    x_label: str = "day",
    projection: tuple[float, float, Color] | None = None,
) -> Image.Image:
    """Line chart of one or more series over ``x``.

    ``projection`` draws a dashed segment from the last point to ``(x, y)`` — used
    for the projected-closure line.
    """
    w, h = size
    img = Image.new("RGB", size, _BG)
    d = ImageDraw.Draw(img)
    left, right, top, bottom = 46, 14, 26, 30
    x0, y0, x1, y1 = left, top, w - right, h - bottom

    all_y = [v for _, ys, _ in series for v in ys if v is not None]
    if projection is not None:
        all_y.append(projection[1])
    if not all_y or not x:
        return img
    y_max = max(all_y) * 1.12 or 1.0
    y_min = min(0.0, min(all_y))
    x_min, x_max = min(x), max(x)
    if projection is not None:
        x_max = max(x_max, projection[0])
    x_span = (x_max - x_min) or 1.0
    y_span = (y_max - y_min) or 1.0

    def px(vx: float) -> float:
        return x0 + (vx - x_min) / x_span * (x1 - x0)

    def py(vy: float) -> float:
        return y1 - (vy - y_min) / y_span * (y1 - y0)

    if title:
        d.text((x0, 4), title, font=font(12, "SemiBold"), fill=_MUTED)
    for i in range(4):  # horizontal gridlines + y ticks
        gy = y_min + y_span * i / 3.0
        yy = py(gy)
        d.line([(x0, yy), (x1, yy)], fill=_HAIR, width=1)
        d.text((6, yy - 6), _fmt(gy), font=font(10), fill=_MUTED)
    for vx in x:  # x ticks
        d.text((px(vx) - 6, y1 + 6), _fmt(vx), font=font(10), fill=_MUTED)
    d.text((x1 - 20, y1 + 6), x_label, font=font(10), fill=_MUTED)

    for _name, ys, color in series:
        pts = [(px(vx), py(vy)) for vx, vy in zip(x, ys, strict=False) if vy is not None]
        if len(pts) >= 2:
            d.line(pts, fill=color, width=3, joint="curve")
        for cx, cy in pts:
            d.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=color, outline=_BG, width=1)

    if projection is not None and x:
        last_series = series[0][1]
        last_val = next((v for v in reversed(last_series) if v is not None), None)
        if last_val is not None:
            _dashed(d, (px(x[-1]), py(last_val)), (px(projection[0]), py(projection[1])),
                    projection[2])
    return img


def stacked_bars(
    size: tuple[int, int],
    labels: list[str],
    stacks: list[list[tuple[Color, float]]],
    *,
    title: str = "",
) -> Image.Image:
    """Per-visit vertical stacked bars (e.g. tissue composition over visits)."""
    w, h = size
    img = Image.new("RGB", size, _BG)
    d = ImageDraw.Draw(img)
    left, right, top, bottom = 14, 14, 26, 26
    x0, y0, x1, y1 = left, top, w - right, h - bottom
    if title:
        d.text((x0, 4), title, font=font(12, "SemiBold"), fill=_MUTED)
    n = max(1, len(stacks))
    slot = (x1 - x0) / n
    bar_w = min(48, slot * 0.55)
    for i, stack in enumerate(stacks):
        cx = x0 + slot * (i + 0.5)
        bx0, bx1 = cx - bar_w / 2, cx + bar_w / 2
        top_y = y1
        d.rounded_rectangle([bx0, y0, bx1, y1], radius=4, fill=_TILE)
        for color, frac in stack:
            seg = (y1 - y0) * frac
            d.rectangle([bx0, top_y - seg, bx1, top_y], fill=color)
            top_y -= seg
        if i < len(labels):
            d.text((cx - _tw(d, labels[i], 10) / 2, y1 + 6), labels[i], font=font(10), fill=_MUTED)
    return img


def _dashed(
    d: ImageDraw.ImageDraw, a: tuple[float, float], b: tuple[float, float], c: Color
) -> None:
    import math

    ax, ay = a
    bx, by = b
    dist = math.hypot(bx - ax, by - ay)
    if dist == 0:
        return
    steps = int(dist // 8)
    for i in range(steps + 1):
        if i % 2:
            continue
        t0, t1 = i / (steps + 1), min(1.0, (i + 1) / (steps + 1))
        d.line([(ax + (bx - ax) * t0, ay + (by - ay) * t0),
                (ax + (bx - ax) * t1, ay + (by - ay) * t1)], fill=c, width=2)


def _tw(d: ImageDraw.ImageDraw, text: str, size: int) -> int:
    left, _, right, _ = d.textbbox((0, 0), text, font=font(size))
    return right - left


def _fmt(v: float) -> str:
    if abs(v) >= 100:
        return f"{v:.0f}"
    if abs(v) >= 1:
        return f"{v:.1f}"
    return f"{v:.2f}"

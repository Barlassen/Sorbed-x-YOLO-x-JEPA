"""Minimal monoline icons drawn directly with Pillow.

Each icon is rendered into a square box so the figures need no icon font or
external assets. Kept deliberately simple and consistent (single stroke weight,
rounded joints) for a clean, modern look.
"""

from __future__ import annotations

from PIL import ImageDraw

Color = tuple[int, int, int]


def draw_icon(
    draw: ImageDraw.ImageDraw, name: str, x: int, y: int, size: int, color: Color, width: int = 2
) -> None:
    """Draw the named icon with its top-left at ``(x, y)`` within a ``size`` box."""
    _ICONS.get(name, _dot)(draw, x, y, size, color, width)


def _image(draw: ImageDraw.ImageDraw, x: int, y: int, s: int, c: Color, w: int) -> None:
    draw.rounded_rectangle([x, y, x + s, y + s], radius=max(2, s // 6), outline=c, width=w)
    draw.ellipse([x + s * 0.18, y + s * 0.18, x + s * 0.36, y + s * 0.36], outline=c, width=w)
    draw.line(
        [(x + s * 0.15, y + s * 0.82), (x + s * 0.45, y + s * 0.5),
         (x + s * 0.65, y + s * 0.68), (x + s * 0.85, y + s * 0.42)],
        fill=c, width=w, joint="curve",
    )


def _mask(draw: ImageDraw.ImageDraw, x: int, y: int, s: int, c: Color, w: int) -> None:
    draw.rounded_rectangle([x, y, x + s, y + s], radius=max(2, s // 6), outline=c, width=w)
    draw.ellipse([x + s * 0.28, y + s * 0.24, x + s * 0.78, y + s * 0.74], fill=c)


def _layers(draw: ImageDraw.ImageDraw, x: int, y: int, s: int, c: Color, w: int) -> None:
    cx = x + s / 2
    for dy in (0.16, 0.4, 0.64):
        top = y + s * dy
        draw.line(
            [(cx, top), (x + s * 0.86, top + s * 0.12), (cx, top + s * 0.24),
             (x + s * 0.14, top + s * 0.12), (cx, top)],
            fill=c, width=w, joint="curve",
        )


def _depth(draw: ImageDraw.ImageDraw, x: int, y: int, s: int, c: Color, w: int) -> None:
    for dy in (0.28, 0.5, 0.72):
        yy = y + s * dy
        left = [x + s * 0.1, yy - s * 0.12, x + s * 0.5, yy + s * 0.12]
        right = [x + s * 0.5, yy - s * 0.12, x + s * 0.9, yy + s * 0.12]
        draw.arc(left, 180, 360, fill=c, width=w)
        draw.arc(right, 0, 180, fill=c, width=w)


def _target(draw: ImageDraw.ImageDraw, x: int, y: int, s: int, c: Color, w: int) -> None:
    m = s * 0.5
    draw.ellipse([x + s * 0.16, y + s * 0.16, x + s * 0.84, y + s * 0.84], outline=c, width=w)
    draw.ellipse([x + s * 0.36, y + s * 0.36, x + s * 0.64, y + s * 0.64], fill=c)
    for a in ((x + m, y), (x + m, y + s), (x, y + m), (x + s, y + m)):
        draw.line([(x + m, y + m), a], fill=c, width=max(1, w - 1))


def _ruler(draw: ImageDraw.ImageDraw, x: int, y: int, s: int, c: Color, w: int) -> None:
    draw.rounded_rectangle(
        [x + s * 0.08, y + s * 0.3, x + s * 0.92, y + s * 0.7], radius=max(2, s // 10),
        outline=c, width=w,
    )
    for i, dx in enumerate((0.25, 0.4, 0.55, 0.7, 0.85)):
        h = 0.62 if i % 2 else 0.5
        draw.line([(x + s * dx, y + s * 0.3), (x + s * dx, y + s * h)], fill=c, width=max(1, w - 1))


def _gauge(draw: ImageDraw.ImageDraw, x: int, y: int, s: int, c: Color, w: int) -> None:
    draw.arc([x + s * 0.12, y + s * 0.22, x + s * 0.88, y + s * 0.98], 180, 360, fill=c, width=w)
    draw.line([(x + s * 0.5, y + s * 0.6), (x + s * 0.72, y + s * 0.36)], fill=c, width=w)
    draw.ellipse([x + s * 0.44, y + s * 0.54, x + s * 0.56, y + s * 0.66], fill=c)


def _droplet(draw: ImageDraw.ImageDraw, x: int, y: int, s: int, c: Color, w: int) -> None:
    cx = x + s / 2
    draw.polygon([(cx, y + s * 0.14), (x + s * 0.78, y + s * 0.58), (x + s * 0.22, y + s * 0.58)],
                 fill=c)
    draw.ellipse([x + s * 0.22, y + s * 0.42, x + s * 0.78, y + s * 0.9], fill=c)


def _spark(draw: ImageDraw.ImageDraw, x: int, y: int, s: int, c: Color, w: int) -> None:
    cx, cy = x + s / 2, y + s / 2
    draw.polygon(
        [(cx, y + s * 0.1), (cx + s * 0.12, cy - s * 0.12), (x + s * 0.9, cy),
         (cx + s * 0.12, cy + s * 0.12), (cx, y + s * 0.9), (cx - s * 0.12, cy + s * 0.12),
         (x + s * 0.1, cy), (cx - s * 0.12, cy - s * 0.12)],
        fill=c,
    )


def _dot(draw: ImageDraw.ImageDraw, x: int, y: int, s: int, c: Color, w: int) -> None:
    draw.ellipse([x + s * 0.3, y + s * 0.3, x + s * 0.7, y + s * 0.7], fill=c)


_ICONS = {
    "image": _image,
    "mask": _mask,
    "layers": _layers,
    "depth": _depth,
    "target": _target,
    "ruler": _ruler,
    "gauge": _gauge,
    "droplet": _droplet,
    "spark": _spark,
}

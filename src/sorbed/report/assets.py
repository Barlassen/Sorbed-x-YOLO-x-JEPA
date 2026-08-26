"""Embed images and fonts as ``data:`` URIs for self-contained reports.

A rendered report must not depend on the network or on sibling files, so every
image, figure, and font is inlined. Chromium then renders and prints from a
single HTML string.
"""

from __future__ import annotations

import base64
import io
from functools import lru_cache
from pathlib import Path

from PIL import Image

_ASSETS = Path(__file__).resolve().parent / "assets"
_FONT = _ASSETS / "fonts" / "Manrope.ttf"


def png_data_uri(image: Image.Image, *, max_side: int | None = None) -> str:
    """Encode a PIL image as a PNG ``data:`` URI, optionally downscaled."""
    img = image.convert("RGB") if image.mode not in ("RGB", "RGBA") else image
    if max_side:
        img = img.copy()
        img.thumbnail((max_side, max_side), Image.Resampling.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    b64 = base64.b64encode(buf.getvalue()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def png_bytes_data_uri(payload: bytes) -> str:
    return f"data:image/png;base64,{base64.b64encode(payload).decode('ascii')}"


def file_image_data_uri(path: str | Path, *, max_side: int | None = 900) -> str:
    """Load an image file and return it as a PNG ``data:`` URI."""
    with Image.open(path) as im:
        return png_data_uri(im, max_side=max_side)


@lru_cache(maxsize=1)
def manrope_font_face() -> str:
    """A ``@font-face`` block embedding the vendored Manrope variable font."""
    if not _FONT.is_file():
        return ""
    b64 = base64.b64encode(_FONT.read_bytes()).decode("ascii")
    return (
        "@font-face{font-family:'Manrope';font-style:normal;"
        "font-weight:200 800;font-display:block;"
        f"src:url(data:font/ttf;base64,{b64}) format('truetype');}}"
    )

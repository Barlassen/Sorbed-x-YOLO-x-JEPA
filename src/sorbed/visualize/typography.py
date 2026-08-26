"""Font loading for the rendered figures.

Ships the Geist typeface (SIL Open Font License, see ``fonts/Geist-OFL.txt``) — a
clean, modern UI face — so every figure has consistent typography regardless of
the host's installed fonts. The weight is driven numerically through the font's
variable ``Weight`` axis (the reliable path in Pillow), and the result falls back
to DejaVu, then Pillow's built-in bitmap font, if Geist cannot be loaded.
"""

from __future__ import annotations

import contextlib
from functools import lru_cache
from pathlib import Path

from PIL import ImageFont

_FONT_PATH = Path(__file__).resolve().parent / "fonts" / "Geist.ttf"

# Named weights → numeric values on the variable Weight axis.
_WEIGHTS = {
    "Light": 300,
    "Regular": 400,
    "Medium": 500,
    "SemiBold": 600,
    "Bold": 700,
    "ExtraBold": 800,
}


@lru_cache(maxsize=128)
def font(size: int, weight: str = "Regular") -> ImageFont.ImageFont:
    """Return the UI font at ``size`` px and the named ``weight``.

    Results are cached, so repeated calls are cheap.
    """
    if _FONT_PATH.is_file():
        try:
            face = ImageFont.truetype(str(_FONT_PATH), size)
            _apply_weight(face, _WEIGHTS.get(weight, 400))
            return face
        except OSError:
            pass
    bold = weight in {"Medium", "SemiBold", "Bold", "ExtraBold"}
    for name in (("DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"), "DejaVuSans.ttf"):
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def _apply_weight(face: ImageFont.ImageFont, weight_value: int) -> None:
    """Set the variable Weight axis numerically; no-op for static fonts."""
    try:
        axes = face.get_variation_axes()
    except OSError:
        return
    if not axes:
        return
    values = []
    for axis in axes:
        name = axis["name"]
        name = name.decode("latin-1") if isinstance(name, bytes) else name
        values.append(weight_value if "eight" in name.lower() else axis["default"])
    with contextlib.suppress(OSError):
        face.set_variation_by_axes(values)

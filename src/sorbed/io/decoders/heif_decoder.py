"""HEIC/HEIF/AVIF decoder (iPhone photos) via pillow-heif.

pillow-heif registers itself as a Pillow plugin, so once available we decode
through the same Pillow path (EXIF orientation, ICC handling) for consistency.
"""

from __future__ import annotations

from sorbed.io.decoders.base import DecodedImage, Decoder, DependencyMissingError
from sorbed.io.decoders.pillow_decoder import PillowDecoder
from sorbed.io.sniff import ImageFormat


class HeifDecoder(Decoder):
    """Decodes the HEIF family by registering the pillow-heif opener."""

    formats = (ImageFormat.HEIF,)

    def __init__(self) -> None:
        self._registered = False

    @property
    def available(self) -> bool:
        try:
            import pillow_heif  # noqa: F401
        except ImportError:
            return False
        return True

    def _ensure_registered(self) -> None:
        if self._registered:
            return
        try:
            import pillow_heif
        except ImportError as exc:
            raise DependencyMissingError(
                "HEIC/HEIF support needs the 'formats' extra: pip install 'sorbed[formats]'"
            ) from exc
        pillow_heif.register_heif_opener()
        self._registered = True

    def decode(self, data: bytes, *, filename: str | None = None) -> DecodedImage:
        self._ensure_registered()
        # With the opener registered, Pillow decodes HEIF natively.
        return PillowDecoder().decode(data, filename=filename)

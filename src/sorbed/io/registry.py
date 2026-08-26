"""Registry mapping detected formats to available decoders."""

from __future__ import annotations

from sorbed.io.decoders import (
    Decoder,
    DicomDecoder,
    HeifDecoder,
    PillowDecoder,
    RawDecoder,
    TiffDecoder,
)
from sorbed.io.sniff import ImageFormat

# Ordered candidates per format. For TIFF, Pillow is tried first (baseline), then
# tifffile for the files Pillow rejects.
_DEFAULT_CANDIDATES: dict[ImageFormat, tuple[Decoder, ...]] = {}


def default_registry() -> dict[ImageFormat, tuple[Decoder, ...]]:
    """Build the default format-to-decoder mapping (fresh instances)."""
    pillow = PillowDecoder()
    heif = HeifDecoder()
    dicom = DicomDecoder()
    tiff = TiffDecoder()
    raw = RawDecoder()
    return {
        ImageFormat.PNG: (pillow,),
        ImageFormat.JPEG: (pillow,),
        ImageFormat.WEBP: (pillow,),
        ImageFormat.BMP: (pillow,),
        ImageFormat.GIF: (pillow,),
        ImageFormat.TIFF: (pillow, tiff),
        ImageFormat.HEIF: (heif,),
        ImageFormat.DICOM: (dicom,),
        ImageFormat.RAW: (raw,),
    }


class DecoderRegistry:
    """Selects a decoder for a detected format, honoring availability."""

    def __init__(self, candidates: dict[ImageFormat, tuple[Decoder, ...]] | None = None) -> None:
        self._candidates = candidates or default_registry()

    def decoder_for(self, fmt: ImageFormat) -> Decoder | None:
        for decoder in self._candidates.get(fmt, ()):
            if decoder.available:
                return decoder
        return None

    def supported_formats(self) -> dict[str, bool]:
        """Map each known format to whether a decoder for it is currently usable."""
        return {
            fmt.value: any(d.available for d in decoders)
            for fmt, decoders in self._candidates.items()
        }

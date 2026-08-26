"""Camera RAW decoder via rawpy/libraw (CR2/CR3, NEF, ARW, DNG, RAF, ...).

Demosaics with fixed, reproducible parameters — camera white balance, no auto
brightness — so repeated runs on the same file yield identical pixels.
"""

from __future__ import annotations

from sorbed.io.decoders.base import DecodedImage, DecodeError, Decoder, DependencyMissingError
from sorbed.io.normalize import ensure_canonical, to_float01
from sorbed.io.sniff import ImageFormat


class RawDecoder(Decoder):
    formats = (ImageFormat.RAW,)

    @property
    def available(self) -> bool:
        try:
            import rawpy  # noqa: F401
        except ImportError:
            return False
        return True

    def decode(self, data: bytes, *, filename: str | None = None) -> DecodedImage:
        try:
            import rawpy
        except ImportError as exc:
            raise DependencyMissingError(
                "Camera RAW support needs the 'formats' extra: pip install 'sorbed[formats]'"
            ) from exc

        from io import BytesIO

        try:
            with rawpy.imread(BytesIO(data)) as raw:
                rgb16 = raw.postprocess(
                    use_camera_wb=True,
                    no_auto_bright=True,
                    output_bps=16,
                )
        except Exception as exc:
            raise DecodeError(f"rawpy could not demosaic RAW: {exc}") from exc

        rgb = ensure_canonical(to_float01(rgb16))
        return DecodedImage(
            rgb=rgb,
            alpha=None,
            bit_depth=16,
            channels=3,
            color_space="sRGB",
        )

"""Pillow-backed decoder for PNG, JPEG, WEBP, BMP, GIF, and baseline TIFF.

Applies EXIF orientation and converts an embedded ICC profile to sRGB so color
metrics (which the tissue classifier depends on) are consistent across sources.
"""

from __future__ import annotations

import io

import numpy as np

from sorbed.io.decoders.base import DecodedImage, DecodeError, Decoder
from sorbed.io.normalize import ensure_canonical, split_alpha, to_float01
from sorbed.io.sniff import ImageFormat

_PILLOW_FORMATS = (
    ImageFormat.PNG,
    ImageFormat.JPEG,
    ImageFormat.WEBP,
    ImageFormat.BMP,
    ImageFormat.GIF,
    ImageFormat.TIFF,
)


class PillowDecoder(Decoder):
    """General raster decoder built on Pillow."""

    formats = _PILLOW_FORMATS

    @property
    def available(self) -> bool:
        try:
            import PIL  # noqa: F401
        except ImportError:
            return False
        return True

    def decode(self, data: bytes, *, filename: str | None = None) -> DecodedImage:
        from PIL import Image, ImageCms, ImageOps

        try:
            image = Image.open(io.BytesIO(data))
            image.load()
        except Exception as exc:  # Pillow raises a variety of errors
            raise DecodeError(f"Pillow could not open image: {exc}") from exc

        # Take the first frame of animated/multipage images.
        if getattr(image, "n_frames", 1) > 1:
            image.seek(0)

        oriented = ImageOps.exif_transpose(image) or image
        exif_applied = oriented is not image

        original_mode = oriented.mode
        bit_depth = 16 if original_mode in {"I;16", "I;16B", "I", "F"} else 8

        icc_converted = False
        icc = oriented.info.get("icc_profile")
        if icc and original_mode in {"RGB", "RGBA"}:
            try:
                src_profile = ImageCms.ImageCmsProfile(io.BytesIO(icc))
                srgb = ImageCms.createProfile("sRGB")
                oriented = ImageCms.profileToProfile(
                    oriented, src_profile, srgb, outputMode=original_mode
                )
                icc_converted = True
            except Exception:  # a bad profile must not fail the decode
                icc_converted = False

        array = np.array(oriented)
        channels = 1 if array.ndim == 2 else array.shape[2]
        rgb_f, alpha = split_alpha(to_float01(array))
        rgb = ensure_canonical(rgb_f)
        alpha_f = alpha.astype(np.float32) if alpha is not None else None

        return DecodedImage(
            rgb=rgb,
            alpha=alpha_f,
            bit_depth=bit_depth,
            channels=channels,
            color_space="sRGB",
            icc_converted=icc_converted,
            exif_oriented=exif_applied,
        )

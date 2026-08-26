"""Advanced TIFF decoder via tifffile (multipage, BigTIFF, 16-bit microscopy).

Pillow handles baseline TIFF; this decoder covers the files Pillow cannot, and is
used as the TIFF path when the 'formats' extra is installed.
"""

from __future__ import annotations

import numpy as np

from sorbed.io.decoders.base import DecodedImage, DecodeError, Decoder, DependencyMissingError
from sorbed.io.normalize import ensure_canonical, split_alpha, to_float01
from sorbed.io.sniff import ImageFormat


class TiffDecoder(Decoder):
    formats = (ImageFormat.TIFF,)

    @property
    def available(self) -> bool:
        try:
            import tifffile  # noqa: F401
        except ImportError:
            return False
        return True

    def decode(self, data: bytes, *, filename: str | None = None) -> DecodedImage:
        try:
            import tifffile
        except ImportError as exc:
            raise DependencyMissingError(
                "Advanced TIFF support needs the 'formats' extra: pip install 'sorbed[formats]'"
            ) from exc

        from io import BytesIO

        try:
            array = tifffile.imread(BytesIO(data))
        except Exception as exc:
            raise DecodeError(f"tifffile could not read TIFF: {exc}") from exc

        # Reduce a multipage/volumetric stack to its first plane.
        while array.ndim > 3:
            array = array[0]
        bit_depth = 16 if array.dtype.itemsize >= 2 else 8
        rgb_f, alpha = split_alpha(to_float01(array))
        rgb = ensure_canonical(rgb_f)
        channels = 1 if array.ndim == 2 else array.shape[-1]

        return DecodedImage(
            rgb=rgb,
            alpha=alpha.astype(np.float32) if alpha is not None else None,
            bit_depth=bit_depth,
            channels=channels,
            color_space="sRGB",
        )

"""Runtime image container.

:class:`RasterImage` carries the decoded pixel buffer through the pipeline. It is
deliberately *not* a pydantic model — the heavy ndarray never lands in a report.
Only its :class:`~sorbed.domain.image.ImageMetadata` and
:class:`~sorbed.domain.image.Calibration` are serialized.

Canonical form: ``pixels`` is float32, shape ``(H, W, 3)``, RGB, range ``[0, 1]``.
An optional ``alpha`` channel (float32 ``(H, W)`` in ``[0, 1]``) records
transparency so it can be treated as "not wound" during segmentation.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, replace

import numpy as np

from sorbed.domain.image import Calibration, ImageMetadata


@dataclass(frozen=True)
class RasterImage:
    """A decoded, normalized image plus its metadata and scale."""

    pixels: np.ndarray  # float32 (H, W, 3), RGB, [0, 1]
    metadata: ImageMetadata
    calibration: Calibration
    alpha: np.ndarray | None = None  # float32 (H, W), [0, 1]

    def __post_init__(self) -> None:
        if self.pixels.ndim != 3 or self.pixels.shape[2] != 3:
            raise ValueError(f"pixels must be (H, W, 3), got {self.pixels.shape}")
        if self.pixels.dtype != np.float32:
            raise ValueError(f"pixels must be float32, got {self.pixels.dtype}")
        if self.alpha is not None and self.alpha.shape != self.pixels.shape[:2]:
            raise ValueError("alpha shape must match (H, W)")

    @property
    def height(self) -> int:
        return int(self.pixels.shape[0])

    @property
    def width(self) -> int:
        return int(self.pixels.shape[1])

    @property
    def shape_hw(self) -> tuple[int, int]:
        return self.height, self.width

    def to_uint8_rgb(self) -> np.ndarray:
        """Return an ``(H, W, 3)`` uint8 RGB view for rendering/encoding."""
        arr = np.clip(self.pixels, 0.0, 1.0) * 255.0
        return arr.round().astype(np.uint8)

    def with_calibration(self, calibration: Calibration) -> RasterImage:
        """Return a copy with a new calibration (pixels shared, not copied)."""
        return replace(self, calibration=calibration)


def pixel_sha256(pixels: np.ndarray) -> str:
    """Stable content hash of a pixel buffer, used for provenance and caching."""
    contiguous = np.ascontiguousarray(pixels)
    digest = hashlib.sha256()
    digest.update(str(contiguous.shape).encode("ascii"))
    digest.update(str(contiguous.dtype).encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def build_metadata(
    *,
    pixels: np.ndarray,
    source_format: str,
    bit_depth: int,
    channels: int,
    color_space: str = "sRGB",
    icc_converted: bool = False,
    exif_oriented: bool = False,
    retained_tags: dict[str, str] | None = None,
) -> ImageMetadata:
    """Assemble :class:`ImageMetadata` from a normalized pixel buffer."""
    height, width = int(pixels.shape[0]), int(pixels.shape[1])
    return ImageMetadata(
        source_format=source_format,
        width_px=width,
        height_px=height,
        channels=channels,
        bit_depth=bit_depth,
        color_space=color_space,
        icc_converted=icc_converted,
        exif_oriented=exif_oriented,
        retained_tags=retained_tags or {},
        pixel_sha256=pixel_sha256(pixels),
    )

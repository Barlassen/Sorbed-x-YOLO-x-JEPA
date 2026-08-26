"""Decoder protocol and the common decode result.

Each decoder turns raw bytes into a :class:`DecodedImage` in canonical form. A
decoder may be *unavailable* when its optional dependency is not installed; the
registry skips unavailable decoders and reports a clear error only if the format
is actually requested.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

import numpy as np

from sorbed.domain.enums import CalibrationStatus
from sorbed.io.sniff import ImageFormat


@dataclass(frozen=True)
class DecodedImage:
    """Canonical decoder output before calibration is resolved."""

    rgb: np.ndarray  # float32 (H, W, 3), [0, 1]
    alpha: np.ndarray | None = None
    bit_depth: int = 8
    channels: int = 3
    color_space: str = "sRGB"
    icc_converted: bool = False
    exif_oriented: bool = False
    retained_tags: dict[str, str] = field(default_factory=dict)
    # A scale hint the decoder could read from the file itself (DICOM only).
    mm_per_px: float | None = None
    calibration_status: CalibrationStatus | None = None


class DecodeError(RuntimeError):
    """Raised when a decoder cannot read the given bytes."""


class DependencyMissingError(RuntimeError):
    """Raised when a decoder's optional dependency is not installed."""


@runtime_checkable
class Decoder(Protocol):
    """Reads a specific family of image formats into a :class:`DecodedImage`."""

    formats: tuple[ImageFormat, ...]

    @property
    def available(self) -> bool:
        """Whether this decoder's dependencies are importable."""
        ...

    def decode(self, data: bytes, *, filename: str | None = None) -> DecodedImage:
        """Decode ``data`` or raise :class:`DecodeError`."""
        ...

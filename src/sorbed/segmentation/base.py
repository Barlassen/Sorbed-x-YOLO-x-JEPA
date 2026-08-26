"""Segmentation contracts.

A :class:`WoundSegmenter` maps a :class:`~sorbed.imaging.RasterImage` to a
:class:`SegmentationResult` — a boolean wound mask plus a confidence derived from
the image itself (never a constant). ML backends implement the same protocol so
they are swappable via configuration.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from sorbed.imaging import RasterImage


@dataclass(frozen=True)
class SegmentationResult:
    """A wound mask and how confident the segmenter is in it."""

    wound_mask: np.ndarray  # bool (H, W); True = wound region
    confidence: float  # [0, 1], computed from image evidence
    backend: str
    weights_sha256: str | None = None

    def __post_init__(self) -> None:
        if self.wound_mask.dtype != np.bool_:
            raise ValueError("wound_mask must be a boolean array")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError(f"confidence out of range: {self.confidence}")

    @property
    def area_px(self) -> int:
        return int(self.wound_mask.sum())


@runtime_checkable
class WoundSegmenter(Protocol):
    name: str

    def segment(self, image: RasterImage) -> SegmentationResult:
        """Return a wound mask for ``image``."""
        ...

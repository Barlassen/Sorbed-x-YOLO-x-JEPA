"""Wound-region segmentation (pluggable backends)."""

from __future__ import annotations

from sorbed.segmentation.base import SegmentationResult, WoundSegmenter
from sorbed.segmentation.classical import ClassicalSegmenter

__all__ = ["ClassicalSegmenter", "SegmentationResult", "WoundSegmenter"]

"""Image ingestion: format detection, decoding, and normalization."""

from __future__ import annotations

from sorbed.io.loader import UnsupportedFormatError, load_image
from sorbed.io.registry import DecoderRegistry
from sorbed.io.sniff import ImageFormat, sniff_format

__all__ = [
    "DecoderRegistry",
    "ImageFormat",
    "UnsupportedFormatError",
    "load_image",
    "sniff_format",
]

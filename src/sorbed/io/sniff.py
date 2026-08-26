"""Detect image container format from content, not filename.

Extensions lie (a ``.jpg`` may be HEIC from a phone). We read the leading bytes
and match known magic signatures. DICOM is special: its magic ``DICM`` sits at
byte offset 128, and many valid DICOM files omit the preamble entirely.
"""

from __future__ import annotations

from enum import StrEnum


class ImageFormat(StrEnum):
    PNG = "png"
    JPEG = "jpeg"
    WEBP = "webp"
    BMP = "bmp"
    GIF = "gif"
    TIFF = "tiff"
    HEIF = "heif"  # HEIC/HEIF/AVIF family
    DICOM = "dicom"
    RAW = "raw"  # camera raw (CR2/NEF/ARW/DNG/...)
    UNKNOWN = "unknown"


# HEIF brand codes that appear in the 'ftyp' box.
_HEIF_BRANDS = {
    b"heic", b"heix", b"hevc", b"hevx", b"mif1", b"msf1", b"heim",
    b"heis", b"avif", b"avis", b"mif2",
}

# libraw-recognized raw signatures (a pragmatic subset).
_RAW_SIGNATURES: tuple[bytes, ...] = (
    b"II\x2a\x00\x10\x00\x00\x00CR",  # Canon CR2
    b"FUJIFILMCCD-RAW",  # Fujifilm RAF
)


def sniff_format(data: bytes, *, filename: str | None = None) -> ImageFormat:
    """Return the detected :class:`ImageFormat` for ``data``.

    ``filename`` is used only as a low-priority tie-breaker for raw formats
    whose signatures overlap with TIFF (DNG/NEF/ARW are TIFF-based).
    """
    if len(data) < 12:
        return ImageFormat.UNKNOWN

    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return ImageFormat.PNG
    if data[:2] == b"\xff\xd8":
        return ImageFormat.JPEG
    if data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return ImageFormat.WEBP
    if data[:2] == b"BM":
        return ImageFormat.BMP
    if data[:6] in (b"GIF87a", b"GIF89a"):
        return ImageFormat.GIF

    if _is_dicom(data):
        return ImageFormat.DICOM

    if data[4:8] == b"ftyp" and data[8:12] in _HEIF_BRANDS:
        return ImageFormat.HEIF

    if any(data.startswith(sig) for sig in _RAW_SIGNATURES):
        return ImageFormat.RAW

    if data[:4] in (b"II\x2a\x00", b"MM\x00\x2a"):
        # TIFF container. Raw formats (DNG/NEF/ARW/ORF) are TIFF-based; use the
        # extension to route them to the raw decoder when it is obviously raw.
        if filename is not None:
            ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
            if ext in {"dng", "nef", "arw", "orf", "rw2", "pef", "srw", "raf", "cr2", "cr3"}:
                return ImageFormat.RAW
        return ImageFormat.TIFF

    return ImageFormat.UNKNOWN


def _is_dicom(data: bytes) -> bool:
    """DICOM has 'DICM' at offset 128 (with preamble) or starts with a valid
    group-0002 element (preamble-less)."""
    if len(data) >= 132 and data[128:132] == b"DICM":
        return True
    # Preamble-less: first bytes are a File Meta element, group (0002).
    return data[:4] in (b"\x02\x00\x00\x00", b"\x08\x00\x00\x00")

"""High-level image loading: bytes/path in, canonical :class:`RasterImage` out."""

from __future__ import annotations

from pathlib import Path

from sorbed.domain.enums import CalibrationStatus
from sorbed.domain.image import Calibration
from sorbed.imaging import RasterImage, build_metadata
from sorbed.io.decoders.base import DecodeError
from sorbed.io.registry import DecoderRegistry
from sorbed.io.sniff import ImageFormat, sniff_format


class UnsupportedFormatError(RuntimeError):
    """Raised when no available decoder can handle the detected format."""


def load_image(
    source: str | Path | bytes,
    *,
    mm_per_px: float | None = None,
    registry: DecoderRegistry | None = None,
) -> RasterImage:
    """Load an image from a path or raw bytes into a :class:`RasterImage`.

    A ``mm_per_px`` override takes precedence over any scale embedded in the file
    and is recorded as a manual calibration. When neither is available the image
    is marked ``UNCALIBRATED`` and physical measurements downstream become
    ``None`` — the scale is never guessed.
    """
    registry = registry or DecoderRegistry()

    filename: str | None
    if isinstance(source, (str, Path)):
        path = Path(source)
        data = path.read_bytes()
        filename = path.name
    else:
        data = source
        filename = None

    fmt = sniff_format(data, filename=filename)
    if fmt is ImageFormat.UNKNOWN:
        raise UnsupportedFormatError("could not detect image format from content")

    decoder = registry.decoder_for(fmt)
    if decoder is None:
        raise UnsupportedFormatError(
            f"no available decoder for format '{fmt.value}'. "
            "Install optional support with: pip install 'sorbed[formats]'"
        )

    try:
        decoded = decoder.decode(data, filename=filename)
    except DecodeError:
        raise
    except Exception as exc:
        raise DecodeError(f"decode failed for format '{fmt.value}': {exc}") from exc

    calibration = _resolve_calibration(mm_per_px, decoded.mm_per_px, decoded.calibration_status)

    metadata = build_metadata(
        pixels=decoded.rgb,
        source_format=fmt.value,
        bit_depth=decoded.bit_depth,
        channels=decoded.channels,
        color_space=decoded.color_space,
        icc_converted=decoded.icc_converted,
        exif_oriented=decoded.exif_oriented,
        retained_tags=decoded.retained_tags,
    )
    return RasterImage(
        pixels=decoded.rgb,
        metadata=metadata,
        calibration=calibration,
        alpha=decoded.alpha,
    )


def _resolve_calibration(
    manual: float | None,
    embedded: float | None,
    embedded_status: CalibrationStatus | None,
) -> Calibration:
    if manual is not None:
        return Calibration(
            mm_per_px=manual,
            status=CalibrationStatus.MANUAL,
            source_detail="mm_per_px provided by caller",
        )
    if embedded is not None and embedded_status is not None:
        return Calibration(
            mm_per_px=embedded,
            status=embedded_status,
            source_detail="scale read from file metadata",
        )
    return Calibration(status=CalibrationStatus.UNCALIBRATED)

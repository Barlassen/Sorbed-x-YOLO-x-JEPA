"""Resolve real-world scale for an image, in strict priority order.

Priority: an already-established calibration (manual override or DICOM pixel
spacing) is kept; otherwise an ArUco marker, then a coin, is tried. If nothing is
found the image is reported ``UNCALIBRATED`` and physical measurements downstream
become ``None`` — the scale is never fabricated.
"""

from __future__ import annotations

from sorbed.domain.enums import CalibrationStatus
from sorbed.domain.image import Calibration
from sorbed.imaging import RasterImage
from sorbed.preprocess.fiducial import detect_aruco_scale, detect_coin_scale


def resolve_calibration(
    image: RasterImage,
    *,
    detect_fiducials: bool = True,
    marker_length_mm: float | None = None,
    coin_diameter_mm: float | None = None,
) -> Calibration:
    """Return the best available :class:`Calibration` for ``image``."""
    if image.calibration.is_calibrated:
        return image.calibration

    if not detect_fiducials:
        return image.calibration

    rgb_u8 = image.to_uint8_rgb()

    if marker_length_mm is not None:
        found = detect_aruco_scale(rgb_u8, marker_length_mm)
        if found is not None:
            mm_per_px, uncertainty = found
            return Calibration(
                mm_per_px=mm_per_px,
                status=CalibrationStatus.FIDUCIAL_MARKER,
                uncertainty_pct=uncertainty,
                source_detail=f"ArUco marker, {marker_length_mm} mm side",
            )

    if coin_diameter_mm is not None:
        found = detect_coin_scale(rgb_u8, coin_diameter_mm)
        if found is not None:
            mm_per_px, uncertainty = found
            return Calibration(
                mm_per_px=mm_per_px,
                status=CalibrationStatus.RULER_DETECTED,
                uncertainty_pct=uncertainty,
                source_detail=f"coin, {coin_diameter_mm} mm diameter",
            )

    return Calibration(status=CalibrationStatus.UNCALIBRATED)

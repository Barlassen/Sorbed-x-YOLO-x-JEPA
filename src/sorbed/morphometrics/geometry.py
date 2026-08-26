"""Wound geometry from a binary mask, calibrated to physical units when possible.

Length and width follow the clinical convention (greatest head-to-toe extent x
greatest perpendicular extent) when a head direction is supplied; otherwise they
fall back to the region's major/minor axes and ``axis_convention`` records that.
Physical measurements are emitted only when the image is calibrated.
"""

from __future__ import annotations

import numpy as np
from skimage import measure

from sorbed.domain.image import Calibration
from sorbed.domain.metrics import GeometryMetrics


def measure_geometry(
    wound_mask: np.ndarray,
    calibration: Calibration,
    *,
    head_vector: tuple[float, float] | None = None,
) -> GeometryMetrics:
    """Compute :class:`GeometryMetrics` for ``wound_mask``.

    ``head_vector`` is a ``(dy, dx)`` unit vector pointing toward the patient's
    head; when given, length is measured along it and width perpendicular.
    """
    total_px = int(wound_mask.size)
    if not wound_mask.any():
        return _empty_geometry(total_px)

    labeled = measure.label(wound_mask.astype(np.uint8))
    props = max(measure.regionprops(labeled), key=lambda p: p.area)

    area_px = float(props.area)
    perimeter_px = float(props.perimeter)
    if perimeter_px <= 0:
        perimeter_px = _fallback_perimeter(wound_mask)
    major = float(props.axis_major_length)
    minor = float(props.axis_minor_length)
    cy, cx = props.centroid
    minr, minc, maxr, maxc = props.bbox
    solidity = float(props.solidity) if props.solidity == props.solidity else 1.0

    circularity = 0.0
    if perimeter_px > 0:
        circularity = min(1.0, 4.0 * np.pi * area_px / (perimeter_px**2))

    length_px, width_px, convention = _length_width(wound_mask, props, head_vector)

    geom = GeometryMetrics(
        area_px=area_px,
        perimeter_px=perimeter_px,
        length_px=length_px,
        width_px=width_px,
        major_axis_px=major,
        minor_axis_px=minor,
        circularity=round(circularity, 4),
        solidity=round(solidity, 4),
        centroid_px=(round(float(cy), 2), round(float(cx), 2)),
        bbox_px=(int(minr), int(minc), int(maxr), int(maxc)),
        axis_convention=convention,
        wound_fraction_of_image=round(area_px / total_px, 6),
        area_mm2=_r(calibration.to_mm2(area_px)),
        length_mm=_r(calibration.to_mm(length_px)),
        width_mm=_r(calibration.to_mm(width_px)),
        area_cm2=_r(_to_cm2(calibration.to_mm2(area_px))),
    )
    return geom


def _length_width(
    mask: np.ndarray,
    props: object,
    head_vector: tuple[float, float] | None,
) -> tuple[float, float, str]:
    if head_vector is None:
        return (
            float(props.axis_major_length),
            float(props.axis_minor_length),
            "major_minor_axes",
        )
    dy, dx = head_vector
    norm = float(np.hypot(dy, dx)) or 1.0
    uy, ux = dy / norm, dx / norm
    ys, xs = np.where(mask)
    coords = np.stack([ys, xs], axis=1).astype(np.float64)
    along = coords[:, 0] * uy + coords[:, 1] * ux
    perp = coords[:, 0] * (-ux) + coords[:, 1] * uy
    length = float(along.max() - along.min())
    width = float(perp.max() - perp.min())
    return length, width, "clinical_head_to_toe"


def _fallback_perimeter(mask: np.ndarray) -> float:
    contours = measure.find_contours(mask.astype(float), 0.5)
    if not contours:
        return 0.0
    contour = max(contours, key=len)
    diffs = np.diff(contour, axis=0)
    return float(np.hypot(diffs[:, 0], diffs[:, 1]).sum())


def _to_cm2(mm2: float | None) -> float | None:
    return None if mm2 is None else mm2 / 100.0


def _r(value: float | None) -> float | None:
    return None if value is None else round(value, 4)


def _empty_geometry(total_px: int) -> GeometryMetrics:
    return GeometryMetrics(
        area_px=0.0,
        perimeter_px=0.0,
        length_px=0.0,
        width_px=0.0,
        major_axis_px=0.0,
        minor_axis_px=0.0,
        circularity=0.0,
        solidity=0.0,
        centroid_px=(0.0, 0.0),
        bbox_px=(0, 0, 0, 0),
        axis_convention="none",
        wound_fraction_of_image=0.0,
    )

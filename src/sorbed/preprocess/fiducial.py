"""Detect a scale reference in the image to recover real-world size.

Two real detectors are provided, both computing millimeters-per-pixel from a
reference object of *known physical size*:

* an ArUco fiducial marker (most reliable — print one and place it beside the
  wound), and
* a coin of known diameter (a common improvised ruler).

If neither is found the image stays uncalibrated; a scale is never guessed.
"""

from __future__ import annotations

import cv2
import numpy as np


def detect_aruco_scale(rgb_u8: np.ndarray, marker_length_mm: float) -> tuple[float, float] | None:
    """Return ``(mm_per_px, uncertainty_pct)`` from an ArUco marker, or ``None``."""
    gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)
    corners = _detect_aruco_corners(gray)
    if not corners:
        return None
    # Average the four side lengths of the first marker (in pixels).
    pts = corners[0].reshape(4, 2)
    sides = [float(np.linalg.norm(pts[i] - pts[(i + 1) % 4])) for i in range(4)]
    side_px = float(np.mean(sides))
    if side_px <= 0:
        return None
    mm_per_px = marker_length_mm / side_px
    uncertainty = float(np.std(sides) / (np.mean(sides) + 1e-6) * 100.0)
    return mm_per_px, round(uncertainty, 2)


def detect_coin_scale(rgb_u8: np.ndarray, coin_diameter_mm: float) -> tuple[float, float] | None:
    """Return ``(mm_per_px, uncertainty_pct)`` from a detected coin, or ``None``."""
    gray = cv2.cvtColor(rgb_u8, cv2.COLOR_RGB2GRAY)
    gray = cv2.medianBlur(gray, 5)
    h, w = gray.shape
    circles = cv2.HoughCircles(
        gray,
        cv2.HOUGH_GRADIENT,
        dp=1.5,
        minDist=min(h, w) / 2,
        param1=120,
        param2=60,
        minRadius=int(0.03 * min(h, w)),
        maxRadius=int(0.4 * min(h, w)),
    )
    if circles is None:
        return None
    circles = np.round(circles[0]).astype(int)
    radius_px = float(sorted(circles, key=lambda c: c[2], reverse=True)[0][2])
    if radius_px <= 0:
        return None
    mm_per_px = coin_diameter_mm / (2.0 * radius_px)
    # Hough radius is approximate; report a fixed, honest uncertainty band.
    return mm_per_px, 8.0


def _detect_aruco_corners(gray: np.ndarray) -> list[np.ndarray]:
    aruco = getattr(cv2, "aruco", None)
    if aruco is None:
        return []
    dictionary = aruco.getPredefinedDictionary(aruco.DICT_4X4_50)
    # OpenCV changed the ArUco API across versions; support both.
    if hasattr(aruco, "ArucoDetector"):
        detector = aruco.ArucoDetector(dictionary, aruco.DetectorParameters())
        corners, ids, _ = detector.detectMarkers(gray)
    else:  # pragma: no cover - legacy OpenCV
        corners, ids, _ = aruco.detectMarkers(gray, dictionary)
    return list(corners) if ids is not None and len(corners) else []

"""Geometry, physical scaling, healing sub-scores, and depth cues."""

from __future__ import annotations

from sorbed.morphometrics.depth_proxy import relative_depth_field, shading_depth_proxy
from sorbed.morphometrics.geometry import measure_geometry
from sorbed.morphometrics.periwound import analyze_periwound
from sorbed.morphometrics.scales import compute_healing_scores

__all__ = [
    "analyze_periwound",
    "compute_healing_scores",
    "measure_geometry",
    "relative_depth_field",
    "shading_depth_proxy",
]

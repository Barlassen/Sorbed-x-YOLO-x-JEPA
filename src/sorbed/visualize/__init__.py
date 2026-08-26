"""Rendering: wound masks, tissue overlays, and annotated guide images."""

from __future__ import annotations

from sorbed.visualize.dashboard import render_dashboard, render_depth_overlay
from sorbed.visualize.detection import render_detection
from sorbed.visualize.guide import render_guide, render_schematic_guide
from sorbed.visualize.overlay import render_mask, render_schematic, render_tissue_overlay
from sorbed.visualize.trend import render_trend_dashboard

__all__ = [
    "render_dashboard",
    "render_depth_overlay",
    "render_detection",
    "render_guide",
    "render_mask",
    "render_schematic",
    "render_schematic_guide",
    "render_tissue_overlay",
    "render_trend_dashboard",
]

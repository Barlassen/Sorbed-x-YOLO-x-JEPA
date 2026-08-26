"""Render pipeline rasters to base64 PNG data URIs for the API/web UI."""

from __future__ import annotations

import base64
import io

from PIL import Image

from sorbed.morphometrics import relative_depth_field
from sorbed.pipeline.analyzer import AnalysisBundle
from sorbed.visualize.dashboard import render_depth_overlay
from sorbed.visualize.detection import render_detection
from sorbed.visualize.overlay import render_mask, render_schematic, render_tissue_overlay


def _data_uri(image: Image.Image) -> str:
    buf = io.BytesIO()
    image.convert("RGB").save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode("ascii")


def rendered_panels(bundle: AnalysisBundle) -> dict[str, str]:
    """Return the per-panel images (input/mask/overlay/detection/depth/schematic)
    as PNG data URIs, ready to drop into ``<img src=...>``."""
    display = bundle.display_image.to_uint8_rgb()
    depth_field = relative_depth_field(bundle.display_image.pixels, bundle.wound_mask)
    return {
        "input": _data_uri(Image.fromarray(display, mode="RGB")),
        "mask": _data_uri(render_mask(bundle.wound_mask)),
        "overlay": _data_uri(
            render_tissue_overlay(display, bundle.tissue_label_map, bundle.wound_mask)
        ),
        "detection": _data_uri(render_detection(bundle.analysis, display)),
        "depth": _data_uri(render_depth_overlay(display, depth_field, bundle.wound_mask)),
        "schematic": _data_uri(
            render_schematic(bundle.tissue_label_map, bundle.wound_mask)
        ),
    }


def detection_thumb(bundle: AnalysisBundle) -> str:
    """A single detection-annotated thumbnail data URI (for trend filmstrips)."""
    display = bundle.display_image.to_uint8_rgb()
    return _data_uri(render_detection(bundle.analysis, display))

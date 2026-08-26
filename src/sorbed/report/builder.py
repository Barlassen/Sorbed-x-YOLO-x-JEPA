"""Assemble and write all report artifacts for one analysis."""

from __future__ import annotations

import hashlib
import io
from pathlib import Path

from PIL import Image

from sorbed.domain.report import Report, ReportArtifact
from sorbed.morphometrics import relative_depth_field
from sorbed.pipeline.analyzer import AnalysisBundle
from sorbed.report.html_report import build_html
from sorbed.report.json_report import write_json
from sorbed.visualize.dashboard import render_dashboard, render_depth_overlay
from sorbed.visualize.detection import render_detection
from sorbed.visualize.guide import render_guide, render_schematic_guide
from sorbed.visualize.overlay import render_mask, render_schematic, render_tissue_overlay

DEFAULT_ARTIFACTS = (
    "json", "mask", "overlay", "detection", "depth", "guide", "schematic",
    "schematic_guide", "dashboard", "html",
)


def write_report(
    bundle: AnalysisBundle,
    out_dir: str | Path,
    *,
    artifacts: tuple[str, ...] = DEFAULT_ARTIFACTS,
    stem: str | None = None,
) -> Report:
    """Render and write the requested artifacts, returning a :class:`Report`."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    analysis = bundle.analysis
    stem = stem or f"sorbed_{str(analysis.analysis_id)[:8]}"

    display_u8 = bundle.display_image.to_uint8_rgb()
    written: list[ReportArtifact] = []
    guide_png: bytes | None = None

    if "json" in artifacts:
        written.append(write_json(analysis, out / f"{stem}.json"))

    if "mask" in artifacts:
        written.append(
            _write_png(render_mask(bundle.wound_mask), out / f"{stem}_mask.png", "mask_png")
        )

    if "overlay" in artifacts:
        overlay = render_tissue_overlay(display_u8, bundle.tissue_label_map, bundle.wound_mask)
        written.append(_write_png(overlay, out / f"{stem}_overlay.png", "overlay_png"))

    if "detection" in artifacts:
        detection = render_detection(analysis, display_u8)
        written.append(_write_png(detection, out / f"{stem}_detection.png", "detection_png"))

    depth_field = None
    if "depth" in artifacts or "dashboard" in artifacts:
        depth_field = relative_depth_field(bundle.display_image.pixels, bundle.wound_mask)

    if "depth" in artifacts:
        depth_img = render_depth_overlay(display_u8, depth_field, bundle.wound_mask)
        written.append(_write_png(depth_img, out / f"{stem}_depth.png", "depth_png"))

    if "dashboard" in artifacts:
        dash = render_dashboard(
            analysis, display_u8, bundle.wound_mask, bundle.tissue_label_map, depth_field
        )
        written.append(_write_png(dash, out / f"{stem}_dashboard.png", "dashboard_png"))

    if "schematic" in artifacts:
        schematic = render_schematic(bundle.tissue_label_map, bundle.wound_mask)
        written.append(_write_png(schematic, out / f"{stem}_schematic.png", "schematic_png"))

    if "schematic_guide" in artifacts:
        sguide = render_schematic_guide(
            bundle.analysis, bundle.wound_mask, bundle.tissue_label_map
        )
        written.append(
            _write_artifact(
                out / f"{stem}_schematic_guide.png", _png_bytes(sguide), "schematic_guide_png",
                "image/png",
            )
        )

    if "guide" in artifacts or "html" in artifacts:
        guide = render_guide(analysis, display_u8, bundle.wound_mask, bundle.tissue_label_map)
        guide_png = _png_bytes(guide)
        if "guide" in artifacts:
            written.append(
                _write_artifact(out / f"{stem}_guide.png", guide_png, "guide_png", "image/png")
            )

    if "html" in artifacts:
        payload = build_html(analysis, guide_png).encode("utf-8")
        written.append(
            _write_artifact(out / f"{stem}.html", payload, "html", "text/html")
        )

    return Report(analysis=analysis, artifacts=written)


def _png_bytes(image: Image.Image) -> bytes:
    buf = io.BytesIO()
    image.save(buf, format="PNG")
    return buf.getvalue()


def _write_png(image: Image.Image, path: Path, kind: str) -> ReportArtifact:
    return _write_artifact(path, _png_bytes(image), kind, "image/png")


def _write_artifact(path: Path, payload: bytes, kind: str, mime: str) -> ReportArtifact:
    path.write_bytes(payload)
    return ReportArtifact(
        kind=kind,
        path=str(path),
        mime=mime,
        sha256=hashlib.sha256(payload).hexdigest(),
        bytes=len(payload),
    )

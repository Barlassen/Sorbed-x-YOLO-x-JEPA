"""End-to-end pipeline and report-writing tests."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from sorbed.domain.analysis import WoundAnalysis
from sorbed.pipeline import AnalyzeOptions, analyze_image
from sorbed.report.builder import write_report
from tests.synth import GRANULATION, SLOUGH, blank_image, make_wound

try:
    import io

    from PIL import Image
except ImportError:  # pragma: no cover
    Image = None


def test_full_pipeline_grades_granulating_wound():
    wound = make_wound(fill=GRANULATION, inner=((28, 20), SLOUGH), seed=8)
    bundle = analyze_image(wound.to_png_bytes(), options=AnalyzeOptions(mm_per_px=0.2))
    a = bundle.analysis
    assert a.decision.stage.value == "stage_3"
    assert a.metrics.geometry.area_cm2 is not None
    assert a.metrics.tissue.dominant.value == "granulation"
    assert a.stage_timings_ms  # real per-stage timings recorded


def test_analysis_json_round_trips():
    wound = make_wound(seed=9)
    a = analyze_image(wound.to_png_bytes(), options=AnalyzeOptions(mm_per_px=0.2)).analysis
    restored = WoundAnalysis.model_validate_json(a.model_dump_json())
    assert restored.analysis_id == a.analysis_id
    assert restored.decision.stage == a.decision.stage


def test_analysis_validates_against_exported_schema():
    wound = make_wound(seed=10)
    a = analyze_image(wound.to_png_bytes()).analysis
    schema = WoundAnalysis.model_json_schema()
    assert schema["title"] == "WoundAnalysis"
    # A dumped analysis is a dict with all required top-level keys present.
    dumped = json.loads(a.model_dump_json())
    for key in ("analysis_id", "metrics", "decision", "calibration", "provenance"):
        assert key in dumped


def test_blank_image_abstains():
    assert Image is not None
    buf = io.BytesIO()
    Image.fromarray((blank_image() * 255).astype("uint8")).save(buf, "PNG")
    a = analyze_image(buf.getvalue()).analysis
    assert a.decision.abstained
    assert a.decision.stage.value == "indeterminate"


def test_write_report_produces_artifacts_with_matching_hashes(tmp_path: Path):
    wound = make_wound(fill=GRANULATION, inner=((28, 20), SLOUGH), seed=11)
    bundle = analyze_image(wound.to_png_bytes(), options=AnalyzeOptions(mm_per_px=0.2))
    report = write_report(bundle, tmp_path)
    kinds = {art.kind for art in report.artifacts}
    assert {
        "json", "mask_png", "overlay_png", "detection_png", "depth_png",
        "guide_png", "schematic_png", "dashboard_png", "html",
    } <= kinds
    for art in report.artifacts:
        data = Path(art.path).read_bytes()
        assert len(data) == art.bytes > 0
        assert hashlib.sha256(data).hexdigest() == art.sha256


def test_uncalibrated_report_has_null_physical_sizes():
    wound = make_wound(seed=12)
    a = analyze_image(wound.to_png_bytes()).analysis  # no scale
    assert a.calibration.mm_per_px is None
    assert a.metrics.geometry.area_mm2 is None
    assert a.metrics.geometry.area_px > 0  # pixel metrics still computed


@pytest.mark.parametrize("fmt", ["PNG", "JPEG"])
def test_pipeline_accepts_multiple_formats(fmt: str):
    wound = make_wound(seed=13)
    data = wound.to_png_bytes() if fmt == "PNG" else wound.to_jpeg_bytes()
    a = analyze_image(data, options=AnalyzeOptions(mm_per_px=0.2)).analysis
    assert a.image.source_format == fmt.lower()
    assert a.decision.stage is not None

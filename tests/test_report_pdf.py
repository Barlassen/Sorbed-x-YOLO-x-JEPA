"""Report HTML builders, guideline matching, and healing alerts.

These exercise the PDF report layer up to the HTML string (the Chromium print
step is not invoked here), plus the directive-free degradation path and the
alert logic that drives the follow-up verdict.
"""

from __future__ import annotations

from sorbed.domain.enums import PressureInjuryStage, TissueClass
from sorbed.guidelines import build_guideline_context
from sorbed.report import svg
from sorbed.report.templates import (
    build_followup_report_html,
    build_grading_report_html,
    build_summary_report_html,
)
from sorbed.trend import compute_trend
from sorbed.trend.alerts import assess_healing
from sorbed.trend.models import WoundTimePoint


def _pt(day: float, area: float, stage: PressureInjuryStage, gran: float,
        nec: float = 0.0, push: int | None = None) -> WoundTimePoint:
    return WoundTimePoint(
        label=f"visit {int(day)}", day=day, stage=stage, confidence=0.8,
        area_px=area * 100, area_cm2=area, perimeter_px=200.0, perimeter_mm=40.0,
        tissue_fractions={TissueClass.GRANULATION: gran, TissueClass.SLOUGH: nec},
        push_total=push,
    )


def test_grading_report_html_without_pack(granulating_analysis) -> None:
    ctx = build_guideline_context(granulating_analysis, None)
    assert ctx.available is False
    html = build_grading_report_html(
        analysis=granulating_analysis, guideline_ctx=ctx, images={}, patient_ref="PID-1"
    )
    assert html.startswith("<!doctype html>")
    assert "Sorbed" in html
    assert "PID-1" in html
    # Stage label is rendered somewhere in the hero.
    assert "Stage" in html or "Evre" in html


def test_summary_report_html(granulating_analysis) -> None:
    ctx = build_guideline_context(granulating_analysis, None)
    html = build_summary_report_html(
        analysis=granulating_analysis, guideline_ctx=ctx, images={}, patient_ref="PID-2"
    )
    assert html.startswith("<!doctype html>")
    assert "Analiz özeti · Analysis summary" in html
    assert "PID-2" in html


def test_followup_report_html_and_verdict() -> None:
    pts = [
        _pt(0, 4.0, PressureInjuryStage.STAGE_3, gran=0.4, nec=0.4, push=12),
        _pt(14, 2.5, PressureInjuryStage.STAGE_3, gran=0.6, nec=0.2, push=9),
        _pt(28, 1.0, PressureInjuryStage.STAGE_3, gran=0.8, nec=0.05, push=6),
    ]
    trend = compute_trend(pts, patient_ref="PID-9")
    assessment = assess_healing(trend)
    assert assessment.level == "positive"
    assert any(a.metric == "area" for a in assessment.alerts)
    html = build_followup_report_html(
        trend=trend, assessment=assessment,
        guideline_ctx=build_guideline_context_stub(),
        visit_thumbs={}, patient_ref="PID-9",
    )
    assert "PID-9" in html
    assert "Patient improving" in html
    # An SVG chart is inlined, not an <img>.
    assert "<svg" in html


def test_deteriorating_verdict() -> None:
    pts = [
        _pt(0, 1.0, PressureInjuryStage.STAGE_2, gran=0.7, nec=0.05, push=6),
        _pt(14, 2.0, PressureInjuryStage.STAGE_3, gran=0.3, nec=0.5, push=10),
    ]
    trend = compute_trend(pts)
    assessment = assess_healing(trend)
    assert assessment.level == "critical"
    # Enlarging area, rising necrosis, PUSH up, and stage progression should fire.
    kinds = {a.metric for a in assessment.alerts}
    assert "area" in kinds
    assert "stage" in kinds


def test_svg_helpers_return_svg() -> None:
    assert svg.line_chart_svg([0, 1, 2], [3.0, 2.0, 1.0], accent="#4F46E5").startswith("<svg")
    assert svg.gauge_svg(72.0, accent="#16A34A").startswith("<svg")
    rows = [("Evre 3", 0.7, "#DB4A4F"), ("Evre 2", 0.2, "#E4711C")]
    assert "svg" in svg.hbars_svg(rows)


def build_guideline_context_stub():
    # A no-pack context: report renders without directive grounding.
    from sorbed.guidelines.match import GuidelineContext

    return GuidelineContext(available=False)

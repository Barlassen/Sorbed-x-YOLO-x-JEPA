"""Tests for the longitudinal healing-trend engine."""

from __future__ import annotations

import pytest

from sorbed.domain.enums import PressureInjuryStage, TissueClass
from sorbed.trend import compute_trend
from sorbed.trend.models import WoundTimePoint


def _pt(day: float, area_cm2: float | None, area_px: float, push: int | None = None,
        gran: float = 0.8) -> WoundTimePoint:
    return WoundTimePoint(
        label=f"Day {day:g}",
        day=day,
        stage=PressureInjuryStage.STAGE_3,
        confidence=0.8,
        area_px=area_px,
        area_cm2=area_cm2,
        perimeter_px=100.0,
        perimeter_mm=12.0 if area_cm2 is not None else None,
        tissue_fractions={TissueClass.GRANULATION: gran, TissueClass.SLOUGH: 1 - gran},
        push_total=push,
    )


def test_healing_series_trajectory_and_par():
    pts = [_pt(0, 4.0, 4000), _pt(14, 2.6, 2600), _pt(28, 1.3, 1300), _pt(42, 0.5, 500)]
    trend = compute_trend(pts, patient_ref="demo")
    assert trend.trajectory == "healing"
    assert trend.percent_area_reduction == pytest.approx(87.5, abs=0.1)  # (4-0.5)/4
    assert trend.healing_rate_per_week is not None and trend.healing_rate_per_week < 0
    assert trend.projected_days_to_closure is not None
    assert trend.par_at_4_weeks is not None
    assert trend.is_calibrated
    assert len(trend.intervals) == 3


def test_deteriorating_series():
    pts = [_pt(0, 2.0, 2000), _pt(14, 3.0, 3000), _pt(28, 4.5, 4500)]
    trend = compute_trend(pts)
    assert trend.trajectory == "deteriorating"
    assert trend.percent_area_reduction < 0  # area grew
    assert trend.projected_days_to_closure is None  # never closes


def test_four_week_par_threshold_flags_likely_to_heal():
    # >50% reduction by day 28 -> above the 40% threshold.
    pts = [_pt(0, 10.0, 10000), _pt(28, 4.0, 4000)]
    trend = compute_trend(pts)
    assert trend.par_at_4_weeks == pytest.approx(60.0, abs=1.0)
    assert trend.likely_to_heal is True


def test_push_trend_improving():
    pts = [_pt(0, 4.0, 4000, push=12), _pt(28, 1.0, 1000, push=6)]
    trend = compute_trend(pts)
    assert trend.push_trend == "improving"


def test_uncalibrated_series_uses_pixels():
    pts = [_pt(0, None, 4000), _pt(28, None, 1000)]
    trend = compute_trend(pts)
    assert trend.unit == "px"
    assert not trend.is_calibrated
    assert trend.baseline_area == 4000
    assert any("pixel" in n.lower() for n in trend.notes)


def test_single_point_is_indeterminate():
    trend = compute_trend([_pt(0, 4.0, 4000)])
    assert trend.trajectory == "indeterminate"
    assert trend.intervals == []


def test_gilman_edge_advance_positive_when_healing():
    pts = [_pt(0, 4.0, 4000), _pt(7, 2.0, 2000)]
    trend = compute_trend(pts)
    assert trend.intervals[0].edge_advance_mm_per_day is not None
    assert trend.intervals[0].edge_advance_mm_per_day > 0  # shrinking => positive advance


def test_empty_series_raises():
    with pytest.raises(ValueError):
        compute_trend([])

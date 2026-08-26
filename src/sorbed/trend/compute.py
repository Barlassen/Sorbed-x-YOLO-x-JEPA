"""Compute a :class:`HealingTrend` from a series of per-visit analyses."""

from __future__ import annotations

from itertools import pairwise

import numpy as np

from sorbed.domain.analysis import WoundAnalysis
from sorbed.domain.enums import TissueClass
from sorbed.trend.models import HealingTrend, IntervalDelta, WoundTimePoint

# Validated 4-week percent-area-reduction threshold predicting eventual healing.
_PAR_HEAL_THRESHOLD = 40.0
# Weekly area-change magnitude (% of baseline) separating healing/stalled/worsening.
_STALL_BAND_PCT_PER_WEEK = 5.0


def timepoint_from_analysis(analysis: WoundAnalysis, *, day: float, label: str) -> WoundTimePoint:
    """Build a :class:`WoundTimePoint` from a single-image analysis."""
    g = analysis.metrics.geometry
    mm = analysis.calibration.mm_per_px
    hs = analysis.metrics.healing_scores
    return WoundTimePoint(
        label=label,
        day=day,
        stage=analysis.decision.stage,
        confidence=analysis.decision.confidence,
        area_px=g.area_px,
        area_cm2=g.area_cm2,
        perimeter_px=g.perimeter_px,
        perimeter_mm=(g.perimeter_px * mm) if mm else None,
        tissue_fractions=dict(analysis.metrics.tissue.fractions),
        push_total=hs.push_partial_total if hs else None,
        analysis_id=analysis.analysis_id,
    )


def compute_trend(
    points: list[WoundTimePoint], *, patient_ref: str | None = None
) -> HealingTrend:
    """Assemble the healing trend and summary statistics from ordered visits."""
    if not points:
        raise ValueError("a healing trend needs at least one time point")
    pts = sorted(points, key=lambda p: p.day)
    calibrated = all(p.area_cm2 is not None for p in pts)
    unit = "cm2" if calibrated else "px"

    def area(p: WoundTimePoint) -> float:
        return float(p.area_cm2) if calibrated and p.area_cm2 is not None else p.area_px

    baseline, latest = area(pts[0]), area(pts[-1])
    par = ((baseline - latest) / baseline * 100.0) if baseline > 0 else 0.0

    slope = intercept = None
    rate_week = rate_pct_week = par_4wk = projected = None
    if len(pts) >= 2 and pts[-1].day > pts[0].day:
        days = np.array([p.day for p in pts], dtype=float)
        areas = np.array([area(p) for p in pts], dtype=float)
        slope, intercept = (float(v) for v in np.polyfit(days, areas, 1))
        rate_week = slope * 7.0
        rate_pct_week = (rate_week / baseline * 100.0) if baseline > 0 else None
        area_28 = slope * 28.0 + intercept
        par_4wk = ((baseline - area_28) / baseline * 100.0) if baseline > 0 else None
        if slope < 0:
            closure_day = -intercept / slope
            if closure_day > pts[-1].day:
                projected = round(closure_day - pts[-1].day, 1)

    trajectory = _trajectory(rate_pct_week, len(pts))
    likely = None if par_4wk is None else bool(par_4wk >= _PAR_HEAL_THRESHOLD)
    push_trend = _push_trend(pts)

    return HealingTrend(
        patient_ref=patient_ref,
        unit=unit,
        points=pts,
        intervals=[_interval(a, b, calibrated) for a, b in pairwise(pts)],
        baseline_area=round(baseline, 4),
        latest_area=round(latest, 4),
        percent_area_reduction=round(par, 2),
        healing_rate_per_week=None if rate_week is None else round(rate_week, 4),
        healing_rate_pct_per_week=None if rate_pct_week is None else round(rate_pct_week, 2),
        par_at_4_weeks=None if par_4wk is None else round(par_4wk, 2),
        likely_to_heal=likely,
        projected_days_to_closure=projected,
        trajectory=trajectory,
        push_trend=push_trend,
        notes=_notes(unit, pts),
    )


def _trajectory(rate_pct_week: float | None, n: int) -> str:
    if n < 2 or rate_pct_week is None:
        return "indeterminate"
    if rate_pct_week <= -_STALL_BAND_PCT_PER_WEEK:  # area shrinking
        return "healing"
    if rate_pct_week >= _STALL_BAND_PCT_PER_WEEK:  # area growing
        return "deteriorating"
    return "stalled"


def _push_trend(pts: list[WoundTimePoint]) -> str | None:
    scored = [p for p in pts if p.push_total is not None]
    if len(scored) < 2:
        return None
    delta = scored[-1].push_total - scored[0].push_total  # type: ignore[operator]
    if delta <= -1:
        return "improving"
    if delta >= 1:
        return "worsening"
    return "unchanged"


def _interval(a: WoundTimePoint, b: WoundTimePoint, calibrated: bool) -> IntervalDelta:
    days = b.day - a.day
    a_area = a.area_cm2 if calibrated and a.area_cm2 is not None else a.area_px
    b_area = b.area_cm2 if calibrated and b.area_cm2 is not None else b.area_px
    pct = ((b_area - a_area) / a_area * 100.0) if a_area > 0 else 0.0

    edge = None
    if calibrated and a.area_cm2 and b.area_cm2 and a.perimeter_mm and b.perimeter_mm and days > 0:
        mean_perim = (a.perimeter_mm + b.perimeter_mm) / 2.0
        if mean_perim > 0:
            # Gilman: (area shrinkage in mm^2) / mean perimeter / days = mm/day edge advance.
            edge = round((a.area_cm2 - b.area_cm2) * 100.0 / mean_perim / days, 4)

    return IntervalDelta(
        from_label=a.label,
        to_label=b.label,
        days_elapsed=days,
        area_delta_cm2=(round(b.area_cm2 - a.area_cm2, 3) if calibrated and a.area_cm2 is not None
                        and b.area_cm2 is not None else None),
        area_delta_px=round(b.area_px - a.area_px, 1),
        area_pct_change=round(pct, 2),
        push_delta=(b.push_total - a.push_total if a.push_total is not None
                    and b.push_total is not None else None),
        granulation_delta=round(b.fraction(TissueClass.GRANULATION)
                                - a.fraction(TissueClass.GRANULATION), 4),
        slough_delta=round(b.fraction(TissueClass.SLOUGH) - a.fraction(TissueClass.SLOUGH), 4),
        eschar_delta=round(b.fraction(TissueClass.ESCHAR) - a.fraction(TissueClass.ESCHAR), 4),
        edge_advance_mm_per_day=edge,
        trajectory="improving" if pct < -2 else "worsening" if pct > 2 else "stalled",
    )


def _notes(unit: str, pts: list[WoundTimePoint]) -> list[str]:
    notes = [
        "Trends assume consistent imaging technique, distance, and lighting across visits; "
        "inconsistent capture is the dominant source of error."
    ]
    if unit == "px":
        notes.append(
            "Sizes are in pixels because at least one visit was uncalibrated — provide a scale "
            "(mm-per-pixel, ruler, or marker) at every visit for physical trend units."
        )
    span = pts[-1].day - pts[0].day
    if len(pts) >= 2 and span < 7:
        notes.append(f"Short observation window ({span:.0f} days); projections are low-confidence.")
    return notes

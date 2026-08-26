"""Data contracts for longitudinal (multi-visit) wound tracking.

Metrics follow the validated wound-healing literature (see docs/TREND.md):
percent area reduction, healing velocity, the perimeter-normalized (Gilman) edge
advance, and the 4-week percent-area-reduction predictor of eventual healing.
Every value is computed from the per-visit analyses; nothing is invented, and
sizes are reported in cm² only when every visit is calibrated (else in pixels).
"""

from __future__ import annotations

from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from sorbed.domain.enums import PressureInjuryStage, TissueClass


class WoundTimePoint(BaseModel):
    """One visit: the wound's state at a point in time."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    label: str
    day: float = Field(description="Days since the baseline visit (day 0).")
    stage: PressureInjuryStage
    confidence: float = Field(ge=0, le=1)
    area_px: float = Field(ge=0)
    area_cm2: float | None = None
    perimeter_px: float = Field(ge=0)
    perimeter_mm: float | None = None
    tissue_fractions: dict[TissueClass, float] = Field(default_factory=dict)
    push_total: int | None = None
    analysis_id: UUID | None = None

    def fraction(self, cls: TissueClass) -> float:
        return self.tissue_fractions.get(cls, 0.0)


class IntervalDelta(BaseModel):
    """Change between two consecutive visits."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    from_label: str
    to_label: str
    days_elapsed: float
    area_delta_cm2: float | None = None
    area_delta_px: float
    area_pct_change: float = Field(description="Signed % change vs. the earlier visit.")
    push_delta: int | None = None
    granulation_delta: float
    slough_delta: float
    eschar_delta: float
    edge_advance_mm_per_day: float | None = Field(
        default=None,
        description="Perimeter-normalized (Gilman) healing rate; positive = healing.",
    )
    trajectory: str = Field(description="improving | stalled | worsening")


class HealingTrend(BaseModel):
    """The full trajectory across all visits, with summary statistics."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    patient_ref: str | None = None
    unit: str = Field(description="'cm2' when every visit is calibrated, else 'px'.")
    points: list[WoundTimePoint]
    intervals: list[IntervalDelta]

    baseline_area: float
    latest_area: float
    percent_area_reduction: float = Field(
        description="PAR from baseline to latest, in % (positive = shrinking)."
    )
    healing_rate_per_week: float | None = Field(
        default=None, description="Linear-fit area change per week (unit/week; negative = healing)."
    )
    healing_rate_pct_per_week: float | None = None
    par_at_4_weeks: float | None = Field(
        default=None, description="PAR projected/observed at 28 days."
    )
    likely_to_heal: bool | None = Field(
        default=None,
        description="True when 4-week PAR meets the validated ~40-50% threshold.",
    )
    projected_days_to_closure: float | None = None
    trajectory: str = Field(description="healing | stalled | deteriorating | indeterminate")
    push_trend: str | None = None
    notes: list[str] = Field(default_factory=list)

    @property
    def is_calibrated(self) -> bool:
        return self.unit == "cm2"

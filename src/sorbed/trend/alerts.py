"""Derive clinical *alerts* from a healing trend.

Alerts translate the quantitative trajectory into the directive's own decision
language: a decreasing PUSH total and rising granulation mean healing (§4.5.8,
§4.5.9); growing area, rising slough/eschar, or a worsening stage mean the
wound is deteriorating and the plan must be escalated. Each alert carries the
directive section(s) that justify it, so a report can footnote every call.

Nothing here invents data — alerts are pure functions of a :class:`HealingTrend`
that was itself computed from per-visit model outputs.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from sorbed.domain.enums import PressureInjuryStage, TissueClass
from sorbed.trend.models import HealingTrend, WoundTimePoint

# Alert severities, ordered most→least urgent for banner selection.
LEVELS = ("critical", "warning", "positive", "info")

# Stage ordinals for detecting progression to a worse stage. Non-graded outcomes
# map to None and are excluded from progression checks.
_STAGE_ORDER: dict[PressureInjuryStage, int] = {
    PressureInjuryStage.STAGE_1: 1,
    PressureInjuryStage.STAGE_2: 2,
    PressureInjuryStage.STAGE_3: 3,
    PressureInjuryStage.STAGE_4: 4,
    PressureInjuryStage.DEEP_TISSUE: 3,  # DTI may evolve to a full-thickness loss
    PressureInjuryStage.UNSTAGEABLE: 4,  # obscured full-thickness
}

# Devitalized-tissue increase (fraction points) that warrants a debridement flag.
_NECROSIS_RISE = 0.03
# Granulation increase (fraction points) worth surfacing as a positive signal.
_GRANULATION_RISE = 0.03


class HealingAlert(BaseModel):
    """One actionable signal about the wound's trajectory."""

    model_config = ConfigDict(frozen=True)

    level: str  # one of LEVELS
    title: str
    detail: str
    directive_refs: tuple[str, ...] = ()
    metric: str | None = None


class HealingAssessment(BaseModel):
    """The overall follow-up verdict plus its supporting alerts."""

    model_config = ConfigDict(frozen=True)

    level: str
    headline: str
    summary: str
    alerts: tuple[HealingAlert, ...] = ()

    @property
    def is_worsening(self) -> bool:
        return self.level in ("critical", "warning")


def _stage_progressed(pts: list[WoundTimePoint]) -> tuple[bool, WoundTimePoint | None]:
    base = _STAGE_ORDER.get(pts[0].stage)
    worst_pt = None
    worst = base
    for p in pts[1:]:
        o = _STAGE_ORDER.get(p.stage)
        if o is not None and (worst is None or o > worst):
            worst, worst_pt = o, p
    if base is None or worst is None:
        return False, None
    return worst > base, worst_pt


def _necrosis_fraction(p: WoundTimePoint) -> float:
    return p.fraction(TissueClass.SLOUGH) + p.fraction(TissueClass.ESCHAR)


def assess_healing(trend: HealingTrend) -> HealingAssessment:
    """Produce the overall verdict and the ranked alerts behind it."""
    pts = trend.points
    alerts: list[HealingAlert] = []

    # --- Size / PAR (§4.5.11 size; validated 4-week PAR predictor) ---
    if trend.percent_area_reduction >= 5:
        alerts.append(HealingAlert(
            level="positive", metric="area",
            title="Wound surface shrinking",
            detail=(f"{trend.percent_area_reduction:.0f}% area reduction from baseline "
                    f"({trend.baseline_area:.3g}→{trend.latest_area:.3g} {trend.unit})."),
            directive_refs=("4.5.11",),
        ))
    elif trend.percent_area_reduction <= -5:
        alerts.append(HealingAlert(
            level="critical", metric="area",
            title="Wound surface enlarging",
            detail=(f"Area grew {abs(trend.percent_area_reduction):.0f}% from baseline "
                    f"({trend.baseline_area:.3g}→{trend.latest_area:.3g} {trend.unit}). "
                    "Escalate and review the prevention/care plan."),
            directive_refs=("4.5.11", "4.6.10"),
        ))

    if trend.likely_to_heal is False and trend.par_at_4_weeks is not None:
        alerts.append(HealingAlert(
            level="warning", metric="par_4wk",
            title="Below the 4-week healing threshold",
            detail=(f"Projected 4-week area reduction is {trend.par_at_4_weeks:.0f}%, under the "
                    "~40% mark that predicts eventual closure. Reassess the treatment strategy."),
            directive_refs=("4.6.10",),
        ))

    # --- PUSH total (§3.11, §4.5.8: falling total = healing) ---
    if trend.push_trend == "improving":
        alerts.append(HealingAlert(
            level="positive", metric="push",
            title="PUSH total falling",
            detail="The PUSH score is decreasing across visits, indicating the wound is healing.",
            directive_refs=("4.5.8", "3.11"),
        ))
    elif trend.push_trend == "worsening":
        alerts.append(HealingAlert(
            level="critical", metric="push",
            title="PUSH total rising",
            detail="The PUSH score is increasing across visits, indicating deterioration.",
            directive_refs=("4.5.8", "3.11"),
        ))

    # --- Tissue composition (§4.5.9 RYB) ---
    gran_delta = (pts[-1].fraction(TissueClass.GRANULATION)
                  - pts[0].fraction(TissueClass.GRANULATION))
    if gran_delta >= _GRANULATION_RISE:
        alerts.append(HealingAlert(
            level="positive", metric="granulation",
            title="Granulation increasing",
            detail=(f"Red granulation/epithelial tissue rose {gran_delta * 100:.0f} points — the "
                    "desired healing tissue (§4.5.9, kırmızı yara)."),
            directive_refs=("4.5.9",),
        ))
    necrosis_delta = _necrosis_fraction(pts[-1]) - _necrosis_fraction(pts[0])
    if necrosis_delta >= _NECROSIS_RISE:
        alerts.append(HealingAlert(
            level="warning", metric="necrosis",
            title="Devitalized tissue increasing",
            detail=(f"Yellow slough / black eschar rose {necrosis_delta * 100:.0f} points. Per "
                    "§4.5.9, black necrosis is debrided and yellow slough signals infection to "
                    "treat; notify the wound-care nurse."),
            directive_refs=("4.5.9", "3.13", "4.6.3"),
        ))

    # --- Stage progression ---
    progressed, worst_pt = _stage_progressed(pts)
    if progressed and worst_pt is not None:
        alerts.append(HealingAlert(
            level="critical", metric="stage",
            title="Stage progression",
            detail=(f"Graded stage advanced to {worst_pt.stage.value.replace('_', ' ')} at "
                    f"'{worst_pt.label}'. A deeper stage is never back-staged; escalate care."),
            directive_refs=("3.5", "3.6", "4.6.10"),
        ))

    alerts.sort(key=lambda a: LEVELS.index(a.level))
    level, headline, summary = _verdict(trend, alerts)
    return HealingAssessment(level=level, headline=headline, summary=summary,
                             alerts=tuple(alerts))


def _verdict(
    trend: HealingTrend, alerts: list[HealingAlert]
) -> tuple[str, str, str]:
    """Roll the trajectory + alerts into a single banner."""
    has_critical = any(a.level == "critical" for a in alerts)
    has_warning = any(a.level == "warning" for a in alerts)

    if trend.trajectory == "deteriorating" or has_critical:
        return ("critical", "Patient deteriorating",
                "One or more indicators show the wound is worsening. Escalate to the wound-care "
                "nurse and review the prevention and treatment plan.")
    if trend.trajectory == "stalled" or has_warning:
        return ("warning", "Healing stalled",
                "The wound is not progressing as expected. Reassess dressing choice, offloading, "
                "nutrition, and comorbid factors.")
    if trend.trajectory == "healing":
        return ("positive", "Patient improving",
                "Size, tissue quality, and/or PUSH score are all trending toward healing. Continue "
                "the current plan and keep monitoring.")
    return ("info", "Insufficient trend",
            "Not enough visits or time have elapsed to establish a reliable trajectory. Continue "
            "scheduled reassessment.")

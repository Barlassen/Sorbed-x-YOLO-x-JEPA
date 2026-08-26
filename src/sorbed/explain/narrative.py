"""Deterministic, template-based explanation of a staging decision.

No language model is involved: the narrative is assembled from the decision's own
fields so it is reproducible and cannot drift from the evidence. Every quantity
it states is a field in the analysis.
"""

from __future__ import annotations

from sorbed.domain.decision import StageDecision
from sorbed.domain.enums import PressureInjuryStage, TissueClass
from sorbed.domain.metrics import Metrics

_STAGE_LABEL: dict[PressureInjuryStage, str] = {
    PressureInjuryStage.STAGE_1: "Stage 1 (non-blanchable erythema, intact skin)",
    PressureInjuryStage.STAGE_2: "Stage 2 (partial-thickness skin loss)",
    PressureInjuryStage.STAGE_3: "Stage 3 (full-thickness skin loss)",
    PressureInjuryStage.STAGE_4: "Stage 4 (full-thickness tissue loss)",
    PressureInjuryStage.UNSTAGEABLE: "Unstageable (base obscured)",
    PressureInjuryStage.DEEP_TISSUE: "Deep Tissue Pressure Injury",
    PressureInjuryStage.MUCOSAL: "Mucosal membrane (not stageable)",
    PressureInjuryStage.NOT_PRESSURE_INJURY: "Not a pressure injury",
    PressureInjuryStage.INDETERMINATE: "Indeterminate",
}


def build_narrative(decision: StageDecision, metrics: Metrics) -> str:
    """Compose a clinician-readable summary of the grade and its basis."""
    label = _STAGE_LABEL.get(decision.stage, decision.stage.value)
    lead = (
        f"Provisional grade: {label}, confidence {decision.confidence:.2f}."
        if not decision.abstained
        else f"Provisional grade withheld ({label}); the evidence was insufficient "
        f"for a confident grade (confidence {decision.confidence:.2f})."
    )

    parts = [lead]

    if decision.evidence:
        top = decision.evidence[0]
        parts.append(f"Primary basis: {top.description}")

    parts.append(_size_sentence(metrics))
    parts.append(_tissue_sentence(metrics))

    if decision.caveats:
        parts.append("Caveats: " + " ".join(c.message for c in decision.caveats))

    if decision.requires_clinician_review:
        parts.append("Clinician review is required before this informs care.")

    return " ".join(p for p in parts if p)


def _size_sentence(metrics: Metrics) -> str:
    geom = metrics.geometry
    if geom.area_cm2 is not None:
        return (
            f"Wound area {geom.area_cm2:.1f} cm^2 "
            f"({geom.length_mm:.0f} x {geom.width_mm:.0f} mm)."
        )
    return (
        f"Wound area {geom.area_px:.0f} px "
        f"({geom.length_px:.0f} x {geom.width_px:.0f} px); no scale reference detected."
    )


def _tissue_sentence(metrics: Metrics) -> str:
    fr = metrics.tissue.fractions
    ordered = sorted(fr.items(), key=lambda kv: kv[1], reverse=True)
    named = [
        f"{cls.value.replace('_', ' ')} {frac * 100:.0f}%"
        for cls, frac in ordered
        if frac >= 0.02 and cls is not TissueClass.BACKGROUND
    ]
    return "Tissue composition: " + ", ".join(named) + "." if named else ""

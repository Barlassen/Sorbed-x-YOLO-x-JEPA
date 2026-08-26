"""Combine the rule-based stage and an optional ML stage into one decision.

Safety-critical rule outcomes (Unstageable, exposed-structure Stage 4, the
Indeterminate guard, and Deep Tissue Injury) always override the ML head — you
cannot stage what you cannot see, and those calls must not be softened by a model.
Otherwise agreement boosts confidence, and genuine disagreement lowers it and
adds a caveat rather than silently trusting either side.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from sorbed.domain.decision import Evidence
from sorbed.domain.enums import EvidenceDirection, PressureInjuryStage
from sorbed.staging.ml_head import MLStagePrediction

# Rule outcomes that the ML head may never override.
_HARD_RULE_STAGES = frozenset(
    {
        PressureInjuryStage.UNSTAGEABLE,
        PressureInjuryStage.STAGE_4,
        PressureInjuryStage.DEEP_TISSUE,
        PressureInjuryStage.INDETERMINATE,
    }
)
_ML_LEAD_MIN_PROB = 0.6


@dataclass(frozen=True)
class ArbiterResult:
    stage: PressureInjuryStage
    confidence: float
    arbitration: str
    agreement: bool
    ml_evidence: list[Evidence] = field(default_factory=list)
    caveat: str | None = None


def arbitrate(
    rule_stage: PressureInjuryStage,
    rule_confidence: float,
    ml: MLStagePrediction,
) -> ArbiterResult:
    """Reconcile the rule stage with the ML prediction."""
    ml_prob = ml.probabilities.get(ml.stage, 0.0)
    agreement = rule_stage == ml.stage
    evidence = _ml_evidence(ml)

    if rule_stage in _HARD_RULE_STAGES:
        return ArbiterResult(
            stage=rule_stage,
            confidence=rule_confidence,
            arbitration="rule_override",
            agreement=agreement,
            ml_evidence=evidence,
            caveat=None
            if agreement
            else "Rule safety-override; the ML head suggested a different grade.",
        )

    if agreement:
        boosted = min(1.0, rule_confidence + 0.1 * ml_prob)
        return ArbiterResult(
            stage=rule_stage,
            confidence=round(boosted, 4),
            arbitration="consensus",
            agreement=True,
            ml_evidence=evidence,
        )

    if ml_prob >= _ML_LEAD_MIN_PROB:
        return ArbiterResult(
            stage=ml.stage,
            confidence=round(min(rule_confidence, ml_prob) * 0.9, 4),
            arbitration="ml_lead",
            agreement=False,
            ml_evidence=evidence,
            caveat="Rule and ML head disagreed; the more confident ML grade was taken. "
            "Clinician review is advised.",
        )

    return ArbiterResult(
        stage=rule_stage,
        confidence=round(rule_confidence * 0.75, 4),
        arbitration="rule_lead_low_agreement",
        agreement=False,
        ml_evidence=evidence,
        caveat="Rule and ML head disagreed; the rule grade was kept at reduced confidence.",
    )


def _ml_evidence(ml: MLStagePrediction) -> list[Evidence]:
    return [
        Evidence(
            code=f"ml_feature_{name}",
            description=f"ML head weighted the '{name.replace('_', ' ')}' feature "
            f"(contribution {contribution:.3f}).",
            metric_ref="decision.ml_stage_probabilities",
            observed=round(contribution, 4),
            threshold=None,
            comparison="attribution",
            direction=EvidenceDirection.NEUTRAL,
            weight=float(contribution),
            source="ml:gbm",
        )
        for name, contribution in ml.top_features
    ]

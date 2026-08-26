"""The rule-based staging engine and its arbiter.

Evaluates every clinical rule against the feature vector, selects the highest-
priority firing, and produces a :class:`StageDecision` with a confidence computed
from segmentation quality and evidence margin, the full rule-firing trace, and
safety caveats. It deliberately abstains (``INDETERMINATE``) rather than emit a
low-confidence depth-dependent grade, and it lowers confidence and warns on
darker skin, where early-stage and deep-tissue injuries are harder to detect.

A learned staging head can be supplied to the arbiter; when absent the decision
is transparently rule-only.
"""

from __future__ import annotations

from sorbed.domain.decision import Caveat, RuleFiring, StageDecision
from sorbed.domain.enums import PressureInjuryStage, Severity, SkinToneBand
from sorbed.staging.arbiter import arbitrate
from sorbed.staging.features import FeatureVector
from sorbed.staging.ml_head import GBMStagingHead
from sorbed.staging.rules import RULES, Thresholds

# Confidence is dampened for grades whose determination needs depth.
_DEPTH_DEPENDENT_DAMPING = 0.85


class StagingEngine:
    """Explainable pressure-injury staging."""

    name = "rule_engine"

    def __init__(
        self,
        thresholds: Thresholds | None = None,
        *,
        low_confidence_threshold: float = 0.5,
        ml_head: GBMStagingHead | None = None,
    ) -> None:
        self._t = thresholds or Thresholds()
        self._low_confidence = low_confidence_threshold
        self._ml_head = ml_head

    def decide(
        self,
        features: FeatureVector,
        *,
        skin_tone: SkinToneBand = SkinToneBand.UNKNOWN,
        is_calibrated: bool = False,
    ) -> StageDecision:
        firings = self._evaluate(features)
        fired = [f for f in firings if f.fired]

        if not fired:
            return self._indeterminate(features, firings, skin_tone, is_calibrated,
                                       reason="No clinical rule matched the observed features.")

        winner = max(fired, key=lambda f: f.priority)
        stage = winner.implied_stage or PressureInjuryStage.INDETERMINATE
        confidence = self._confidence(stage, features, winner)

        # Optional learned head, reconciled by the arbiter. Rule-only by default.
        ml_stage: PressureInjuryStage | None = None
        ml_probs: dict[PressureInjuryStage, float] = {}
        agreement: bool | None = None
        arbitration = "rule_only"
        ml_evidence: list = []
        ml_caveat: str | None = None
        if self._ml_head is not None:
            ml_pred = self._ml_head.predict(features)
            result = arbitrate(stage, confidence, ml_pred)
            stage = result.stage
            confidence = result.confidence
            arbitration = result.arbitration
            agreement = result.agreement
            ml_stage = ml_pred.stage
            ml_probs = ml_pred.probabilities
            ml_evidence = result.ml_evidence
            ml_caveat = result.caveat

        caveats = self._caveats(stage, features, skin_tone, is_calibrated)
        if ml_caveat:
            caveats.append(Caveat(severity=Severity.WARNING, message=ml_caveat))
        confidence, abstained = self._maybe_abstain(stage, confidence)
        if abstained:
            stage = PressureInjuryStage.INDETERMINATE

        evidence = list(winner.evidence)
        for other in fired:
            if other is not winner:
                evidence.extend(other.evidence)
        evidence.extend(ml_evidence)

        return StageDecision(
            stage=stage,
            confidence=round(confidence, 4),
            abstained=abstained,
            rule_stage=winner.implied_stage,
            ml_stage=ml_stage,
            ml_stage_probabilities=ml_probs,
            agreement=agreement,
            arbitration=arbitration,
            evidence=evidence,
            rule_firings=firings,
            caveats=caveats,
        )

    def _evaluate(self, features: FeatureVector) -> list[RuleFiring]:
        firings: list[RuleFiring] = []
        for rule in RULES:
            fired = rule.predicate(features, self._t)
            firings.append(
                RuleFiring(
                    rule_id=rule.id,
                    fired=fired,
                    implied_stage=rule.implied_stage if fired else None,
                    priority=rule.priority,
                    evidence=rule.evidence(features, self._t) if fired else [],
                )
            )
        return firings

    def _confidence(
        self, stage: PressureInjuryStage, features: FeatureVector, winner: RuleFiring
    ) -> float:
        base = features.seg_confidence
        # Evidence margin: how strongly the winning evidence clears its threshold.
        margin = 0.6
        for ev in winner.evidence:
            # bool is an int subclass, so test categorical evidence first.
            if isinstance(ev.observed, bool):
                if ev.observed:  # a directly-observed categorical finding
                    margin = max(margin, 0.9)
            elif isinstance(ev.observed, (int, float)) and isinstance(ev.threshold, (int, float)):
                spread = abs(float(ev.observed) - float(ev.threshold))
                margin = max(margin, min(1.0, 0.6 + spread))
        conf = base * margin
        if stage.is_depth_dependent:
            conf *= _DEPTH_DEPENDENT_DAMPING
        return float(max(0.0, min(1.0, conf)))

    def _caveats(
        self,
        stage: PressureInjuryStage,
        features: FeatureVector,
        skin_tone: SkinToneBand,
        is_calibrated: bool,
    ) -> list[Caveat]:
        caveats: list[Caveat] = []
        if not is_calibrated:
            caveats.append(
                Caveat(
                    severity=Severity.INFO,
                    message="No scale reference was found: sizes are reported in pixels only. "
                    "Provide mm-per-pixel, a ruler, or a fiducial marker for physical units.",
                )
            )
        if stage.is_depth_dependent:
            caveats.append(
                Caveat(
                    severity=Severity.WARNING,
                    message="This grade depends on tissue depth, which a 2D photograph cannot "
                    "confirm. In-person probing and clinician confirmation are required.",
                )
            )
        if skin_tone is SkinToneBand.IV_VI and stage in {
            PressureInjuryStage.STAGE_1,
            PressureInjuryStage.DEEP_TISSUE,
            PressureInjuryStage.INDETERMINATE,
            PressureInjuryStage.NOT_PRESSURE_INJURY,
        }:
            caveats.append(
                Caveat(
                    severity=Severity.CRITICAL,
                    message="Skin tone appears darker (Fitzpatrick IV-VI), where early-stage and "
                    "deep-tissue injuries are markedly harder to detect from color. A negative or "
                    "low-stage result does NOT rule out injury — assess temperature, firmness, and "
                    "edema, and lower the threshold for clinician review.",
                )
            )
        return caveats

    def _maybe_abstain(
        self, stage: PressureInjuryStage, confidence: float
    ) -> tuple[float, bool]:
        if stage is PressureInjuryStage.INDETERMINATE:
            return confidence, True
        # Unstageable and Stage 4 are direct visual findings (obscured base /
        # exposed structure); only the depth-*inferred* grades abstain when weak.
        inferred = {PressureInjuryStage.STAGE_3, PressureInjuryStage.DEEP_TISSUE}
        if confidence < self._low_confidence and stage in inferred:
            return confidence, True
        return confidence, False

    def _indeterminate(
        self,
        features: FeatureVector,
        firings: list[RuleFiring],
        skin_tone: SkinToneBand,
        is_calibrated: bool,
        *,
        reason: str,
    ) -> StageDecision:
        caveats = self._caveats(
            PressureInjuryStage.INDETERMINATE, features, skin_tone, is_calibrated
        )
        caveats.insert(0, Caveat(severity=Severity.WARNING, message=reason))
        return StageDecision(
            stage=PressureInjuryStage.INDETERMINATE,
            confidence=round(min(0.4, 1.0 - features.seg_confidence), 4),
            abstained=True,
            rule_stage=None,
            arbitration="rule_only",
            evidence=[],
            rule_firings=firings,
            caveats=caveats,
        )

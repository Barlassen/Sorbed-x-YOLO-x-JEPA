"""The grading decision and its explainable evidence trace.

``StageDecision`` is the auditable output of the staging engine. Every grade is
accompanied by a ranked list of :class:`Evidence`, each item pointing at a real
metric field, so a clinician can see exactly why a stage was assigned.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from sorbed.domain.enums import EvidenceDirection, PressureInjuryStage, Severity


class Evidence(BaseModel):
    """One atom of the explanation: an observed value compared to a threshold.

    ``metric_ref`` is a dotted path into the :class:`~sorbed.domain.analysis.
    WoundAnalysis` (e.g. ``metrics.tissue.fractions.eschar``). A contract test
    verifies every ``metric_ref`` resolves to a real field, which structurally
    forbids explanations that cite metrics the system does not actually produce.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    code: str = Field(description="Stable identifier, e.g. 'eschar_obscures_base'.")
    description: str
    metric_ref: str
    observed: float | str | bool | None = None
    threshold: float | str | bool | None = None
    comparison: str = Field(default="n/a")
    direction: EvidenceDirection = EvidenceDirection.SUPPORTS
    weight: float = Field(ge=0, description="Contribution magnitude (rule weight or |SHAP|).")
    source: str = Field(description="Origin, e.g. 'rule:R_UNSTAGEABLE' or 'ml:gbm'.")


class RuleFiring(BaseModel):
    """Record of a single clinical rule evaluated against the feature vector."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    rule_id: str
    fired: bool
    implied_stage: PressureInjuryStage | None = None
    priority: int = 0
    evidence: list[Evidence] = Field(default_factory=list)


class Caveat(BaseModel):
    """A limitation attached to the decision, surfaced prominently to the user."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    severity: Severity
    message: str


class StageDecision(BaseModel):
    """The provisional grade, its confidence, and the full evidence trace."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    stage: PressureInjuryStage
    confidence: float = Field(ge=0, le=1)
    abstained: bool = False

    rule_stage: PressureInjuryStage | None = None
    ml_stage: PressureInjuryStage | None = None
    ml_stage_probabilities: dict[PressureInjuryStage, float] = Field(default_factory=dict)
    agreement: bool | None = None
    arbitration: str = Field(default="rule_only")

    evidence: list[Evidence] = Field(default_factory=list)
    rule_firings: list[RuleFiring] = Field(default_factory=list)
    caveats: list[Caveat] = Field(default_factory=list)

    narrative: str = Field(default="", description="Deterministic human-readable summary.")

    @property
    def requires_clinician_review(self) -> bool:
        """True when the grade is inherently uncertain and must be confirmed."""
        return (
            self.abstained
            or self.confidence < 0.5
            or self.stage.is_depth_dependent
            or self.stage
            in {
                PressureInjuryStage.UNSTAGEABLE,
                PressureInjuryStage.DEEP_TISSUE,
                PressureInjuryStage.INDETERMINATE,
            }
        )

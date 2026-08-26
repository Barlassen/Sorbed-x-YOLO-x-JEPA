"""Declarative clinical staging rules (NPIAP 2016 / International 2019).

Each rule is data: a predicate over the :class:`FeatureVector`, the stage it
implies, a priority, and an evidence builder. The engine evaluates all rules and
the highest-priority firing wins; every firing keeps its evidence so the full
trace is auditable. The logic mirrors the decision tree in ``docs/CLINICAL.md``.

Thresholds are conservative and configurable. The engine biases toward
``UNSTAGEABLE``/``DEEP_TISSUE``/``INDETERMINATE`` (defer to clinician) rather than
over-claiming a depth-dependent grade from a 2D image.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from sorbed.domain.decision import Evidence
from sorbed.domain.enums import EvidenceDirection, PressureInjuryStage
from sorbed.staging.features import FeatureVector

# Default thresholds (see docs/CLINICAL.md).
OBSCURED_UNSTAGEABLE = 0.5
ERYTHEMA_MIN = 0.15
MAROON_MIN = 0.15
DEPTH_STAGE3 = 0.35
DEVITALIZED_MAX_STAGE2 = 0.1
SEG_CONF_MIN = 0.2
MIN_WOUND_FRACTION = 0.001


@dataclass(frozen=True)
class Rule:
    id: str
    priority: int
    implied_stage: PressureInjuryStage
    description: str
    predicate: Callable[[FeatureVector, Thresholds], bool]
    evidence: Callable[[FeatureVector, Thresholds], list[Evidence]]


@dataclass(frozen=True)
class Thresholds:
    obscured_unstageable: float = OBSCURED_UNSTAGEABLE
    erythema_min: float = ERYTHEMA_MIN
    maroon_min: float = MAROON_MIN
    depth_stage3: float = DEPTH_STAGE3
    devitalized_max_stage2: float = DEVITALIZED_MAX_STAGE2
    seg_conf_min: float = SEG_CONF_MIN
    min_wound_fraction: float = MIN_WOUND_FRACTION


def _ev(
    code: str,
    description: str,
    metric_ref: str,
    observed: float | str | bool | None,
    threshold: float | str | bool | None,
    comparison: str,
    rule_id: str,
    *,
    direction: EvidenceDirection = EvidenceDirection.SUPPORTS,
    weight: float = 1.0,
) -> Evidence:
    return Evidence(
        code=code,
        description=description,
        metric_ref=metric_ref,
        observed=observed,
        threshold=threshold,
        comparison=comparison,
        direction=direction,
        weight=weight,
        source=f"rule:{rule_id}",
    )


# --- Guard: too little evidence to grade ------------------------------------
def _indeterminate_pred(f: FeatureVector, t: Thresholds) -> bool:
    return f.seg_confidence < t.seg_conf_min or f.wound_fraction_of_image < t.min_wound_fraction


def _indeterminate_ev(f: FeatureVector, t: Thresholds) -> list[Evidence]:
    return [
        _ev(
            "insufficient_segmentation",
            "The wound could not be localized with enough confidence to grade.",
            "decision.rule_firings",
            round(f.seg_confidence, 3),
            t.seg_conf_min,
            "<",
            "R_INDETERMINATE",
        )
    ]


# --- Unstageable ------------------------------------------------------------
def _unstageable_pred(f: FeatureVector, t: Thresholds) -> bool:
    return (not f.skin_intact) and f.obscured_fraction >= t.obscured_unstageable


def _unstageable_ev(f: FeatureVector, t: Thresholds) -> list[Evidence]:
    return [
        _ev(
            "base_obscured",
            f"{f.obscured_fraction * 100:.0f}% of the bed is covered by slough/eschar, "
            "so the wound base — and therefore its depth — cannot be seen.",
            "metrics.tissue.obscured_fraction",
            round(f.obscured_fraction, 3),
            t.obscured_unstageable,
            ">=",
            "R_UNSTAGEABLE",
        )
    ]


# --- Stage 4 ----------------------------------------------------------------
def _stage4_pred(f: FeatureVector, t: Thresholds) -> bool:
    return (not f.skin_intact) and (f.muscle > 0.0 or f.tendon_bone > 0.0)


def _stage4_ev(f: FeatureVector, t: Thresholds) -> list[Evidence]:
    return [
        _ev(
            "deep_structure_exposed",
            "Muscle, tendon, or bone appears exposed, which defines Stage 4.",
            "metrics.tissue.has_deep_structure",
            True,
            True,
            "==",
            "R_STAGE_4",
        )
    ]


# --- Deep Tissue Pressure Injury -------------------------------------------
def _dti_pred(f: FeatureVector, t: Thresholds) -> bool:
    return f.skin_intact and f.maroon_purple >= t.maroon_min


def _dti_ev(f: FeatureVector, t: Thresholds) -> list[Evidence]:
    return [
        _ev(
            "maroon_purple_intact_skin",
            f"{f.maroon_purple * 100:.0f}% of the area is deep-red/maroon/purple over "
            "seemingly intact skin — the hallmark of deep-tissue injury.",
            "metrics.color_cues.maroon_purple_fraction",
            round(f.maroon_purple, 3),
            t.maroon_min,
            ">=",
            "R_DTI",
        ),
        _ev(
            "skin_intact",
            "The skin appears intact rather than an open wound bed.",
            "metrics.skin_intact",
            True,
            True,
            "==",
            "R_DTI",
        ),
    ]


# --- Stage 3 ----------------------------------------------------------------
# Clinically, granulation tissue forms in full-thickness healing, so an open bed
# with visible fat OR granulation (base still visible, no exposed deep structure)
# is graded Stage 3. A depth cue strengthens but is not required.
_GRANULATION_STAGE3 = 0.15


def _stage3_pred(f: FeatureVector, t: Thresholds) -> bool:
    if f.skin_intact or f.has_deep_structure:
        return False
    if f.obscured_fraction >= t.obscured_unstageable:
        return False
    return f.adipose > 0.0 or f.granulation >= _GRANULATION_STAGE3


def _stage3_ev(f: FeatureVector, t: Thresholds) -> list[Evidence]:
    if f.adipose > 0.0:
        return [
            _ev(
                "adipose_visible",
                "Subcutaneous fat appears visible, which places the wound at least at Stage 3.",
                "metrics.tissue.fractions.adipose",
                round(f.adipose, 3),
                0.0,
                ">",
                "R_STAGE_3",
            )
        ]
    return [
        _ev(
            "granulation_full_thickness",
            f"An open bed with {f.granulation * 100:.0f}% granulation (which forms in "
            "full-thickness healing) and a visible base is consistent with Stage 3.",
            "metrics.tissue.fractions.granulation",
            round(f.granulation, 3),
            _GRANULATION_STAGE3,
            ">=",
            "R_STAGE_3",
        ),
        _ev(
            "depth_cue",
            f"Relative-depth cue {f.depth_index:.2f} (a weak 2D shading proxy, not a "
            "measurement).",
            "metrics.depth_proxy.relative_depth_index",
            round(f.depth_index, 3),
            t.depth_stage3,
            ">=" if f.depth_index >= t.depth_stage3 else "<",
            "R_STAGE_3",
            direction=EvidenceDirection.NEUTRAL,
            weight=0.3,
        ),
    ]


# --- Stage 2 ----------------------------------------------------------------
# Partial-thickness: a shallow, viable pink/red bed with NO granulation, slough,
# eschar, or visible fat (granulation would indicate full-thickness, i.e. >= 3).
def _stage2_pred(f: FeatureVector, t: Thresholds) -> bool:
    return (
        (not f.skin_intact)
        and f.epithelial > 0.05
        and f.granulation < _GRANULATION_STAGE3
        and f.obscured_fraction < t.devitalized_max_stage2
        and f.adipose == 0.0
        and not f.has_deep_structure
    )


def _stage2_ev(f: FeatureVector, t: Thresholds) -> list[Evidence]:
    return [
        _ev(
            "shallow_viable_bed",
            "A shallow, open, viable (pink/epithelial) bed with no granulation, slough, "
            "eschar, or visible fat is consistent with partial-thickness (Stage 2) loss.",
            "metrics.tissue.obscured_fraction",
            round(f.obscured_fraction, 3),
            t.devitalized_max_stage2,
            "<",
            "R_STAGE_2",
        )
    ]


# --- Stage 1 ----------------------------------------------------------------
def _stage1_pred(f: FeatureVector, t: Thresholds) -> bool:
    return f.skin_intact and f.maroon_purple < t.maroon_min and f.erythema >= t.erythema_min


def _stage1_ev(f: FeatureVector, t: Thresholds) -> list[Evidence]:
    return [
        _ev(
            "erythema_intact_skin",
            f"Localized erythema over intact skin ({f.erythema * 100:.0f}% of the region) "
            "with no open bed is consistent with Stage 1.",
            "metrics.color_cues.erythema_fraction",
            round(f.erythema, 3),
            t.erythema_min,
            ">=",
            "R_STAGE_1",
        )
    ]


RULES: list[Rule] = [
    Rule("R_INDETERMINATE", 200, PressureInjuryStage.INDETERMINATE,
         "Insufficient evidence to grade.", _indeterminate_pred, _indeterminate_ev),
    Rule("R_UNSTAGEABLE", 150, PressureInjuryStage.UNSTAGEABLE,
         "Base obscured by slough/eschar.", _unstageable_pred, _unstageable_ev),
    Rule("R_STAGE_4", 140, PressureInjuryStage.STAGE_4,
         "Exposed deep structure.", _stage4_pred, _stage4_ev),
    Rule("R_DTI", 130, PressureInjuryStage.DEEP_TISSUE,
         "Maroon/purple over intact skin.", _dti_pred, _dti_ev),
    Rule("R_STAGE_3", 120, PressureInjuryStage.STAGE_3,
         "Full-thickness, fat visible.", _stage3_pred, _stage3_ev),
    Rule("R_STAGE_2", 100, PressureInjuryStage.STAGE_2,
         "Partial-thickness open bed.", _stage2_pred, _stage2_ev),
    Rule("R_STAGE_1", 90, PressureInjuryStage.STAGE_1,
         "Non-blanchable erythema, intact skin.", _stage1_pred, _stage1_ev),
]

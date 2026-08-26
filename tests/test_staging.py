"""Staging engine behavior across clinical scenarios."""

from __future__ import annotations

from sorbed.domain.enums import PressureInjuryStage as S
from sorbed.domain.enums import SkinToneBand
from sorbed.staging.engine import StagingEngine
from sorbed.staging.features import FeatureVector


def _fv(**over) -> FeatureVector:
    base = {
        "skin_intact": False, "open_bed_fraction": 1.0, "obscured_fraction": 0.0,
        "granulation": 0.0, "slough": 0.0, "eschar": 0.0, "epithelial": 0.0,
        "adipose": 0.0, "muscle": 0.0, "tendon_bone": 0.0, "has_deep_structure": False,
        "maroon_purple": 0.0, "erythema": 0.0, "depth_index": 0.2, "area_cm2": 5.0,
        "is_calibrated": True, "seg_confidence": 0.8, "wound_fraction_of_image": 0.2,
    }
    base.update(over)
    return FeatureVector(**base)


def _decide(fv: FeatureVector, skin=SkinToneBand.I_III):
    return StagingEngine().decide(fv, skin_tone=skin, is_calibrated=True)


def test_granulation_is_stage_3():
    d = _decide(_fv(granulation=0.9, open_bed_fraction=0.9))
    assert d.stage is S.STAGE_3
    assert not d.abstained


def test_obscured_bed_is_unstageable():
    d = _decide(_fv(eschar=0.6, slough=0.3, obscured_fraction=0.9, open_bed_fraction=0.9))
    assert d.stage is S.UNSTAGEABLE


def test_exposed_structure_is_stage_4():
    d = _decide(
        _fv(tendon_bone=0.2, granulation=0.5, has_deep_structure=True, open_bed_fraction=0.7)
    )
    assert d.stage is S.STAGE_4


def test_maroon_intact_skin_is_dti():
    d = _decide(_fv(skin_intact=True, open_bed_fraction=0.0, maroon_purple=0.4))
    assert d.stage is S.DEEP_TISSUE


def test_erythema_intact_skin_is_stage_1():
    d = _decide(_fv(skin_intact=True, open_bed_fraction=0.0, erythema=0.4))
    assert d.stage is S.STAGE_1


def test_low_segmentation_abstains():
    d = _decide(_fv(granulation=0.9, seg_confidence=0.05))
    assert d.stage is S.INDETERMINATE
    assert d.abstained


def test_depth_dependent_grade_is_flagged_for_review():
    d = _decide(_fv(granulation=0.9, open_bed_fraction=0.9))
    assert d.requires_clinician_review
    assert any(c.severity.value == "warning" for c in d.caveats)


def test_dark_skin_adds_critical_caveat_for_early_stage():
    d = _decide(_fv(skin_intact=True, open_bed_fraction=0.0, erythema=0.4), skin=SkinToneBand.IV_VI)
    assert any(c.severity.value == "critical" for c in d.caveats)


def test_uncalibrated_adds_info_caveat():
    d = StagingEngine().decide(
        _fv(granulation=0.9, is_calibrated=False, area_cm2=None),
        skin_tone=SkinToneBand.I_III,
        is_calibrated=False,
    )
    assert any(c.severity.value == "info" for c in d.caveats)


def test_every_fired_grade_has_evidence():
    for fv in [
        _fv(granulation=0.9, open_bed_fraction=0.9),
        _fv(eschar=0.7, obscured_fraction=0.7, open_bed_fraction=0.9),
        _fv(skin_intact=True, open_bed_fraction=0.0, maroon_purple=0.4),
    ]:
        d = _decide(fv)
        assert d.evidence, f"no evidence for {d.stage}"

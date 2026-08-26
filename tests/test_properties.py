"""Property-based invariants that must hold for any valid input."""

from __future__ import annotations

import numpy as np
from hypothesis import given, settings
from hypothesis import strategies as st

from sorbed.domain.enums import PressureInjuryStage, SkinToneBand
from sorbed.staging.engine import StagingEngine
from sorbed.staging.features import FeatureVector
from sorbed.tissue.color_model import ColorTissueClassifier

_frac = st.floats(min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False)


@given(
    granulation=_frac, slough=_frac, eschar=_frac, epithelial=_frac,
    maroon=_frac, erythema=_frac, depth=_frac, seg=_frac, obscured=_frac,
    intact=st.booleans(), structure=st.booleans(),
)
@settings(max_examples=200, deadline=None)
def test_engine_always_returns_valid_decision(
    granulation, slough, eschar, epithelial, maroon, erythema, depth, seg, obscured,
    intact, structure,
):
    fv = FeatureVector(
        skin_intact=intact, open_bed_fraction=min(1.0, granulation + slough + eschar),
        obscured_fraction=obscured, granulation=granulation, slough=slough, eschar=eschar,
        epithelial=epithelial, adipose=0.0, muscle=0.0, tendon_bone=0.0,
        has_deep_structure=structure, maroon_purple=maroon, erythema=erythema,
        depth_index=depth, area_cm2=5.0, is_calibrated=True, seg_confidence=seg,
        wound_fraction_of_image=0.2,
    )
    decision = StagingEngine().decide(fv, skin_tone=SkinToneBand.UNKNOWN, is_calibrated=True)
    assert isinstance(decision.stage, PressureInjuryStage)
    assert 0.0 <= decision.confidence <= 1.0
    # A confident, non-abstaining grade must always carry supporting evidence.
    if not decision.abstained:
        assert decision.evidence


@given(
    r=st.floats(0.1, 0.9), g=st.floats(0.1, 0.9), b=st.floats(0.1, 0.9),
)
@settings(max_examples=100, deadline=None)
def test_tissue_fractions_sum_to_one_for_any_solid_color(r, g, b):
    rgb = np.zeros((40, 40, 3), dtype=np.float32)
    rgb[:] = (r, g, b)
    mask = np.zeros((40, 40), dtype=bool)
    mask[10:30, 10:30] = True
    comp = ColorTissueClassifier().classify(rgb, mask).composition
    assert abs(sum(comp.fractions.values()) - 1.0) < 1e-3
    for frac in comp.fractions.values():
        assert 0.0 <= frac <= 1.0


def test_tissue_fractions_rotation_invariant():
    from tests.synth import GRANULATION, make_wound

    wound = make_wound(fill=GRANULATION, seed=21)
    classifier = ColorTissueClassifier()
    up = classifier.classify(wound.rgb, wound.wound_mask).composition
    rot = classifier.classify(
        np.rot90(wound.rgb).copy(), np.rot90(wound.wound_mask).copy()
    ).composition
    # Dominant tissue and its fraction survive a 90-degree rotation.
    assert up.dominant == rot.dominant
    assert abs(up.fraction_of(up.dominant) - rot.fraction_of(rot.dominant)) < 0.02

"""Shared fixtures — all built on real, computed data."""

from __future__ import annotations

import pytest

from sorbed.pipeline import AnalyzeOptions, analyze_image
from tests.synth import GRANULATION, SLOUGH, SynthWound, make_wound


@pytest.fixture
def granulating_wound() -> SynthWound:
    """A granulating wound with a small slough island (clinically Stage 3)."""
    return make_wound(
        axes=(80, 55),
        fill=GRANULATION,
        inner=((28, 20), SLOUGH),
        seed=1,
    )


@pytest.fixture
def granulating_analysis(granulating_wound: SynthWound):
    """The full analysis of the granulating wound, calibrated at 0.2 mm/px."""
    bundle = analyze_image(
        granulating_wound.to_png_bytes(), options=AnalyzeOptions(mm_per_px=0.2)
    )
    return bundle.analysis

"""Tests for the HuggingFace SAM/MedSAM segmentation backend.

These verify the integration is real and degrades cleanly: with a wound present
it attempts to load the model and, when transformers/torch or the Hub are
unavailable, raises a clear, actionable error; with no wound it returns the
honest empty proposal without loading anything.
"""

from __future__ import annotations

import importlib.util

import numpy as np
import pytest

from sorbed.domain.enums import CalibrationStatus
from sorbed.domain.image import Calibration
from sorbed.imaging import RasterImage, build_metadata
from sorbed.segmentation.backends.hf_backend import HuggingFaceSAMSegmenter
from tests.synth import GRANULATION, blank_image, make_wound

_HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None


def _raster(rgb: np.ndarray) -> RasterImage:
    meta = build_metadata(pixels=rgb, source_format="png", bit_depth=8, channels=3)
    return RasterImage(pixels=rgb, metadata=meta, calibration=Calibration(
        status=CalibrationStatus.UNCALIBRATED))


def test_from_settings_uses_default_model():
    seg = HuggingFaceSAMSegmenter.from_settings(object())
    assert seg.name == "hf_sam"
    assert "sam" in seg._model_id.lower()


def test_blank_image_needs_no_model():
    # No wound -> the classical proposal is empty, so no model load is attempted.
    seg = HuggingFaceSAMSegmenter()
    result = seg.segment(_raster(blank_image()))
    assert result.area_px == 0
    assert result.backend == "hf_sam"


@pytest.mark.skipif(
    _HAS_TRANSFORMERS, reason="transformers installed; Hub load path exercised elsewhere"
)
def test_missing_transformers_raises_clear_error():
    seg = HuggingFaceSAMSegmenter()
    wound = make_wound(fill=GRANULATION, seed=42)
    with pytest.raises(RuntimeError, match=r"transformers|huggingface|Hub"):
        seg.segment(_raster(wound.rgb))

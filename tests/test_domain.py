"""Contract validation tests for the domain models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from sorbed.domain.enums import CalibrationStatus, PressureInjuryStage, TissueClass
from sorbed.domain.image import Calibration
from sorbed.domain.metrics import TissueComposition


def test_calibration_uncalibrated_has_no_scale():
    cal = Calibration(status=CalibrationStatus.UNCALIBRATED)
    assert cal.mm_per_px is None
    assert not cal.is_calibrated
    assert cal.to_mm(100) is None
    assert cal.to_mm2(100) is None


def test_calibration_manual_requires_scale():
    with pytest.raises(ValidationError):
        Calibration(status=CalibrationStatus.MANUAL, mm_per_px=None)


def test_calibration_uncalibrated_rejects_scale():
    with pytest.raises(ValidationError):
        Calibration(status=CalibrationStatus.UNCALIBRATED, mm_per_px=0.1)


def test_calibration_conversions():
    cal = Calibration(status=CalibrationStatus.MANUAL, mm_per_px=0.5)
    assert cal.to_mm(10) == pytest.approx(5.0)
    assert cal.to_mm2(4) == pytest.approx(1.0)


def test_tissue_fractions_must_sum_to_one():
    with pytest.raises(ValidationError):
        TissueComposition(
            fractions={TissueClass.GRANULATION: 0.5, TissueClass.SLOUGH: 0.2},
            areas_px={TissueClass.GRANULATION: 5, TissueClass.SLOUGH: 2},
            dominant=TissueClass.GRANULATION,
        )


def test_tissue_obscured_fraction():
    comp = TissueComposition(
        fractions={TissueClass.GRANULATION: 0.4, TissueClass.SLOUGH: 0.3, TissueClass.ESCHAR: 0.3},
        areas_px={TissueClass.GRANULATION: 4, TissueClass.SLOUGH: 3, TissueClass.ESCHAR: 3},
        dominant=TissueClass.GRANULATION,
    )
    assert comp.obscured_fraction == pytest.approx(0.6)


def test_stage_depth_dependence():
    assert PressureInjuryStage.STAGE_4.is_depth_dependent
    assert PressureInjuryStage.UNSTAGEABLE.is_depth_dependent
    assert not PressureInjuryStage.STAGE_1.is_depth_dependent


def test_tissue_class_helpers():
    assert TissueClass.GRANULATION.is_viable
    assert TissueClass.ESCHAR.is_devitalized
    assert TissueClass.TENDON_BONE.is_deep_structure

"""Controlled vocabularies for wound analysis.

These enums encode clinical categories from the NPIAP 2016 / EPUAP-NPIAP-PPPIA
2019 International Guideline. String values are stable identifiers safe to
serialize into reports and to compare across versions.
"""

from __future__ import annotations

from enum import StrEnum


class PressureInjuryStage(StrEnum):
    """Pressure-injury stage per the international staging system.

    A wound is classified by the *maximum visible anatomical tissue loss* and is
    never back-staged. ``INDETERMINATE`` is a first-class, safe outcome used when
    the evidence does not support any confident grade.
    """

    STAGE_1 = "stage_1"  # non-blanchable erythema, intact skin
    STAGE_2 = "stage_2"  # partial-thickness loss, exposed dermis
    STAGE_3 = "stage_3"  # full-thickness, adipose visible
    STAGE_4 = "stage_4"  # full-thickness, muscle/tendon/bone visible
    UNSTAGEABLE = "unstageable"  # base obscured by slough/eschar
    DEEP_TISSUE = "deep_tissue_injury"  # maroon/purple discoloration or blood blister
    MUCOSAL = "mucosal_not_stageable"  # mucosal membrane; cannot be staged
    NOT_PRESSURE_INJURY = "not_pressure_injury"  # classifier abstains / other cause
    INDETERMINATE = "indeterminate"  # insufficient evidence to grade

    @property
    def is_depth_dependent(self) -> bool:
        """Stages whose determination hinges on depth or obscured tissue.

        These cannot be fully resolved from a single 2D image and always warrant
        clinician confirmation.
        """
        return self in {
            PressureInjuryStage.STAGE_3,
            PressureInjuryStage.STAGE_4,
            PressureInjuryStage.UNSTAGEABLE,
            PressureInjuryStage.DEEP_TISSUE,
        }


class TissueClass(StrEnum):
    """Wound-bed tissue categories with their canonical color signatures."""

    EPITHELIAL = "epithelial"  # pale pink / pearly, migrating edge
    GRANULATION = "granulation"  # beefy red, moist, cobblestone
    SLOUGH = "slough"  # yellow / tan / gray / green, devitalized
    ESCHAR = "eschar"  # black / brown, dry, leathery necrosis
    ADIPOSE = "adipose"  # exposed fat, pale yellow (implies >= Stage 3)
    MUSCLE = "muscle"  # dark red, striated (implies Stage 4)
    TENDON_BONE = "tendon_bone"  # yellow-white shiny / white hard (implies Stage 4)
    INTACT_SKIN = "intact_skin"  # unbroken skin within the analysis region
    BACKGROUND = "background"  # outside the wound and periwound
    UNKNOWN = "unknown"

    @property
    def is_viable(self) -> bool:
        return self in {TissueClass.EPITHELIAL, TissueClass.GRANULATION}

    @property
    def is_devitalized(self) -> bool:
        return self in {TissueClass.SLOUGH, TissueClass.ESCHAR}

    @property
    def is_deep_structure(self) -> bool:
        """Tissue whose exposure gates Stage 3 (adipose) or Stage 4."""
        return self in {
            TissueClass.ADIPOSE,
            TissueClass.MUSCLE,
            TissueClass.TENDON_BONE,
        }


# The Red-Yellow-Black legacy proxy, retained because it maps directly to the
# PUSH "worst tissue" subscore and is a useful, interpretable cross-check.
RYB_TO_TISSUE: dict[str, TissueClass] = {
    "red": TissueClass.GRANULATION,
    "yellow": TissueClass.SLOUGH,
    "black": TissueClass.ESCHAR,
}


class CalibrationStatus(StrEnum):
    """How real-world scale (mm per pixel) was established, if at all.

    ``UNCALIBRATED`` is not an error — it means the image carried no scale
    reference, so physical measurements are reported as ``null`` while
    pixel-space metrics still compute. Scale is never fabricated.
    """

    DICOM_SPACING = "dicom_pixel_spacing"
    FIDUCIAL_MARKER = "fiducial_marker"
    RULER_DETECTED = "ruler_detected"
    MANUAL = "manual_mm_per_px"
    UNCALIBRATED = "uncalibrated"


class SkinToneBand(StrEnum):
    """Coarse Fitzpatrick grouping used to modulate confidence and warnings.

    Early-stage and deep-tissue injuries are substantially harder to detect on
    darker skin; the pipeline lowers confidence and raises an explicit warning
    in the ``IV_VI`` band rather than risking a false-negative.
    """

    I_III = "fitzpatrick_i_iii"
    IV_VI = "fitzpatrick_iv_vi"
    UNKNOWN = "unknown"


class EvidenceDirection(StrEnum):
    SUPPORTS = "supports"
    CONTRADICTS = "contradicts"
    NEUTRAL = "neutral"


class Severity(StrEnum):
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"

"""Assemble the staging feature vector from computed metrics.

Every field is derived from a real metric; nothing is synthesized. The vector is
the sole input to the rule layer, keeping staging auditable.
"""

from __future__ import annotations

from dataclasses import dataclass

from sorbed.domain.enums import TissueClass
from sorbed.domain.metrics import Metrics


@dataclass(frozen=True)
class FeatureVector:
    skin_intact: bool
    open_bed_fraction: float
    obscured_fraction: float
    granulation: float
    slough: float
    eschar: float
    epithelial: float
    adipose: float
    muscle: float
    tendon_bone: float
    has_deep_structure: bool
    maroon_purple: float
    erythema: float
    depth_index: float
    area_cm2: float | None
    is_calibrated: bool
    seg_confidence: float
    wound_fraction_of_image: float


def build_features(metrics: Metrics, seg_confidence: float) -> FeatureVector:
    """Build the :class:`FeatureVector` the rule layer consumes."""
    tissue = metrics.tissue
    cues = metrics.color_cues
    depth = metrics.depth_proxy.relative_depth_index if metrics.depth_proxy else 0.0
    return FeatureVector(
        skin_intact=metrics.skin_intact,
        open_bed_fraction=cues.open_bed_fraction,
        obscured_fraction=tissue.obscured_fraction,
        granulation=tissue.fraction_of(TissueClass.GRANULATION),
        slough=tissue.fraction_of(TissueClass.SLOUGH),
        eschar=tissue.fraction_of(TissueClass.ESCHAR),
        epithelial=tissue.fraction_of(TissueClass.EPITHELIAL),
        adipose=tissue.fraction_of(TissueClass.ADIPOSE),
        muscle=tissue.fraction_of(TissueClass.MUSCLE),
        tendon_bone=tissue.fraction_of(TissueClass.TENDON_BONE),
        has_deep_structure=tissue.has_deep_structure,
        maroon_purple=cues.maroon_purple_fraction,
        erythema=cues.erythema_fraction,
        depth_index=depth,
        area_cm2=metrics.geometry.area_cm2,
        is_calibrated=metrics.geometry.area_cm2 is not None,
        seg_confidence=seg_confidence,
        wound_fraction_of_image=metrics.geometry.wound_fraction_of_image,
    )

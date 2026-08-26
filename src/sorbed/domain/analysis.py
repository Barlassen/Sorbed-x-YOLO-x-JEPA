"""The root aggregate: a complete, serializable wound analysis.

The JSON serialization of :class:`WoundAnalysis` is the single source of truth
for the whole system. Overlays, guides, and HTML/PDF reports are all renderings
of this object, and golden tests pin its output.
"""

from __future__ import annotations

from datetime import datetime
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field

from sorbed.domain.decision import StageDecision
from sorbed.domain.enums import SkinToneBand
from sorbed.domain.image import Calibration, ImageMetadata
from sorbed.domain.metrics import Metrics

SCHEMA_VERSION = "1.0.0"

DISCLAIMER = (
    "Sorbed is clinical decision-support software, not a medical device and not "
    "a diagnostic tool. This result is a provisional estimate computed from a 2D "
    "image and must be reviewed by a qualified clinician. Depth, undermining, and "
    "tunneling cannot be measured from a photograph. Detection of early-stage and "
    "deep-tissue injury is less reliable on darker skin. See DISCLAIMER.md."
)


class ModelProvenance(BaseModel):
    """What produced each stage's output, for reproducibility and audit."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    segmentation_backend: str
    tissue_backend: str
    staging_backend: str
    weights_sha256: dict[str, str] = Field(default_factory=dict)


class WoundAnalysis(BaseModel):
    """Everything Sorbed determined about one wound in one image."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    schema_version: str = SCHEMA_VERSION
    analysis_id: UUID
    created_at: datetime
    image: ImageMetadata
    calibration: Calibration
    skin_tone_band: SkinToneBand = SkinToneBand.UNKNOWN
    metrics: Metrics
    decision: StageDecision
    provenance: ModelProvenance
    stage_timings_ms: dict[str, float] = Field(default_factory=dict)
    config_digest: str = Field(description="SHA-256 of the effective configuration.")
    disclaimer: str = DISCLAIMER

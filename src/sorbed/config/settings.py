"""Effective configuration, assembled from defaults and environment variables.

Every analysis records a ``config_digest`` (a SHA-256 of the effective settings)
so a result can be tied back to exactly the configuration that produced it.
"""

from __future__ import annotations

import hashlib
import json

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings. Override any field with a ``SORBED_`` env var."""

    model_config = SettingsConfigDict(env_prefix="SORBED_", frozen=True, extra="forbid")

    # Backends. "classical" needs no weights and always works offline;
    # "hf_sam"/"onnx" select learned backends (see docs/MODELS.md).
    segmentation_backend: str = "classical"
    tissue_backend: str = "color_model"
    staging_backend: str = "rule_engine"
    # Depth cue for the Stage-3 rule. "classical" is the weight-free shading proxy;
    # "yolo" uses a YOLO26 monocular-depth model (needs the [ml]/ultralytics extra).
    depth_backend: str = "classical"

    # HuggingFace segmentation model id (used when segmentation_backend="hf_sam").
    hf_model_id: str = "facebook/sam-vit-base"

    # Segmentation controls.
    min_wound_area_fraction: float = Field(default=0.0015, ge=0, le=1)
    grabcut_iterations: int = Field(default=5, ge=1, le=20)

    # Staging safety thresholds (see docs/CLINICAL.md).
    obscured_fraction_unstageable: float = Field(default=0.5, ge=0, le=1)
    abstain_confidence_margin: float = Field(default=0.12, ge=0, le=1)
    low_confidence_threshold: float = Field(default=0.5, ge=0, le=1)

    # Preprocessing.
    apply_color_normalization: bool = True
    detect_fiducials: bool = True

    # Output.
    output_dir: str = "reports"
    overlay_alpha: float = Field(default=0.45, ge=0, le=1)

    def digest(self) -> str:
        payload = json.dumps(self.model_dump(mode="json"), sort_keys=True)
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def get_settings(**overrides: object) -> Settings:
    """Build a settings object, applying explicit overrides last."""
    return Settings(**overrides)  # type: ignore[arg-type]

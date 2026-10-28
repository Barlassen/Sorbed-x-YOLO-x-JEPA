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
    # "hf_sam"/"onnx"/"yolo" and tissue "cnn" select learned backends (see docs/MODELS.md).
    segmentation_backend: str = "classical"
    tissue_backend: str = "color_model"
    staging_backend: str = "rule_engine"
    # Depth cue for the Stage-3 rule. "yolo" is the default: a YOLO26 monocular-depth
    # model (needs the [ml]/ultralytics extra). It falls back to the weight-free
    # "classical" shading proxy automatically if ultralytics/torch is unavailable, so
    # the core still runs offline. Set SORBED_DEPTH_BACKEND=classical to force the proxy.
    depth_backend: str = "yolo"

    # HuggingFace segmentation model id (used when segmentation_backend="hf_sam").
    hf_model_id: str = "facebook/sam-vit-base"

    # YOLO wound segmentation (segmentation_backend="yolo"). The fallback model runs
    # only when the main one finds no wound ("old-first"). The pressure-injury
    # configuration validated in HANDOFF (staging step 2) is listed in the README.
    yolo_seg_model: str = "yolo26n-seg.pt"
    yolo_seg_fallback_model: str = ""
    yolo_seg_confidence: float = Field(default=0.25, gt=0, le=1)
    yolo_seg_keep_largest: bool = True

    # Tissue CNN weights (tissue_backend="cnn"; trained by training/train_tissue_cnn.py).
    tissue_cnn_weights: str = ""

    # Segmentation controls.
    min_wound_area_fraction: float = Field(default=0.0015, ge=0, le=1)
    grabcut_iterations: int = Field(default=5, ge=1, le=20)

    # Staging safety thresholds (see docs/CLINICAL.md).
    obscured_fraction_unstageable: float = Field(default=0.3, ge=0, le=1)
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

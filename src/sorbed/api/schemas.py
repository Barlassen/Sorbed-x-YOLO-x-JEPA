"""Request/response envelopes for the HTTP API.

These are thin wrappers only. The canonical
:class:`~sorbed.domain.analysis.WoundAnalysis` remains the single source of
truth for the analysis payload itself; ``AnalyzeResponse`` merely carries its
serialized form (and an optional rendered guide) across the wire.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Liveness answer for ``GET /v1/health``."""

    status: str = Field(description="Always 'ok' when the process is serving.")
    version: str = Field(description="Installed Sorbed package version.")


class ReadyResponse(BaseModel):
    """Readiness answer for ``GET /v1/ready``."""

    ready: bool = Field(description="True when the analyzer and its backends built.")
    backends: dict[str, str] = Field(
        default_factory=dict,
        description="Configured backend names keyed by pipeline stage.",
    )
    detail: str | None = Field(
        default=None, description="Reason the service is not ready, when applicable."
    )


class ModelsResponse(BaseModel):
    """Configured backends and any discoverable learned models."""

    backends: dict[str, str] = Field(
        description="Configured backend names keyed by pipeline stage."
    )
    ml_models: list[Any] = Field(
        default_factory=list,
        description="Learned models discovered via the optional model registry.",
    )


class AnalyzeResponse(BaseModel):
    """Wrapper around a serialized analysis plus rendered panels."""

    analysis: dict[str, Any] = Field(
        description="The WoundAnalysis serialized with model_dump(mode='json')."
    )
    images: dict[str, str] = Field(
        default_factory=dict,
        description="Rendered panels (input/mask/overlay/detection/depth/schematic) "
        "as PNG data URIs; empty when images were not requested.",
    )
    guide_png_base64: str | None = Field(
        default=None,
        description="Base64-encoded PNG of the annotated guide when requested.",
    )


class TrendResponse(BaseModel):
    """A healing trend across visits plus per-visit thumbnails."""

    trend: dict[str, Any] = Field(description="The serialized HealingTrend.")
    thumbnails: list[str] = Field(
        default_factory=list, description="Per-visit detection thumbnails as data URIs."
    )


class ErrorResponse(BaseModel):
    """Uniform error envelope returned by the global exception handlers."""

    error: str

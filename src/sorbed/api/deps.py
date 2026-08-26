"""Shared dependencies and API-specific configuration.

The core :class:`~sorbed.config.settings.Settings` governs the pipeline; the
transport-only knobs (upload ceiling, CORS) live in :class:`ApiSettings` so the
frozen core config stays focused on analysis behavior.
"""

from __future__ import annotations

from fastapi import Request
from pydantic_settings import BaseSettings, SettingsConfigDict

from sorbed.config.settings import Settings
from sorbed.pipeline import WoundAnalyzer


class ApiSettings(BaseSettings):
    """Transport-layer settings. Override with ``SORBED_API_`` env vars."""

    model_config = SettingsConfigDict(env_prefix="SORBED_API_", extra="ignore")

    max_upload_bytes: int = 25 * 1024 * 1024
    cors_origins: list[str] = ["*"]
    cors_allow_credentials: bool = False


def get_core_settings(request: Request) -> Settings:
    """Return the pipeline settings bound to this application."""
    return request.app.state.settings


def get_api_settings(request: Request) -> ApiSettings:
    """Return the transport-layer settings bound to this application."""
    return request.app.state.api_settings


def get_analyzer(request: Request) -> WoundAnalyzer:
    """Return a process-wide analyzer, building its backends on first use.

    The analyzer owns the configured segmentation, tissue, and staging backends;
    reusing one instance across requests avoids rebuilding them per call.
    """
    analyzer = getattr(request.app.state, "analyzer", None)
    if analyzer is None:
        analyzer = WoundAnalyzer(request.app.state.settings)
        request.app.state.analyzer = analyzer
    return analyzer


def backend_names(settings: Settings) -> dict[str, str]:
    """Map each pipeline stage to its configured backend name."""
    return {
        "segmentation": settings.segmentation_backend,
        "tissue": settings.tissue_backend,
        "staging": settings.staging_backend,
    }

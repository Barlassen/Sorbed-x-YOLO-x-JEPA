"""Introspection endpoints: supported formats, backends/models, and schema."""

from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends

from sorbed.api.deps import backend_names, get_core_settings
from sorbed.api.schemas import ModelsResponse
from sorbed.config.settings import Settings
from sorbed.domain.analysis import WoundAnalysis
from sorbed.io import DecoderRegistry

router = APIRouter(tags=["meta"])


@router.get("/formats")
async def formats() -> dict[str, bool]:
    """Map each known container format to whether a decoder is currently usable."""
    return DecoderRegistry().supported_formats()


@router.get("/models", response_model=ModelsResponse)
async def models(
    settings: Settings = Depends(get_core_settings),  # noqa: B008
) -> ModelsResponse:
    """List configured backends and any learned models the registry exposes."""
    ml_models: list[Any] = []
    try:
        from sorbed.models.registry import list_models

        ml_models = list(list_models())
    except Exception:  # the model registry is optional
        ml_models = []
    return ModelsResponse(backends=backend_names(settings), ml_models=ml_models)


@router.get("/schema")
async def schema() -> dict[str, Any]:
    """Return the JSON schema of the canonical WoundAnalysis result."""
    return WoundAnalysis.model_json_schema()

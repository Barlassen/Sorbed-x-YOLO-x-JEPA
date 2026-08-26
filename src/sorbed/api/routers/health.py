"""Liveness and readiness probes."""

from __future__ import annotations

from fastapi import APIRouter, Depends, Request
from fastapi.responses import JSONResponse

from sorbed.api.deps import backend_names, get_analyzer, get_core_settings
from sorbed.api.schemas import HealthResponse, ReadyResponse
from sorbed.config.settings import Settings
from sorbed.version import __version__

router = APIRouter(tags=["health"])


@router.get("/health", response_model=HealthResponse)
async def health() -> HealthResponse:
    """Report that the process is up and serving."""
    return HealthResponse(status="ok", version=__version__)


@router.get("/ready", response_model=ReadyResponse)
async def ready(
    request: Request,
    settings: Settings = Depends(get_core_settings),  # noqa: B008
) -> ReadyResponse | JSONResponse:
    """Report readiness by constructing the analyzer and its backends.

    A 503 is returned if any backend fails to build, so orchestrators can gate
    traffic until the pipeline is genuinely usable.
    """
    try:
        get_analyzer(request)
    except Exception as exc:  # readiness surfaces any backend build failure
        body = ReadyResponse(ready=False, backends={}, detail=str(exc))
        return JSONResponse(status_code=503, content=body.model_dump())
    return ReadyResponse(ready=True, backends=backend_names(settings))

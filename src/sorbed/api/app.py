"""Application factory and the module-level ASGI app.

``create_app`` wires the routers, CORS, and uniform JSON error handling. A
module-level ``app`` is provided so ``uvicorn sorbed.api.app:app`` works out of
the box.
"""

from __future__ import annotations

from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.encoders import jsonable_encoder
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.requests import Request
from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from starlette.exceptions import HTTPException as StarletteHTTPException

from sorbed.api.deps import ApiSettings
from sorbed.api.routers import analyze, health, meta
from sorbed.config.settings import Settings, get_settings
from sorbed.version import __version__

_DESCRIPTION = (
    "Explainable pressure-injury (bedsore) image analysis. Sorbed is clinical "
    "decision-support software: it returns a provisional, evidence-backed grade "
    "for clinician review and is NOT a medical device or a diagnostic tool. "
    "Upload a wound photograph to receive a structured WoundAnalysis. Images are "
    "processed in memory and are not retained by the service."
)


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build a configured :class:`FastAPI` application.

    ``settings`` overrides the pipeline configuration; transport concerns (upload
    ceiling, CORS) come from :class:`~sorbed.api.deps.ApiSettings` read from the
    environment.
    """
    settings = settings or get_settings()
    api_settings = ApiSettings()

    app = FastAPI(
        title="Sorbed API",
        version=__version__,
        description=_DESCRIPTION,
    )
    app.state.settings = settings
    app.state.api_settings = api_settings
    app.state.analyzer = None

    app.add_middleware(
        CORSMiddleware,
        allow_origins=api_settings.cors_origins,
        allow_credentials=api_settings.cors_allow_credentials,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    app.include_router(health.router, prefix="/v1")
    app.include_router(meta.router, prefix="/v1")
    app.include_router(analyze.router, prefix="/v1")

    _mount_web_ui(app)
    _install_error_handlers(app)
    return app


class _RevalidateStaticFiles(StaticFiles):
    """Serve static assets with ``Cache-Control: no-cache``.

    The SPA's HTML/CSS/JS evolve without content-hashed filenames, so browsers
    must revalidate on every load rather than apply heuristic freshness — a
    stale ``app.js`` otherwise masks a fixed one. ``no-cache`` still permits a
    conditional request (ETag/If-None-Match), so unchanged files return a cheap
    304 while changed files are re-fetched.
    """

    def file_response(self, *args: object, **kwargs: object) -> Response:
        response = super().file_response(*args, **kwargs)  # type: ignore[arg-type]
        response.headers["Cache-Control"] = "no-cache"
        return response


def _mount_web_ui(app: FastAPI) -> None:
    """Serve the single-page web UI and its static assets, if present."""
    static_dir = Path(__file__).resolve().parent / "static"
    index = static_dir / "index.html"
    if not index.is_file():
        return
    app.mount("/static", _RevalidateStaticFiles(directory=static_dir), name="static")

    @app.get("/", include_in_schema=False)
    async def _index() -> FileResponse:
        return FileResponse(index, headers={"Cache-Control": "no-cache"})


def _install_error_handlers(app: FastAPI) -> None:
    """Register handlers that render every error as ``{"error": ...}``."""

    @app.exception_handler(StarletteHTTPException)
    async def _http_error(_: Request, exc: StarletteHTTPException) -> JSONResponse:
        return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})

    @app.exception_handler(RequestValidationError)
    async def _validation_error(_: Request, exc: RequestValidationError) -> JSONResponse:
        return JSONResponse(
            status_code=422,
            content={
                "error": "request validation failed",
                "detail": jsonable_encoder(exc.errors()),
            },
        )

    @app.exception_handler(Exception)
    async def _unhandled_error(_: Request, exc: Exception) -> JSONResponse:
        if isinstance(exc, HTTPException):
            return JSONResponse(status_code=exc.status_code, content={"error": exc.detail})
        return JSONResponse(status_code=500, content={"error": "internal server error"})


app = create_app()

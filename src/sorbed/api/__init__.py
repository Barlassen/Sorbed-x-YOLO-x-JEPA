"""The Sorbed FastAPI service.

Exposes the wound-analysis pipeline over HTTP. Import the ready-made ``app`` for
``uvicorn sorbed.api.app:app``, or call :func:`create_app` to build a fresh
instance with custom :class:`~sorbed.config.settings.Settings`.
"""

from __future__ import annotations

from sorbed.api.app import app, create_app

__all__ = ["app", "create_app"]

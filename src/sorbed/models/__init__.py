"""Model weight registry and integrity-verified download.

Sorbed never vendors model weights. Each learned artifact is described by a
:class:`~sorbed.models.registry.ModelEntry` (URL, SHA-256, license, backend), and
is fetched only on explicit request and verified on arrival. The registry ships
empty of remote entries by default — real weights are added through a local
``models/registry.json`` — plus the weight-free ``classical`` backend, which is
always available and needs no download.
"""

from __future__ import annotations

from sorbed.models.registry import (
    ModelEntry,
    download_weights,
    get_entry,
    list_models,
    resolve_cache_dir,
    verify_sha256,
)

__all__ = [
    "ModelEntry",
    "download_weights",
    "get_entry",
    "list_models",
    "resolve_cache_dir",
    "verify_sha256",
]

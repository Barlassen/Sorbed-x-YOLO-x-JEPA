"""Load directive packs from disk.

A pack directory contains ``meta.json``, ``sections.json``, ``topics.json`` and
a ``figures/`` folder, as produced by ``scripts/build_directive_pack.py``.

Resolution order for the *default* pack:

1. ``SORBED_DIRECTIVE_PACK`` environment variable (a pack directory), if set.
2. ``var/directive_packs/<slug>`` relative to the current working directory.

The default is intentionally file-based rather than vendored: institutional
guidelines are third-party content and are not committed to this repository.
"""

from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

from sorbed.guidelines.models import (
    DirectiveMeta,
    DirectivePack,
    DirectiveSection,
    TopicGuidance,
)

DEFAULT_SLUG = "hd_t86"


class DirectivePackError(RuntimeError):
    """Raised when a requested pack cannot be found or is malformed."""


def _read_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise DirectivePackError(f"missing pack file: {path}")
    data: dict[str, Any] = json.loads(path.read_text(encoding="utf-8"))
    return data


def load_pack(pack_dir: str | Path) -> DirectivePack:
    """Load a :class:`DirectivePack` from ``pack_dir``."""
    root = Path(pack_dir).resolve()
    if not root.is_dir():
        raise DirectivePackError(f"pack directory not found: {root}")

    meta = DirectiveMeta.model_validate(_read_json(root / "meta.json"))
    raw_sections = _read_json(root / "sections.json")
    raw_topics = _read_json(root / "topics.json")

    sections = {
        sid: DirectiveSection.model_validate(payload)
        for sid, payload in raw_sections.items()
    }
    topics = {
        key: TopicGuidance.model_validate(payload)
        for key, payload in raw_topics.items()
    }
    return DirectivePack(meta=meta, sections=sections, topics=topics, root=root)


def default_pack_dir(slug: str = DEFAULT_SLUG) -> Path | None:
    """Locate the default pack directory, or ``None`` if none is present."""
    env = os.environ.get("SORBED_DIRECTIVE_PACK")
    if env:
        candidate = Path(env)
        if candidate.is_dir():
            return candidate
    candidate = Path.cwd() / "var" / "directive_packs" / slug
    return candidate if candidate.is_dir() else None


@lru_cache(maxsize=8)
def _load_cached(resolved: str) -> DirectivePack:
    return load_pack(resolved)


def load_default_pack(slug: str = DEFAULT_SLUG) -> DirectivePack | None:
    """Load the default pack if one is configured/present, else ``None``.

    Reports degrade gracefully when no pack is available: analysis still runs,
    it simply carries no directive citations.
    """
    directory = default_pack_dir(slug)
    if directory is None:
        return None
    return _load_cached(str(directory.resolve()))

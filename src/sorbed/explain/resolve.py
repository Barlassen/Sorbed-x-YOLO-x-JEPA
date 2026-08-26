"""Resolve an ``Evidence.metric_ref`` dotted path against a model.

Every piece of evidence must point at a field the system actually produces. This
resolver walks attributes, dictionaries (including enum-keyed ones), and indexable
containers, returning the referenced value. A contract test uses it to verify
that no explanation cites a metric that does not exist.
"""

from __future__ import annotations

from enum import Enum
from typing import Any

_MISSING = object()


def resolve_metric_ref(root: Any, ref: str) -> Any:
    """Return the value at ``ref`` (dotted path) under ``root``, or ``_MISSING``."""
    current: Any = root
    for part in ref.split("."):
        current = _step(current, part)
        if current is _MISSING:
            return _MISSING
    return current


def metric_ref_exists(root: Any, ref: str) -> bool:
    """Whether ``ref`` resolves to a real field/value under ``root``."""
    return resolve_metric_ref(root, ref) is not _MISSING


def _step(obj: Any, part: str) -> Any:
    # Attribute (pydantic field or property).
    if hasattr(obj, part):
        return getattr(obj, part)
    # Mapping with plain or enum keys.
    if isinstance(obj, dict):
        if part in obj:
            return obj[part]
        for key, value in obj.items():
            if isinstance(key, Enum) and key.value == part:
                return value
            if str(key) == part:
                return value
    # Sequence index.
    if isinstance(obj, (list, tuple)):
        try:
            return obj[int(part)]
        except (ValueError, IndexError):
            return _MISSING
    return _MISSING

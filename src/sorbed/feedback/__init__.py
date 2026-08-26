"""Runtime feedback store — emit analyses, ingest human corrections.

This is the shipped, CPU-only feedback subsystem: it appends the canonical
``inferences.jsonl`` / ``feedback.jsonl`` logs plus their mask PNGs and reads
them back. It deliberately depends only on stdlib, ``pydantic``, ``numpy``, and
``PIL`` — never on torch, timm, sklearn, or the offline ``training`` stack — so
the deployed inference wheel stays free of the training dependencies.
"""

from __future__ import annotations

from sorbed.feedback.emit import emit_inference
from sorbed.feedback.records import (
    SCHEMA_VERSION,
    FeedbackRecord,
    InferenceRecord,
    ModelVersions,
)
from sorbed.feedback.store import FeedbackStore

__all__ = [
    "SCHEMA_VERSION",
    "FeedbackRecord",
    "FeedbackStore",
    "InferenceRecord",
    "ModelVersions",
    "emit_inference",
]

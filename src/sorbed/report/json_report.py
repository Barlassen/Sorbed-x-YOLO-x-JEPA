"""Canonical JSON serialization of a wound analysis.

The JSON form is the single source of truth; all other artifacts render from it.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from sorbed.domain.analysis import WoundAnalysis
from sorbed.domain.report import ReportArtifact


def analysis_to_json(analysis: WoundAnalysis) -> str:
    """Return the indented canonical JSON for an analysis."""
    return analysis.model_dump_json(indent=2)


def write_json(analysis: WoundAnalysis, path: Path) -> ReportArtifact:
    """Write the canonical JSON to ``path`` and describe the artifact."""
    payload = analysis_to_json(analysis).encode("utf-8")
    path.write_bytes(payload)
    return ReportArtifact(
        kind="json",
        path=str(path),
        mime="application/json",
        sha256=hashlib.sha256(payload).hexdigest(),
        bytes=len(payload),
    )

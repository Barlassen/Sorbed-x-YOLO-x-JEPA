"""Report aggregate: a wound analysis plus the rendered artifacts it produced."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from sorbed.domain.analysis import WoundAnalysis


class ReportArtifact(BaseModel):
    """A file rendered from the analysis (mask, overlay, guide, JSON, HTML, PDF)."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    kind: str  # "mask_png" | "overlay_png" | "guide_png" | "json" | "html" | "pdf"
    path: str
    mime: str
    sha256: str
    bytes: int = Field(ge=0)


class Report(BaseModel):
    """The analysis together with every artifact written for it."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    analysis: WoundAnalysis
    artifacts: list[ReportArtifact] = Field(default_factory=list)

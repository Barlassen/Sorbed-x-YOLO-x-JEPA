"""Pure, serializable data contracts for Sorbed.

This package has no heavy runtime dependencies beyond pydantic, so the contracts
are cheap to import and validate anywhere.
"""

from __future__ import annotations

from sorbed.domain.analysis import DISCLAIMER, SCHEMA_VERSION, ModelProvenance, WoundAnalysis
from sorbed.domain.decision import Caveat, Evidence, RuleFiring, StageDecision
from sorbed.domain.enums import (
    CalibrationStatus,
    EvidenceDirection,
    PressureInjuryStage,
    Severity,
    SkinToneBand,
    TissueClass,
)
from sorbed.domain.image import Calibration, ImageMetadata
from sorbed.domain.metrics import (
    ColorCues,
    DepthProxy,
    GeometryMetrics,
    HealingScores,
    Metrics,
    PeriwoundFindings,
    TissueComposition,
)
from sorbed.domain.report import Report, ReportArtifact

__all__ = [
    "DISCLAIMER",
    "SCHEMA_VERSION",
    "Calibration",
    "CalibrationStatus",
    "Caveat",
    "ColorCues",
    "DepthProxy",
    "Evidence",
    "EvidenceDirection",
    "GeometryMetrics",
    "HealingScores",
    "ImageMetadata",
    "Metrics",
    "ModelProvenance",
    "PeriwoundFindings",
    "PressureInjuryStage",
    "Report",
    "ReportArtifact",
    "RuleFiring",
    "Severity",
    "SkinToneBand",
    "StageDecision",
    "TissueClass",
    "TissueComposition",
    "WoundAnalysis",
]

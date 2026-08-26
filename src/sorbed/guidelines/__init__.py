"""Clinical directive grounding.

Loads an institutional pressure-injury guideline (a *directive pack* built from
a PDF by ``scripts/build_directive_pack.py``) and maps a :class:`WoundAnalysis`
onto its staging definitions, tissue-colour model, sizing, PUSH, and care
cadence — with verbatim section text, the guideline's own figures, and
section/page citations.
"""

from __future__ import annotations

from sorbed.guidelines.match import (
    BRADEN_BANDS,
    Citation,
    GuidelineContext,
    TissueReading,
    TopicBlock,
    build_guideline_context,
)
from sorbed.guidelines.models import (
    DirectiveMeta,
    DirectivePack,
    DirectiveSection,
    TopicGuidance,
)
from sorbed.guidelines.pack import (
    DirectivePackError,
    default_pack_dir,
    load_default_pack,
    load_pack,
)

__all__ = [
    "BRADEN_BANDS",
    "Citation",
    "DirectiveMeta",
    "DirectivePack",
    "DirectivePackError",
    "DirectiveSection",
    "GuidelineContext",
    "TissueReading",
    "TopicBlock",
    "TopicGuidance",
    "build_guideline_context",
    "default_pack_dir",
    "load_default_pack",
    "load_pack",
]

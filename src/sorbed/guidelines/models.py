"""Data contracts for a clinical *directive pack*.

A directive pack is an institutional pressure-injury guideline distilled into
citable, numbered sections plus the guideline's own figures. Packs are built
from an operator-supplied PDF by ``scripts/build_directive_pack.py`` and loaded
at runtime — the copyrighted directive content lives in the pack on disk, never
in this source tree.
"""

from __future__ import annotations

from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field


class DirectiveMeta(BaseModel):
    """Identity of the source directive, read from its cover page."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    slug: str
    doc_name: str = "Clinical directive"
    doc_no: str | None = None
    revision_no: str | None = None
    revision_date: str | None = None
    first_issue: str | None = None
    source_pdf: str | None = None
    page_count: int | None = None


class DirectiveSection(BaseModel):
    """One numbered section, extracted verbatim from the directive."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    id: str
    text: str
    page: int


class TopicGuidance(BaseModel):
    """A named topic mapped to directive sections and an optional figure."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    sections: tuple[str, ...] = ()
    figure: str | None = None
    caption: str | None = None


class DirectivePack(BaseModel):
    """A fully-loaded directive pack: metadata, sections, topic map, root dir."""

    model_config = ConfigDict(frozen=True, extra="ignore")

    meta: DirectiveMeta
    sections: dict[str, DirectiveSection] = Field(default_factory=dict)
    topics: dict[str, TopicGuidance] = Field(default_factory=dict)
    root: Path

    def section(self, section_id: str) -> DirectiveSection | None:
        return self.sections.get(section_id)

    def topic(self, key: str) -> TopicGuidance | None:
        return self.topics.get(key)

    def topic_text(self, key: str) -> str:
        """Joined verbatim text of every section a topic references."""
        topic = self.topics.get(key)
        if not topic:
            return ""
        parts = [self.sections[s].text for s in topic.sections if s in self.sections]
        return "\n\n".join(parts)

    def figure_path(self, key: str) -> Path | None:
        """Absolute path to a topic's figure, if the pack shipped one."""
        topic = self.topics.get(key)
        if not topic or not topic.figure:
            return None
        candidate = self.root / topic.figure
        return candidate if candidate.is_file() else None

    def citation(self, key: str) -> list[tuple[str, int]]:
        """``(section_id, page)`` pairs backing a topic, for footnoting."""
        topic = self.topics.get(key)
        if not topic:
            return []
        return [
            (s, self.sections[s].page) for s in topic.sections if s in self.sections
        ]

"""Map a :class:`WoundAnalysis` onto the directive's criteria.

The result, :class:`GuidelineContext`, is what a report renders: the staging
definition and figure the directive uses for the detected stage, its tissue
(RYB) colour model, size and PUSH guidance, dressing/reassessment cadence, and
the section/page citations backing each. All directive prose is pulled verbatim
from the loaded pack at call time.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from sorbed.domain.analysis import WoundAnalysis
from sorbed.domain.enums import PressureInjuryStage, TissueClass
from sorbed.guidelines.models import DirectiveMeta, DirectivePack

# Detected stage -> directive topic key (staging definitions live in §3).
_STAGE_TOPIC: dict[PressureInjuryStage, str] = {
    PressureInjuryStage.STAGE_1: "stage.stage_1",
    PressureInjuryStage.STAGE_2: "stage.stage_2",
    PressureInjuryStage.STAGE_3: "stage.stage_3",
    PressureInjuryStage.STAGE_4: "stage.stage_4",
    PressureInjuryStage.UNSTAGEABLE: "stage.unstageable",
    PressureInjuryStage.DEEP_TISSUE: "stage.deep_tissue_injury",
}

# Detected stage -> stage-specific care topic key (§4.6.2 - §4.6.4).
_CARE_TOPIC: dict[PressureInjuryStage, str] = {
    PressureInjuryStage.STAGE_1: "care.stage_1",
    PressureInjuryStage.STAGE_2: "care.stage_2",
    PressureInjuryStage.STAGE_3: "care.stage_3",
    PressureInjuryStage.STAGE_4: "care.stage_4",
    PressureInjuryStage.UNSTAGEABLE: "care.unstageable",
    PressureInjuryStage.DEEP_TISSUE: "care.stage_1",
}

# §4.5.1 reassessment cadence when the wound is left open (no dressing). Encoded
# as structured data — the section citation carries the directive's own wording.
_REASSESS_OPEN: dict[PressureInjuryStage, str] = {
    PressureInjuryStage.STAGE_1: "her şift / every shift",
    PressureInjuryStage.STAGE_2: "24 saatte bir / every 24 h",
    PressureInjuryStage.STAGE_3: "24 saatte bir / every 24 h",
    PressureInjuryStage.STAGE_4: "24 saatte bir / every 24 h",
    PressureInjuryStage.DEEP_TISSUE: "her şift / every shift",
    PressureInjuryStage.UNSTAGEABLE: "24 saatte bir / every 24 h",
}

# §4.3.2 Braden reassessment bands (5 yaş ve üzeri), for reference context.
BRADEN_BANDS: tuple[tuple[str, str, str], ...] = (
    ("≤ 11", "Çok riskli / very high risk", "8 saat / 8 h"),
    ("12–16", "Riskli / at risk", "12 saat / 12 h"),
    ("17–23", "Az riskli / low risk", "24 saat / 24 h"),
)


class Citation(BaseModel):
    model_config = ConfigDict(frozen=True)

    section: str
    page: int


class TopicBlock(BaseModel):
    """A rendered directive topic: verbatim text, optional figure, citations."""

    model_config = ConfigDict(frozen=True)

    key: str
    text: str = ""
    figure: str | None = None
    caption: str | None = None
    citations: tuple[Citation, ...] = ()


class TissueReading(BaseModel):
    """One tissue class present in the wound, tied to the directive's RYB model."""

    model_config = ConfigDict(frozen=True)

    tissue: TissueClass
    fraction: float
    ryb: str  # "kırmızı" | "sarı" | "siyah" | "—"
    directive_note: str


class GuidelineContext(BaseModel):
    """Everything a report needs to ground an analysis in the directive."""

    model_config = ConfigDict(frozen=True)

    available: bool
    meta: DirectiveMeta | None = None
    stage: TopicBlock | None = None
    tissue_model: TopicBlock | None = None
    tissue_readings: tuple[TissueReading, ...] = ()
    size: TopicBlock | None = None
    push: TopicBlock | None = None
    exudate: TopicBlock | None = None
    care: TopicBlock | None = None
    reassessment_open: str | None = None
    reassessment_citation: Citation | None = None
    definition: TopicBlock | None = None
    reference_stages: tuple[TopicBlock, ...] = ()  # the directive's full staging ladder


# Tissue class -> (RYB colour bucket, short interpretive note echoing §4.5.9).
_TISSUE_RYB: dict[TissueClass, tuple[str, str]] = {
    TissueClass.GRANULATION: ("kırmızı", "granülasyon — iyileşme dokusu, istenilen durum (§4.5.9)"),
    TissueClass.EPITHELIAL: ("kırmızı", "epitelizasyon — iyileşme dokusu (§4.5.9)"),
    TissueClass.SLOUGH: ("sarı", "fibröz/ölü doku — enfeksiyon göstergesi, tedavi edilir (§4.5.9)"),
    TissueClass.ESCHAR: ("siyah", "nekrotik doku — debride edilir (§4.5.9)"),
    TissueClass.ADIPOSE: ("—", "cilt altı yağ dokusu görünür (≥ Evre 3)"),
    TissueClass.MUSCLE: ("—", "kas görünür (Evre 4)"),
    TissueClass.TENDON_BONE: ("—", "tendon/kemik görünür (Evre 4)"),
}


def _block(pack: DirectivePack, key: str, *, with_figure: bool = False) -> TopicBlock:
    topic = pack.topic(key)
    text = pack.topic_text(key)
    citations = tuple(
        Citation(section=s, page=p) for s, p in pack.citation(key)
    )
    figure = None
    if with_figure and topic and topic.figure:
        fig_path = pack.figure_path(key)
        figure = str(fig_path) if fig_path else None
    caption = topic.caption if topic else None
    return TopicBlock(key=key, text=text, figure=figure, caption=caption, citations=citations)


def build_guideline_context(
    analysis: WoundAnalysis, pack: DirectivePack | None
) -> GuidelineContext:
    """Assemble the directive context for one analysis.

    Returns an ``available=False`` context when no pack is loaded, so reports can
    render without directive grounding rather than fail.
    """
    if pack is None:
        return GuidelineContext(available=False)

    stage = analysis.decision.stage
    stage_key = _STAGE_TOPIC.get(stage)
    care_key = _CARE_TOPIC.get(stage)

    tissue_readings: list[TissueReading] = []
    fractions = analysis.metrics.tissue.fractions
    for tissue, frac in sorted(fractions.items(), key=lambda kv: kv[1], reverse=True):
        if frac <= 0.005:
            continue
        ryb, note = _TISSUE_RYB.get(tissue, ("—", ""))
        if not note:
            continue
        tissue_readings.append(
            TissueReading(tissue=tissue, fraction=float(frac), ryb=ryb, directive_note=note)
        )

    reassess = _REASSESS_OPEN.get(stage)
    reassess_cite = None
    sec = pack.section("4.5.1")
    if reassess and sec:
        reassess_cite = Citation(section="4.5.1", page=sec.page)

    # The directive's full staging ladder (every stage that ships a figure), so a
    # report can show the whole reference scheme, not only the detected stage.
    ladder = tuple(
        _block(pack, key, with_figure=True)
        for key in _STAGE_TOPIC.values()
        if pack.figure_path(key) is not None
    )

    return GuidelineContext(
        available=True,
        meta=pack.meta,
        definition=_block(pack, "definition"),
        stage=_block(pack, stage_key, with_figure=True) if stage_key else None,
        tissue_model=_block(pack, "tissue.ryb"),
        tissue_readings=tuple(tissue_readings),
        size=_block(pack, "size"),
        push=_block(pack, "push"),
        exudate=_block(pack, "exudate"),
        care=_block(pack, care_key) if care_key else None,
        reassessment_open=reassess,
        reassessment_citation=reassess_cite,
        reference_stages=ladder,
    )

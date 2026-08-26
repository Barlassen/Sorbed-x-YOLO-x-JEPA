"""Derived clinical statistics for the reports.

Every value here is a deterministic derivation of fields already on a
:class:`WoundAnalysis` (or a :class:`HealingTrend`) — no new measurement and
nothing fabricated. Tissue fractions are normalised to the wound *bed*
(everything except intact peri-wound skin and background) before viability
ratios are computed; the raw whole-frame fractions remain available so the
denominator switch is transparent.

Design follows a strict wound-care review: the report must never turn a tissue
under-detection into false reassurance. So when the model reports essentially no
slough/eschar while granulation dominates — the classic signature of a
classifier that cannot separate thin fibrin/slough from granulation — that is
surfaced as an explicit *under-detection* signal rather than a "perfect bed".
No composite 0-100 "quality score", no photo-derived volume, and no
"undermining" claim (invisible in 2D) are emitted here.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from sorbed.domain.analysis import WoundAnalysis
from sorbed.domain.enums import TissueClass
from sorbed.trend.models import HealingTrend

_EPS = 1e-6

# Under-detection signature: (slough+eschar) essentially absent while granulation
# dominates the bed — treat the reassuring viability numbers with suspicion.
_UNDERDETECT_NONVIABLE = 0.02
_UNDERDETECT_GRAN = 0.50
# Necrotic-burden grading over the bed (graded, not a single permissive gate).
_NONVIABLE_CAUTION = 0.20
_NONVIABLE_HIGH = 0.50
# Healing-velocity bands (% wound area change per week; negative = shrinking).
_VEL_HEALING = -10.0
_VEL_STALL = -3.0


class ViabilitySplit(BaseModel):
    model_config = ConfigDict(frozen=True)
    viable_pct: float
    non_viable_pct: float
    granulation_pct: float
    epithelial_pct: float
    slough_pct: float
    eschar_pct: float


class TissueArea(BaseModel):
    model_config = ConfigDict(frozen=True)
    tissue: TissueClass
    fraction: float
    area_cm2: float | None


class GradingStats(BaseModel):
    model_config = ConfigDict(frozen=True)
    viability: ViabilitySplit
    tissue_areas: tuple[TissueArea, ...]
    granulation_slough_ratio: float | None  # None => slough not detected, not meaningful
    bed_descriptor: str                     # qualitative, bilingual
    bed_descriptor_color: str
    under_detection: bool                   # slough/eschar likely under-detected
    standard_size: str                      # "L × W · area"
    necrotic_level: str                     # "none" | "caution" | "high"
    necrotic_detail: str
    intact_skin_pct: float                  # peri-wound / margin, whole-frame


def _bed_fractions(analysis: WoundAnalysis) -> dict[TissueClass, float]:
    fr = dict(analysis.metrics.tissue.fractions)
    bed = 1.0 - fr.get(TissueClass.INTACT_SKIN, 0.0) - fr.get(TissueClass.BACKGROUND, 0.0)
    if bed <= _EPS:
        return dict.fromkeys(fr, 0.0)
    return {k: (v / bed if k not in (TissueClass.INTACT_SKIN, TissueClass.BACKGROUND) else 0.0)
            for k, v in fr.items()}


def viability_split(analysis: WoundAnalysis) -> ViabilitySplit:
    b = _bed_fractions(analysis)
    gran = b.get(TissueClass.GRANULATION, 0.0)
    epi = b.get(TissueClass.EPITHELIAL, 0.0)
    slough = b.get(TissueClass.SLOUGH, 0.0)
    eschar = b.get(TissueClass.ESCHAR, 0.0)
    return ViabilitySplit(
        viable_pct=round((gran + epi) * 100, 1), non_viable_pct=round((slough + eschar) * 100, 1),
        granulation_pct=round(gran * 100, 1), epithelial_pct=round(epi * 100, 1),
        slough_pct=round(slough * 100, 1), eschar_pct=round(eschar * 100, 1),
    )


def tissue_areas_cm2(analysis: WoundAnalysis) -> tuple[TissueArea, ...]:
    fr = analysis.metrics.tissue.fractions
    areas_mm2 = analysis.metrics.tissue.areas_mm2 or {}
    total_cm2 = analysis.metrics.geometry.area_cm2
    out: list[TissueArea] = []
    for tc in (TissueClass.GRANULATION, TissueClass.EPITHELIAL, TissueClass.SLOUGH,
               TissueClass.ESCHAR, TissueClass.ADIPOSE):
        f = fr.get(tc, 0.0)
        if f <= 0.004:
            continue
        cm2: float | None = None
        if tc in areas_mm2:
            cm2 = round(areas_mm2[tc] / 100.0, 2)
        elif total_cm2 is not None:
            cm2 = round(total_cm2 * f, 2)
        out.append(TissueArea(tissue=tc, fraction=round(f, 3), area_cm2=cm2))
    return tuple(out)


def _under_detection(analysis: WoundAnalysis) -> bool:
    b = _bed_fractions(analysis)
    non_viable = b.get(TissueClass.SLOUGH, 0.0) + b.get(TissueClass.ESCHAR, 0.0)
    gran = b.get(TissueClass.GRANULATION, 0.0)
    return non_viable < _UNDERDETECT_NONVIABLE and gran >= _UNDERDETECT_GRAN


def _bed_descriptor(analysis: WoundAnalysis) -> tuple[str, str]:
    b = _bed_fractions(analysis)
    non_viable = b.get(TissueClass.SLOUGH, 0.0) + b.get(TissueClass.ESCHAR, 0.0)
    if non_viable >= _NONVIABLE_HIGH:
        return ("Ağırlıklı cansız doku · Predominantly non-viable", "#B4232A")
    if non_viable >= _NONVIABLE_CAUTION:
        return ("Karışık doku · Mixed viable / non-viable", "#D97706")
    if _under_detection(analysis):
        return ("Görünürde granülasyon (bkz. uyarı) · Appears granulating (see caveat)", "#65A30D")
    return ("Ağırlıklı granülasyon/epitel · Predominantly granulating", "#16A34A")


def _necrotic(analysis: WoundAnalysis) -> tuple[str, str]:
    vs = viability_split(analysis)
    nv = vs.non_viable_pct / 100.0
    if nv >= _NONVIABLE_HIGH:
        return ("high", f"nekroz {vs.non_viable_pct:.0f}% · eskar {vs.eschar_pct:.0f}%")
    if nv >= _NONVIABLE_CAUTION:
        return ("caution", f"nekroz {vs.non_viable_pct:.0f}% · slough {vs.slough_pct:.0f}%")
    return ("none", "")


def _ratio(analysis: WoundAnalysis) -> float | None:
    b = _bed_fractions(analysis)
    slough = b.get(TissueClass.SLOUGH, 0.0)
    if slough < 0.01:
        return None  # not meaningful; do not render a reassuring ">10"
    return round(b.get(TissueClass.GRANULATION, 0.0) / slough, 1)


def grading_stats(analysis: WoundAnalysis) -> GradingStats:
    g = analysis.metrics.geometry
    if g.length_mm and g.width_mm and g.area_cm2 is not None:
        size = f"{g.length_mm / 10:.1f} × {g.width_mm / 10:.1f} cm · {g.area_cm2:.2f} cm²"
    else:
        size = f"{g.length_px:.0f} × {g.width_px:.0f} px"
    desc, color = _bed_descriptor(analysis)
    level, detail = _necrotic(analysis)
    return GradingStats(
        viability=viability_split(analysis),
        tissue_areas=tissue_areas_cm2(analysis),
        granulation_slough_ratio=_ratio(analysis),
        bed_descriptor=desc, bed_descriptor_color=color,
        under_detection=_under_detection(analysis),
        standard_size=size, necrotic_level=level, necrotic_detail=detail,
        intact_skin_pct=round(
            analysis.metrics.tissue.fractions.get(TissueClass.INTACT_SKIN, 0.0) * 100, 1),
    )


# --------------------------------------------------------------------------- #
# Follow-up
# --------------------------------------------------------------------------- #
def healing_velocity_band(trend: HealingTrend) -> tuple[str, str]:
    """Return ``(band, colour)`` from the weekly % area change."""
    r = trend.healing_rate_pct_per_week
    if r is None:
        return ("Belirsiz · Indeterminate", "#8A93A0")
    if r <= _VEL_HEALING:
        return ("Hızlı iyileşme · Healing", "#16A34A")
    if r <= _VEL_STALL:
        return ("Yavaş iyileşme · Slow", "#65A30D")
    if r < abs(_VEL_STALL):
        return ("Duraklamış · Stalled", "#D97706")
    return ("Kötüleşiyor · Deteriorating", "#DC2626")

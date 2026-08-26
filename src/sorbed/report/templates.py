"""HTML templates for the grading and follow-up reports.

Both builders return a complete, self-contained HTML document (fonts, images,
and vector charts inlined) ready for :func:`sorbed.report.pdf.html_to_pdf`. They
are pure assembly over already-rendered ``data:`` URIs, inline SVG charts, and a
:class:`~sorbed.guidelines.match.GuidelineContext`, so they carry no model or
I/O dependencies.
"""

from __future__ import annotations

from collections.abc import Mapping
from html import escape

from sorbed.domain.analysis import WoundAnalysis
from sorbed.domain.decision import StageDecision
from sorbed.domain.enums import PressureInjuryStage, TissueClass
from sorbed.guidelines.match import GuidelineContext, TopicBlock
from sorbed.report import svg
from sorbed.report.stats import GradingStats, grading_stats, healing_velocity_band
from sorbed.report.theme import (
    ALERT_COLORS,
    base_css,
    confidence_tier,
    stage_theme,
)
from sorbed.trend.alerts import HealingAssessment
from sorbed.trend.models import HealingTrend
from sorbed.visualize.palette import TISSUE_COLORS

_TISSUE_LABEL: dict[str, str] = {
    "granulation": "Granülasyon · Granulation",
    "slough": "Fibrin/slough · Slough",
    "eschar": "Eskar · Eschar",
    "epithelial": "Epitel · Epithelial",
    "adipose": "Yağ dokusu · Adipose",
    "muscle": "Kas · Muscle",
    "tendon_bone": "Tendon/kemik · Tendon/bone",
    "intact_skin": "Sağlam deri · Intact skin",
    "unknown": "Belirsiz · Unknown",
}

# Compact chip form (bilingual short) for tight cells.
_STAGE_SHORT: dict[str, str] = {
    "stage_1": "Evre 1 · St. 1",
    "stage_2": "Evre 2 · St. 2",
    "stage_3": "Evre 3 · St. 3",
    "stage_4": "Evre 4 · St. 4",
    "unstageable": "Sınıflandırılamayan · Unstageable",
    "deep_tissue_injury": "Derin doku · DTI",
    "mucosal_not_stageable": "Mukozal · Mucosal",
    "not_pressure_injury": "Basınç yarası değil · Not PI",
    "indeterminate": "Belirsiz · Indeterminate",
}


def _hex(rgb: tuple[int, int, int]) -> str:
    return f"#{rgb[0]:02x}{rgb[1]:02x}{rgb[2]:02x}"


def _tlabel(tc: TissueClass) -> str:
    return _TISSUE_LABEL.get(tc.value, tc.value.replace("_", " ").title())


def _fmt(v: float | None, nd: int = 1, dash: str = "—") -> str:
    return dash if v is None else f"{v:.{nd}f}"


def _doc(css: str, body: str, title: str) -> str:
    return (
        "<!doctype html><html lang='tr'><head><meta charset='utf-8'>"
        f"<title>{escape(title)}</title><style>{css}</style></head>"
        f"<body>{body}</body></html>"
    )


def _brandrow(ctx: GuidelineContext, kicker: str) -> str:
    ref = ""
    if ctx.available and ctx.meta:
        m = ctx.meta
        parts = [escape(m.doc_name)]
        if m.doc_no:
            parts.append(f"{escape(m.doc_no)} · Rev {escape(m.revision_no or '')}")
        if m.revision_date:
            parts.append(escape(m.revision_date))
        ref = "<br>".join(parts)
    return (
        "<div class='brandrow'>"
        "<div class='brand'><span class='logo'></span>"
        "<div><div class='name'>Sorbed</div>"
        f"<div class='kick'>{escape(kicker)}</div></div></div>"
        f"<div class='docref'>{ref}</div></div>"
    )


def _confbox(d: StageDecision) -> str:
    if d.abstained:
        return (
            "<div class='confbox'>"
            "<div class='confhead'><span class='muted'>Karar · Decision</span>"
            "<span class='confpct' style='color:#DC2626'>Geri çekildi · Withheld</span></div>"
            "<div class='confbar'><div class='conffill' style='width:100%;background:#DC2626;opacity:.25'></div></div>"
            "<div class='small muted' style='margin-top:4px'>İnceleme gerekli · needs review</div></div>"
        )
    tier, color = confidence_tier(d.confidence)
    pct = round(d.confidence * 100)
    return (
        "<div class='confbox'>"
        f"<div class='confhead'><span class='muted'>{escape(tier)}</span>"
        f"<span class='confpct' style='color:{color}'>{pct}%</span></div>"
        f"<div class='confbar'><div class='conffill' style='width:{pct}%;background:{color}'></div></div>"
        "<div class='small muted' style='margin-top:4px'>model skoru, kalibre olasılık değil · model score</div></div>"
    )


def _action_banner(d: StageDecision) -> str:
    """A single, actionable banner — shown only when a clinician must act.

    Consolidates the depth/abstention caveat and the next-action line into one
    strip so the report carries at most one prominent banner, and only when
    something must be done (not a wall of informational notices).
    """
    if not d.requires_clinician_review:
        return ""
    if d.abstained:
        reason = ("Model kesin bir evre vermedi. · The model withheld a definite grade.")
    elif d.stage.is_depth_dependent:
        reason = ("Bu evre doku derinliğine bağlıdır ve 2B fotoğraf bunu doğrulayamaz. · This grade is "
                  "depth-dependent and a 2D photo cannot confirm it.")
    else:
        reason = ("Düşük güven. · Low confidence.")
    return (
        "<div class='actbanner'><span class='actdot'></span>"
        "<div><b>Klinisyen doğrulaması gerekli · Clinician confirmation required.</b> "
        f"{escape(reason)} Yatak başı değerlendirme yapılmadan kesin evre kaydetmeyin. · Do not record "
        "a definitive stage without bedside assessment.</div></div>"
    )


def _disclaimer_line() -> str:
    return (
        "<div class='disc'><b>Karar desteği — tanı değildir · Decision support — not a diagnosis.</b> "
        "Tek bir 2B fotoğraftan geçici tahmin; klinisyen doğrulaması gerekir. · Provisional estimate "
        "from one 2D photo; requires clinician confirmation.</div>"
    )


def _pill(label: str, color: str | None = None) -> str:
    dot = f"<span class='dot' style='background:{color}'></span>" if color else ""
    return f"<span class='pill'>{dot}{escape(label)}</span>"


def _metric(k: str, v: str, u: str = "") -> str:
    unit = f" <span class='u'>{escape(u)}</span>" if u else ""
    return f"<div class='metric'><div class='k'>{escape(k)}</div><div class='v mono'>{v}{unit}</div></div>"


def _tissue_bar(fractions: Mapping[TissueClass, float]) -> str:
    # Include intact peri-wound skin so the bar accounts for the whole frame and
    # the "other" component is visible, not a silent gap. Background is excluded.
    items = [(tc, f) for tc, f in fractions.items()
             if f > 0.004 and tc != TissueClass.BACKGROUND]
    items.sort(key=lambda kv: kv[1], reverse=True)
    total = sum(f for _, f in items) or 1.0
    segs, legend = [], []
    for tc, f in items:
        color = _hex(TISSUE_COLORS.get(tc, (130, 130, 130)))
        segs.append(f"<span style='width:{f / total * 100:.2f}%;background:{color}'></span>")
        legend.append(
            f"<div class='item'><span class='swatch' style='background:{color}'></span>"
            f"{escape(_tlabel(tc))} <b class='mono'>{f * 100:.0f}%</b></div>"
        )
    if not segs:
        return "<p class='muted small'>Doku bileşimi belirlenemedi.</p>"
    return (f"<div class='tissuebar'>{''.join(segs)}</div>"
            f"<div class='legend'>{''.join(legend)}</div>")


def _cite_line(block: TopicBlock | None) -> str:
    if not block or not block.citations:
        return ""
    refs = ", ".join(f"§{c.section} (s.{c.page})" for c in block.citations)
    return f"<div class='ref'>Kaynak: {escape(refs)}</div>"


def _directive_figure_block(block: TopicBlock | None, fig_uri: str | None) -> str:
    if not block:
        return ""
    quote = f"<div class='q'>{escape(block.text)}</div>{_cite_line(block)}"
    if fig_uri:
        cap = escape(block.caption or "")
        return (
            "<div class='directivefig'>"
            f"<figure class='fig'><img src='{fig_uri}'><figcaption>{cap}</figcaption></figure>"
            f"<div style='padding-top:2px'>{quote}</div>"
            "</div>"
        )
    return f"<div class='cite'>{quote}</div>"


def _card(title: str, inner: str, *, cls: str = "", sub: str = "") -> str:
    extra = f" {cls}" if cls else ""
    subhtml = f"<span class='cardsub'>{escape(sub)}</span>" if sub else ""
    return (
        f"<section class='card{extra}'><div class='cardtitle'><span class='bar'></span>"
        f"<h2>{escape(title)}</h2>{subhtml}</div>{inner}</section>"
    )


# NPIAP/EPUAP 2019 English-equivalent stage definitions, shown alongside the
# directive's verbatim Turkish so an English reader can follow. Labelled as the
# international equivalent, not a translation of the source document.
_STAGE_EN: dict[str, str] = {
    "stage_1": "Stage 1 — intact skin with non-blanchable erythema; may be preceded by "
               "colour change, warmth, oedema, or firmness. Harder to detect in dark skin.",
    "stage_2": "Stage 2 — partial-thickness skin loss exposing dermis; a shallow open ulcer or "
               "intact/ruptured blister. No slough, granulation, or deeper tissue.",
    "stage_3": "Stage 3 — full-thickness skin loss; subcutaneous fat may be visible, with possible "
               "slough, undermining, and tunnelling. Depth varies by anatomical site.",
    "stage_4": "Stage 4 — full-thickness skin and tissue loss with exposed or palpable fascia, "
               "muscle, tendon, or bone; often with slough/eschar, undermining, and tunnelling.",
    "unstageable": "Unstageable — full-thickness loss where the wound bed is obscured by slough "
                   "and/or eschar; the true depth cannot be determined until it is debrided.",
    "deep_tissue_injury": "Deep tissue pressure injury — intact or non-intact skin with persistent "
                          "non-blanchable deep-red, maroon, or purple discolouration.",
}
_RYB_EN = ("Red-Yellow-Black tissue model: red = granulation/epithelial healing tissue (desirable); "
           "yellow = fibrinous slough, a sign of infection to treat; black = necrotic tissue to debride.")


def _stage_likelihood(analysis: WoundAnalysis) -> str:
    probs = analysis.decision.ml_stage_probabilities or {}
    if not probs:
        return ""
    ranked = sorted(probs.items(), key=lambda kv: kv[1], reverse=True)[:4]
    rows = []
    for stage, p in ranked:
        st = stage if isinstance(stage, PressureInjuryStage) else PressureInjuryStage(stage)
        color = stage_theme(st)[0]
        rows.append((_STAGE_SHORT.get(st.value, st.value), float(p), color))
    return svg.hbars_svg(rows)


def _viability_bar(s: GradingStats) -> str:
    # Encode the MEANING (viable vs non-viable), not the tissue hue: green = viable
    # (granulation + epithelial), red = non-viable (slough + eschar). A fully
    # viable bed then reads green, not an alarming solid red.
    v, nv = s.viability.viable_pct, s.viability.non_viable_pct
    total = (v + nv) or 100.0
    segs = []
    if v > 0.4:
        segs.append(f"<span style='width:{v / total * 100:.1f}%;background:#16A34A'></span>")
    if nv > 0.4:
        segs.append(f"<span style='width:{nv / total * 100:.1f}%;background:#B4232A'></span>")
    if not segs:
        segs.append("<span style='width:100%;background:#e9ecf1'></span>")
    return (f"<div class='tissuebar'>{''.join(segs)}</div>"
            f"<div class='vsplit'><span class='good'>● Canlı · Viable "
            f"<b class='mono'>~{v:.0f}%</b></span>"
            f"<span class='bad'>● Cansız · Non-viable "
            f"<b class='mono'>~{nv:.0f}%</b></span></div>")


def _clinical_stats_card(analysis: WoundAnalysis) -> str:
    s = grading_stats(analysis)
    ratio = ("Slough saptanmadı · not detected — oran anlamlı değil"
             if s.granulation_slough_ratio is None else f"~{_fmt(s.granulation_slough_ratio, 1)}")
    tiles = "".join([
        f"<div class='metric'><div class='k'>Standart ölçü · Size</div>"
        f"<div class='v mono' style='font-size:12px'>{escape(s.standard_size)}</div></div>",
        f"<div class='metric'><div class='k'>Yara yatağı · Wound bed</div>"
        f"<div class='v' style='color:{s.bed_descriptor_color};font-size:11px;line-height:1.25'>"
        f"{escape(s.bed_descriptor)}</div></div>",
        f"<div class='metric'><div class='k'>Gran./Slough oranı · ratio</div>"
        f"<div class='v mono' style='font-size:11px'>{ratio}</div></div>",
    ])
    rows = "".join(
        f"<div class='fi'><span>{escape(_tlabel(t.tissue))}</span>"
        f"<span><b class='mono'>~{t.fraction * 100:.0f}%</b>"
        f"{f' · <b class=mono>~{t.area_cm2:.2f} cm²</b>' if t.area_cm2 is not None else ''}</span></div>"
        for t in s.tissue_areas
    )
    # Under-detection notice — the core safety fix: a slough/eschar-free bed on a
    # deep wound is usually a classifier miss, not a clean wound.
    under = ""
    if s.under_detection:
        under = (
            "<div class='warnbox'><b>⚠ Slough/eskar saptanmadı · No slough/eschar detected.</b> "
            "Otomatik doku ayrımı ince fibrin/slough'u granülasyondan ayıramayabilir. Bunu temiz bir "
            "yara olarak okumayın — yatağı yatak başında doğrulayın. · Automated tissue typing may not "
            "separate thin fibrin/slough from granulation; do not read this as a clean wound — confirm "
            "the bed at the bedside.</div>"
        )
    # Necrotic burden — graded, only when present. No reassuring "no red flags".
    nec = ""
    if s.necrotic_level == "high":
        nec = ("<span class='pill' style='border-color:#F0B4B4;background:#FCEBEB;color:#9B2222'>"
               f"⚑ Yüksek nekrotik yük · High necrotic burden · {escape(s.necrotic_detail)}</span>")
    elif s.necrotic_level == "caution":
        nec = ("<span class='pill' style='border-color:#F4D9A8;background:#FCF3E3;color:#8A5A12'>"
               f"⚑ Nekrotik doku mevcut · Necrotic tissue present · {escape(s.necrotic_detail)}</span>")
    not_ruleout = (
        "<div class='mininote' style='margin-top:8px'>Otomatik uyarı olmaması güvenli demek değildir — "
        "enfeksiyon, alttan oyulma, daha derin doku tutulumu veya osteomiyeliti dışlamaz. · Absence of "
        "automated flags does not rule out infection, undermining, deeper-tissue involvement, or "
        "osteomyelitis.</div>"
        "<div class='mininote'>Alttan oyulma/tünel ve derinlik fotoğraftan değerlendirilemez — kenarları "
        "yatak başında sonda ile kontrol edin. · Undermining/tunnelling and depth cannot be judged from a "
        "photo — probe the edges at the bedside.</div>"
    )
    denom = ("<div class='mininote'>Yüzdeler açık yara yatağına göredir (sağlam çevre deri hariç; "
             f"çevre/margin ~{s.intact_skin_pct:.0f}%). · Percentages are of the open wound bed "
             "(excludes intact peri-wound skin).</div>")
    return _card(
        "Klinik istatistikler · Clinical statistics",
        "<div class='estnote'>Model tahminleri, ölçüm değildir · Model estimates, not measurements</div>"
        f"{under}"
        "<div class='mininote'>Doku canlılığı · Tissue viability</div>"
        f"{_viability_bar(s)}"
        f"<div class='grid3 tight' style='margin-top:11px'>{tiles}</div>"
        f"<div class='findlist' style='margin-top:6px'>{rows}</div>"
        + (f"<div style='margin-top:9px'>{nec}</div>" if nec else "")
        + not_ruleout + denom,
        cls="avoidbreak",
        sub="TIME · wound-bed preparation",
    )


def _guideline_compare_card(
    analysis: WoundAnalysis, guideline_ctx: GuidelineContext, images: Mapping[str, str]
) -> str:
    if not (guideline_ctx.available and guideline_ctx.stage):
        return ""
    block = guideline_ctx.stage
    fig = images.get("directive_stage")
    en = _STAGE_EN.get(analysis.decision.stage.value, "")
    src = (
        "<div class='srcbox'><span class='ribbon src'>◆ Kaynak doküman · Source guideline</span>"
        + (f"<figure class='cmpfig'><img src='{fig}'>"
           f"<figcaption>{escape(block.caption or '')}</figcaption></figure>" if fig else "")
        + f"<div class='q' style='font-size:9px;margin-top:8px'>{escape(block.text[:520])}</div>"
        + _cite_line(block)
        + (f"<div class='enrow'><b>EN (NPIAP eşdeğeri):</b> {escape(en)}</div>" if en else "")
        + "</div>"
    )
    _, __, stage_label = stage_theme(analysis.decision.stage)
    s = grading_stats(analysis)
    findings = "".join([
        f"<div class='fi'><span>Evre · Stage</span><b>{escape(stage_label)}</b></div>",
        f"<div class='fi'><span>Güven · Confidence</span><b class='mono'>{round(analysis.decision.confidence*100)}%</b></div>",
        f"<div class='fi'><span>Ölçü · Size</span><b class='mono'>{escape(s.standard_size)}</b></div>",
        f"<div class='fi'><span>Canlı doku · Viable</span><b class='mono'>{s.viability.viable_pct:.0f}%</b></div>",
        f"<div class='fi'><span>Baskın doku · Dominant</span><b>{escape(_tlabel(analysis.metrics.tissue.dominant))}</b></div>",
    ])
    schem = images.get("schematic")
    gen = (
        "<div class='genbox'><span class='ribbon gen'>◆ Sorbed · Üretilen · Generated</span>"
        + (f"<figure class='cmpfig'><img src='{schem}'>"
           "<figcaption>Model şeması · Generated schematic</figcaption></figure>" if schem else "")
        + f"<div class='findlist' style='margin-top:8px'>{findings}</div>"
        + "</div>"
    )
    return _card(
        "Kılavuz karşılaştırması · Guideline comparison",
        f"<p class='muted small' style='margin:-2px 0 10px'>Soldaki kaynak dokümanın şema ve tanımı, "
        "sağdaki Sorbed'in aynı yara için ürettiği çıktı — doğrudan karşılaştırma için.</p>"
        f"<div class='compare'>{src}{gen}</div>",
        cls="avoidbreak",
    )


def _generated_visuals_card(images: Mapping[str, str]) -> str:
    pairs = [("mask", "İkili maske · Binary mask"),
             ("schematic", "Şema · Synthetic schematic"),
             ("overlay", "Doku overlay · Tissue overlay"),
             ("detection", "Tespit · Detection"),
             ("depth", "Göreli derinlik · Relative depth")]
    figs = [f"<figure><img src='{images[k]}'><figcaption>{escape(lab)}</figcaption></figure>"
            for k, lab in pairs if k in images]
    if not figs and "dashboard" not in images:
        return ""
    thumbs = f"<div class='thumbrow'>{''.join(figs)}</div>" if figs else ""
    dash = (f"<figure class='dashfig'><img src='{images['dashboard']}'>"
            "<figcaption>Analiz paneli · Analysis dashboard</figcaption></figure>"
            if "dashboard" in images else "")
    return _card(
        "Üretilen analiz görselleri · Generated analysis outputs",
        thumbs + dash,
        cls="avoidbreak",
        sub="tek fotoğraftan · from one photo",
    )


# --------------------------------------------------------------------------- #
# Grading report
# --------------------------------------------------------------------------- #
def build_grading_report_html(
    *,
    analysis: WoundAnalysis,
    guideline_ctx: GuidelineContext,
    images: Mapping[str, str],
    patient_ref: str | None = None,
    generated: str = "",
) -> str:
    d = analysis.decision
    g = analysis.metrics.geometry
    hs = analysis.metrics.healing_scores
    accent, soft, stage_label = stage_theme(d.stage)
    css = base_css(accent, soft)

    sub = escape((d.narrative or "").split(". ")[0][:150]) if d.narrative else ""
    tone = analysis.skin_tone_band.value.replace("_", " ").title() if analysis.skin_tone_band else "—"
    calib = "kalibre · cm²" if g.area_cm2 is not None else "kalibresiz · px"
    pills = "".join([
        _pill(f"Cilt tonu · Skin tone · {tone}"),
        _pill(calib),
        _pill(f"ID {str(analysis.analysis_id)[:8]}"),
    ])
    if patient_ref:
        pills = _pill(patient_ref) + pills
    sub_html = (f"<div class='sub'><span class='eng'>Neden · Why: </span>{sub}</div>" if sub else "")
    hero = (
        "<div class='hero'>"
        f"{_brandrow(guideline_ctx, 'pressure-injury analysis')}"
        "<div class='gradewrap'>"
        "<div class='gradeblock'>"
        "<div class='eyebrow'>Otomatik evreleme · Auto grade</div>"
        f"<div class='grade'>{escape(stage_label)}</div>"
        f"{sub_html}"
        f"<div class='pillrow'>{pills}</div>"
        "</div>"
        f"{_confbox(d)}"
        "</div>"
        f"{_action_banner(d)}"
        "</div>"
    )
    disclaimer = _disclaimer_line()

    # Left column: imagery. Photo large, model panels beneath.
    photo = (f"<figure class='big'><img src='{images['photo']}'>"
             "<figcaption>Yüklenen görüntü · Uploaded photo</figcaption></figure>"
             if "photo" in images else "")
    small = []
    labels = {"overlay": "Doku · Tissue", "detection": "Tespit · Detection",
              "depth": "Derinlik · Depth"}
    for key in ("overlay", "detection", "depth"):
        if key in images:
            small.append(f"<figure><img src='{images[key]}'>"
                         f"<figcaption>{escape(labels[key])}</figcaption></figure>")
    left = (f"<div class='imgstack'>{photo}<div class='trio'>{''.join(small)}</div></div>")

    # Right column: measurements + stage likelihood.
    area_v = f"{g.area_cm2:.2f}" if g.area_cm2 is not None else f"{g.area_px:.0f}"
    area_u = "cm²" if g.area_cm2 is not None else "px"
    lw = (f"{g.length_mm:.0f}×{g.width_mm:.0f}" if g.length_mm and g.width_mm
          else f"{g.length_px:.0f}×{g.width_px:.0f}")
    lw_u = "mm" if g.length_mm else "px"
    gran = _fmt(hs.granulation_percent if hs else None, 0)
    push = str(hs.push_partial_total) if hs and hs.push_partial_total is not None else "—"
    metrics = "".join([
        _metric("Yüzey alanı", area_v, area_u),
        _metric("U×G", lw, lw_u),
        _metric("Çevre", f"{g.perimeter_px:.0f}", "px"),
        _metric("Granülasyon", gran, "%"),
        _metric("PUSH", push, "/17"),
        _metric("Baskın doku", escape(_tlabel(analysis.metrics.tissue.dominant))),
    ])
    like = _stage_likelihood(analysis)
    if like:
        extra = f"<div class='mininote'>Evre olasılıkları · Stage likelihood</div>{like}"
    else:
        # No ML stage posterior available — fill the column with the tissue mix,
        # which is always informative and grounds the RYB card below.
        extra = ("<div class='mininote'>Doku bileşimi · Tissue composition</div>"
                 + _tissue_bar(analysis.metrics.tissue.fractions))
    right = f"<div class='grid2 tight'>{metrics}</div>{extra}"

    top_card = _card(
        "Görüntü, ölçüm ve model çıktısı · Image, metrics & model output",
        f"<div class='splitcol'><div>{left}</div><div>{right}</div></div>",
        cls="avoidbreak",
    )

    # RYB tissue-colour model (directive §4.5.9) + per-tissue readings.
    tissue_card = ""
    if guideline_ctx.available and guideline_ctx.tissue_model:
        readings = "".join(
            f"<div class='item'><span class='swatch' style='background:"
            f"{_hex(TISSUE_COLORS.get(tr.tissue, (130, 130, 130)))}'></span>"
            f"{escape(_tlabel(tr.tissue))} · {tr.ryb} — {escape(tr.directive_note)}</div>"
            for tr in guideline_ctx.tissue_readings
        )
        inner = (
            f"<div class='cite'><div class='q'>{escape(guideline_ctx.tissue_model.text)}</div>"
            f"{_cite_line(guideline_ctx.tissue_model)}</div>"
            f"<div class='legend' style='margin-top:8px'>{readings}</div>"
        )
        tissue_card = _card("Doku rengi modeli · Tissue-colour model (RYB · §4.5.9)",
                            inner, cls="avoidbreak")

    stats_card = _clinical_stats_card(analysis)
    compare_card = _guideline_compare_card(analysis, guideline_ctx, images)
    visuals_card = _generated_visuals_card(images)

    care_card = ""
    if guideline_ctx.available:
        bits = []
        if guideline_ctx.reassessment_open:
            ref = ""
            if guideline_ctx.reassessment_citation:
                c = guideline_ctx.reassessment_citation
                ref = f" <span class='muted'>(§{c.section}, s.{c.page})</span>"
            bits.append(f"<p><b>Yeniden değerlendirme (açık yara):</b> "
                        f"{escape(guideline_ctx.reassessment_open)}{ref}</p>")
        if guideline_ctx.care and guideline_ctx.care.text:
            bits.append(f"<div class='cite' style='margin-top:6px'><div class='q'>"
                        f"{escape(guideline_ctx.care.text[:820])}</div>{_cite_line(guideline_ctx.care)}</div>")
        if bits:
            care_card = _card("Bakım ve yeniden değerlendirme · Care & reassessment",
                              "".join(bits), cls="avoidbreak")

    # Order (per clinical-informatics review): validated guideline grounding
    # leads; derived indices follow under caveats; generated visuals last.
    body = (hero + disclaimer + top_card + compare_card + stats_card
            + tissue_card + care_card + visuals_card
            + _footer(guideline_ctx, generated, provenance=_provenance(analysis), limitations=True))
    return _doc(css, body, "Sorbed · Basınç Yaralanması Değerlendirme")


# --------------------------------------------------------------------------- #
# Compact single-image "normal" report (one page)
# --------------------------------------------------------------------------- #
def build_summary_report_html(
    *,
    analysis: WoundAnalysis,
    guideline_ctx: GuidelineContext,
    images: Mapping[str, str],
    patient_ref: str | None = None,
    generated: str = "",
) -> str:
    """A concise one-page single-image report for everyday per-upload use."""
    d = analysis.decision
    g = analysis.metrics.geometry
    hs = analysis.metrics.healing_scores
    accent, soft, stage_label = stage_theme(d.stage)
    css = base_css(accent, soft)
    s = grading_stats(analysis)

    tone = analysis.skin_tone_band.value.replace("_", " ").title() if analysis.skin_tone_band else "—"
    pills = "".join([
        _pill(patient_ref) if patient_ref else "",
        _pill(f"Cilt tonu · Skin tone · {tone}"),
        _pill("kalibre · cm²" if g.area_cm2 is not None else "kalibresiz · px"),
        _pill(f"ID {str(analysis.analysis_id)[:8]}"),
    ])
    hero = (
        "<div class='hero'>"
        f"{_brandrow(guideline_ctx, 'pressure-injury analysis')}"
        "<div class='gradewrap'><div class='gradeblock'>"
        "<div class='eyebrow'>Otomatik evreleme · Auto grade</div>"
        f"<div class='grade'>{escape(stage_label)}</div>"
        f"<div class='pillrow'>{pills}</div></div>"
        f"{_confbox(d)}</div>"
        f"{_action_banner(d)}</div>"
    )

    # Imagery: photo + tissue overlay.
    figs = []
    for key, lab in (("photo", "Yüklenen görüntü · Photo"), ("overlay", "Doku · Tissue overlay")):
        if key in images:
            figs.append(f"<figure><img src='{images[key]}'>"
                        f"<figcaption>{escape(lab)}</figcaption></figure>")
    left = f"<div class='figrow'>{''.join(figs)}</div>"

    area_v = f"{g.area_cm2:.2f}" if g.area_cm2 is not None else f"{g.area_px:.0f}"
    area_u = "cm²" if g.area_cm2 is not None else "px"
    gran = _fmt(hs.granulation_percent if hs else None, 0)
    push = str(hs.push_partial_total) if hs and hs.push_partial_total is not None else "—"
    metrics = "".join([
        _metric("Yüzey alanı · Area", f"~{area_v}", area_u),
        _metric("Ölçü · Size", escape(s.standard_size).split(" · ")[0], ""),
        _metric("Granülasyon", f"~{gran}", "%"),
        _metric("PUSH", push, "/17"),
    ])
    right = (f"<div class='grid2 tight'>{metrics}</div>"
             "<div class='mininote' style='margin-top:10px'>Doku canlılığı · Tissue viability</div>"
             f"{_viability_bar(s)}")
    if s.under_detection:
        right += ("<div class='mininote' style='margin-top:6px;color:#9b1c1c'>⚠ Slough/eskar saptanmadı — "
                  "temiz yara olarak okumayın. · No slough/eschar detected — do not read as a clean wound.</div>")
    summary_card = _card(
        "Analiz özeti · Analysis summary",
        f"<div class='splitcol'><div>{left}</div><div>{right}</div></div>",
        cls="avoidbreak",
    )

    # Brief stage criterion (text + citation only).
    stage_card = ""
    if guideline_ctx.available and guideline_ctx.stage and guideline_ctx.stage.text:
        en = _STAGE_EN.get(d.stage.value, "")
        stage_card = _card(
            "Evre kriteri · Stage criterion",
            f"<div class='cite'><div class='q'>{escape(guideline_ctx.stage.text[:480])}</div>"
            f"{_cite_line(guideline_ctx.stage)}</div>"
            + (f"<div class='mininote' style='margin-top:6px'><b>EN (NPIAP):</b> {escape(en)}</div>" if en else ""),
            cls="avoidbreak",
        )

    body = (hero + _disclaimer_line() + summary_card + stage_card
            + _footer(guideline_ctx, generated, provenance=_provenance(analysis), limitations=True))
    return _doc(css, body, "Sorbed · Tekli Görüntü Raporu")


# --------------------------------------------------------------------------- #
# Follow-up report
def _per_visit_section(trend: HealingTrend,
                       visit_images: Mapping[str, Mapping[str, str]]) -> str:
    """One card per visit: the photo + tissue overlay + that visit's analysis."""
    cards = []
    for p in trend.points:
        imgs = visit_images.get(p.label, {})
        acc = stage_theme(p.stage)[0]
        area = f"{p.area_cm2:.2f}" if p.area_cm2 is not None else f"{p.area_px:.0f}"
        gran = p.fraction(TissueClass.GRANULATION) * 100
        nec = (p.fraction(TissueClass.SLOUGH) + p.fraction(TissueClass.ESCHAR)) * 100
        push = str(p.push_total) if p.push_total is not None else "—"
        thumbs = ""
        if imgs.get("photo"):
            thumbs += (f"<figure><img src='{imgs['photo']}'>"
                       "<figcaption>Fotoğraf · Photo</figcaption></figure>")
        if imgs.get("overlay"):
            thumbs += (f"<figure><img src='{imgs['overlay']}'>"
                       "<figcaption>Doku · Tissue</figcaption></figure>")
        stats = "".join([
            f"<div class='fi'><span>Evre · Stage</span>"
            f"<span class='chip' style='background:{acc}'>"
            f"{escape(_STAGE_SHORT.get(p.stage.value, p.stage.value))}</span></div>",
            f"<div class='fi'><span>Alan · Area</span><b class='mono'>~{area} {trend.unit}</b></div>",
            f"<div class='fi'><span>Granülasyon · Granulation</span><b class='mono'>~{gran:.0f}%</b></div>",
            f"<div class='fi'><span>Nekroz · Necrosis</span><b class='mono'>~{nec:.0f}%</b></div>",
            f"<div class='fi'><span>PUSH</span><b class='mono'>{push}/17</b></div>",
            f"<div class='fi'><span>Güven · Confidence</span><b class='mono'>{round(p.confidence*100)}%</b></div>",
        ])
        cards.append(
            "<div class='visitcard'>"
            f"<div class='vc-head'><b>{escape(p.label)}</b>"
            f"<span class='muted'>gün · day {p.day:.0f}</span></div>"
            f"<div class='vc-imgs'>{thumbs}</div>"
            f"<div class='findlist'>{stats}</div>"
            "</div>"
        )
    if not cards:
        return ""
    return _card(
        "Ziyaret bazlı analiz · Per-visit analysis",
        f"<div class='visitgrid'>{''.join(cards)}</div>",
        cls="avoidbreak",
    )


# --------------------------------------------------------------------------- #
def build_followup_report_html(
    *,
    trend: HealingTrend,
    assessment: HealingAssessment,
    guideline_ctx: GuidelineContext,
    visit_thumbs: Mapping[str, str],
    visit_images: Mapping[str, Mapping[str, str]] | None = None,
    patient_ref: str | None = None,
    generated: str = "",
    attribution: str = "",
) -> str:
    latest_stage = trend.points[-1].stage
    accent, soft, _ = stage_theme(latest_stage)
    css = base_css(accent, soft)
    bannercolor = ALERT_COLORS.get(assessment.level, ALERT_COLORS["info"])

    hero = (
        "<div class='hero'>"
        f"{_brandrow(guideline_ctx, 'pressure-injury analysis')}"
        "<div class='gradeblock'>"
        "<div class='eyebrow'>İyileşme takibi · Healing follow-up</div>"
        f"<div class='grade' style='font-size:23px'>"
        f"{escape(patient_ref or trend.patient_ref or 'Wound trajectory')}</div>"
        f"<div class='sub'>{len(trend.points)} ziyaret · visits · "
        f"{trend.points[-1].day - trend.points[0].day:.0f} gün · days · birim · unit: {trend.unit}</div>"
        "</div></div>"
    )

    # Verdict banner + healing gauge side by side.
    gauge = svg.gauge_svg(trend.percent_area_reduction, accent="#16A34A"
                          if trend.percent_area_reduction >= 0 else "#DC2626",
                          caption="alan azalması · PAR")
    banner = (
        "<div class='verdictrow avoidbreak'>"
        f"<div class='banner' style='background:{bannercolor}'>"
        f"<div class='lvl'>{escape(assessment.level.upper())}</div>"
        f"<div class='hd'>{escape(assessment.headline)}</div>"
        f"<div class='sm'>{escape(assessment.summary)}</div></div>"
        f"<div class='gaugebox'>{gauge}</div>"
        "</div>"
    )

    vband, vcolor = healing_velocity_band(trend)
    likely = ("Evet · Yes" if trend.likely_to_heal
              else "Hayır · No" if trend.likely_to_heal is False else "—")
    tiles = "".join([
        _metric("Baz→son alan · Base→latest",
                f"{trend.baseline_area:.3g}→{trend.latest_area:.3g}", trend.unit),
        _metric("Haftalık hız · Weekly rate", _fmt(trend.healing_rate_pct_per_week, 1), "%/hf"),
        _metric("4-haftalık PAR · 4-wk PAR", _fmt(trend.par_at_4_weeks, 0), "%"),
        _metric("PUSH eğilimi · trend", escape((trend.push_trend or "—").title())),
        _metric("Kapanma · Closure (proj.)", _fmt(trend.projected_days_to_closure, 0), "gün"),
        _metric("İyileşme olasılığı · Likely to heal", likely),
    ])
    velocity = (
        f"<div class='metric' style='grid-column:1/-1;display:flex;align-items:center;"
        f"justify-content:space-between'><div><div class='k'>İyileşme hızı sınıfı · "
        f"Healing velocity</div><div class='v' style='color:{vcolor};font-size:15px'>{escape(vband)}</div></div>"
        f"<div class='mono muted' style='font-size:11px'>{_fmt(trend.healing_rate_pct_per_week, 1)} %/hafta</div></div>"
    )
    tiles_card = _card("Özet · Summary",
                       f"<div class='grid3 tight'>{tiles}</div>"
                       f"<div class='grid tight' style='margin-top:8px'>{velocity}</div>")

    # SVG charts.
    days = [p.day for p in trend.points]
    areas = [(p.area_cm2 if trend.unit == "cm2" and p.area_cm2 is not None else p.area_px)
             for p in trend.points]
    proj = None
    if trend.projected_days_to_closure is not None:
        proj = (days[-1] + trend.projected_days_to_closure, 0.0)
    area_svg = svg.line_chart_svg(days, areas, accent=accent, unit=trend.unit, projection=proj)

    push_pts = [(p.day, float(p.push_total)) for p in trend.points if p.push_total is not None]
    charts_html = [
        f"<figure class='chart'>{area_svg}<figcaption>Yüzey alanı · Wound area "
        f"({trend.unit}) — kesikli çizgi: projeksiyon</figcaption></figure>",
        f"<figure class='chart'>{svg.tissue_stack_svg(trend)}"
        "<figcaption>Doku bileşimi · Tissue mix (%)</figcaption></figure>",
    ]
    if len(push_pts) >= 2:
        push_svg = svg.line_chart_svg([x for x, _ in push_pts], [y for _, y in push_pts],
                                      accent="#0E9C9B", unit="")
        charts_html.append(
            f"<figure class='chart'>{push_svg}<figcaption>PUSH toplamı · PUSH total "
            "(↓ = iyileşme)</figcaption></figure>"
        )
    charts_card = _card("Eğilim grafikleri · Trend charts",
                        f"<div class='chartgrid'>{''.join(charts_html)}</div>", cls="avoidbreak")

    # Alerts
    alert_rows = []
    for a in assessment.alerts:
        color = ALERT_COLORS.get(a.level, ALERT_COLORS["info"])
        refs = ("  ·  " + ", ".join(f"§{r}" for r in a.directive_refs)) if a.directive_refs else ""
        alert_rows.append(
            f"<div class='alert'><div class='stripe' style='background:{color}'></div>"
            f"<div><div class='at'>{escape(a.title)}</div>"
            f"<div class='ad'>{escape(a.detail)}</div>"
            f"<div class='aref'>{escape(a.level.upper())}{escape(refs)}</div></div></div>"
        )
    alerts_card = _card("Uyarılar · Clinical alerts",
                        "".join(alert_rows) or "<p class='muted small'>Uyarı yok.</p>",
                        cls="avoidbreak")

    # Visit timeline table
    rows = []
    for p in trend.points:
        acc = stage_theme(p.stage)[0]
        thumb = visit_thumbs.get(p.label, "")
        img = (f"<img src='{thumb}' class='vthumb'>" if thumb else "")
        area = f"{p.area_cm2:.2f}" if p.area_cm2 is not None else f"{p.area_px:.0f}"
        gran = p.fraction(TissueClass.GRANULATION) * 100
        nec = (p.fraction(TissueClass.SLOUGH) + p.fraction(TissueClass.ESCHAR)) * 100
        push = str(p.push_total) if p.push_total is not None else "—"
        rows.append(
            f"<tr><td>{img}</td><td><b>{escape(p.label)}</b><br>"
            f"<span class='muted small'>gün {p.day:.0f}</span></td>"
            f"<td><span class='chip' style='background:{acc}'>"
            f"{escape(_STAGE_SHORT.get(p.stage.value, p.stage.value))}</span></td>"
            f"<td class='mono'>{area} {trend.unit}</td>"
            f"<td class='mono'>{gran:.0f}%</td><td class='mono'>{nec:.0f}%</td>"
            f"<td class='mono'>{push}</td><td class='mono'>{round(p.confidence * 100)}%</td></tr>"
        )
    table = (
        "<table class='visits'><thead><tr><th></th><th>Ziyaret</th><th>Evre</th>"
        "<th>Alan</th><th>Gran.</th><th>Nekroz</th><th>PUSH</th><th>Güven</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )
    visits_card = _card("Ziyaret zaman çizelgesi · Visit timeline", table, cls="avoidbreak")

    directive_card = ""
    if guideline_ctx.available:
        blocks = []
        for label, block in (("PUSH · §4.5.8", guideline_ctx.push),
                             ("Doku rengi · §4.5.9", guideline_ctx.tissue_model),
                             ("Boyut · §4.5.11", guideline_ctx.size)):
            if block and block.text:
                blocks.append(
                    f"<div class='cite col'><h3>{escape(label)}</h3>"
                    f"<div class='q' style='margin-top:4px'>{escape(block.text[:460])}</div>"
                    f"{_cite_line(block)}</div>"
                )
        if blocks:
            directive_card = _card("Talimat dayanağı · Directive basis",
                                   f"<div class='citegrid'>{''.join(blocks)}</div>", cls="avoidbreak")

    per_visit = _per_visit_section(trend, visit_images or {})
    body = (hero + banner + tiles_card + charts_card + alerts_card
            + per_visit + visits_card + directive_card
            + _footer(guideline_ctx, generated, attribution))
    return _doc(css, body, "Sorbed · İyileşme Takibi")


def _provenance(analysis: WoundAnalysis) -> str:
    p = analysis.provenance
    digest = (analysis.config_digest or "")[:12]
    parts = [f"seg={p.segmentation_backend}", f"tissue={p.tissue_backend}",
             f"stage={p.staging_backend}"]
    if digest:
        parts.append(f"config={digest}")
    parts.append(f"schema={analysis.schema_version}")
    return " · ".join(parts)


def _footer(ctx: GuidelineContext, generated: str, attribution: str = "",
            provenance: str = "", limitations: bool = False) -> str:
    bits = [
        "Sorbed karar destek yazılımıdır; tıbbi cihaz veya tanı aracı değildir. Tüm çıktılar "
        "klinisyen doğrulaması gerektirir. · Sorbed is decision-support software, not a medical device "
        "or a diagnostic tool; all outputs require clinician confirmation.",
    ]
    if limitations:
        bits.append(
            "Fotoğraftan değerlendirilemez · Not assessable from a photo: derinlik · depth, alttan "
            "oyulma/tünel · undermining/tunnelling, enfeksiyon · infection, perfüzyon · perfusion, "
            "osteomiyelit · osteomyelitis, yara etiyolojisi · wound etiology. Ayak yarasında diyabetik/"
            "nöropatik ve arteriyel etiyoloji ayrıca dışlanmalıdır · on the foot, also rule out "
            "diabetic/neuropathic and arterial etiology."
        )
    if provenance:
        bits.append(f"Model · {escape(provenance)}")
    if ctx.available and ctx.meta:
        bits.append(
            f"Klinik dayanak: {escape(ctx.meta.doc_name)} "
            f"({escape(ctx.meta.doc_no or '')} Rev {escape(ctx.meta.revision_no or '')}, "
            f"{escape(ctx.meta.revision_date or '')}). Alıntılar ve şekiller kaynak dokümandan alınmıştır."
        )
    if attribution:
        bits.append(escape(attribution))
    if generated:
        bits.append(f"Oluşturulma: {escape(generated)}")
    return "<div class='foot'>" + "<br>".join(bits) + "</div>"

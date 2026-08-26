"""Shared visual design system for printed reports.

One place defines the grade colour ramp, confidence tiers, alert palette, and
the print stylesheet, so the grading and follow-up reports read as one product.
The look is deliberately restrained: Manrope, generous whitespace, a single
accent tinted by the wound's grade, and colour used only to encode meaning.
"""

from __future__ import annotations

from sorbed.domain.enums import PressureInjuryStage
from sorbed.report.assets import manrope_font_face

# Clinical severity ramp — accent + soft tint per stage.
STAGE_THEME: dict[str, tuple[str, str, str]] = {
    PressureInjuryStage.STAGE_1.value: ("#E8912A", "#FDF3E3", "Evre 1 · Stage 1"),
    PressureInjuryStage.STAGE_2.value: ("#E4711C", "#FCEEE2", "Evre 2 · Stage 2"),
    PressureInjuryStage.STAGE_3.value: ("#DB4A4F", "#FBE9EA", "Evre 3 · Stage 3"),
    PressureInjuryStage.STAGE_4.value: ("#B21E27", "#F7E2E3", "Evre 4 · Stage 4"),
    PressureInjuryStage.DEEP_TISSUE.value: ("#7C3AED", "#F0EAFB", "Derin doku · Deep tissue injury"),
    PressureInjuryStage.UNSTAGEABLE.value: ("#4B5563", "#EBEDF0", "Sınıflandırılamayan · Unstageable"),
    PressureInjuryStage.MUCOSAL.value: ("#0E9C9B", "#E3F5F5", "Mukozal · Mucosal"),
    PressureInjuryStage.NOT_PRESSURE_INJURY.value: ("#6B7280", "#EEF0F2", "Basınç yarası değil"),
    PressureInjuryStage.INDETERMINATE.value: ("#8A93A0", "#EEF0F3", "Belirsiz · Indeterminate"),
}

ALERT_COLORS: dict[str, str] = {
    "critical": "#DC2626",
    "warning": "#D97706",
    "positive": "#16A34A",
    "info": "#4F46E5",
}


def stage_theme(stage: PressureInjuryStage) -> tuple[str, str, str]:
    return STAGE_THEME.get(stage.value, STAGE_THEME[PressureInjuryStage.INDETERMINATE.value])


def confidence_tier(conf: float) -> tuple[str, str]:
    """Return ``(label, colour)`` for a confidence value."""
    if conf >= 0.75:
        return ("Yüksek güven · High", "#16A34A")
    if conf >= 0.5:
        return ("Orta güven · Moderate", "#D97706")
    return ("Düşük güven · Low", "#DC2626")


def base_css(accent: str, accent_soft: str) -> str:
    """The full print stylesheet, parameterised by the grade accent."""
    return f"""
{manrope_font_face()}
:root {{
  --accent: {accent};
  --accent-soft: {accent_soft};
  --ink: #14161c;
  --muted: #626b7a;
  --line: #e7eaef;
  --paper: #ffffff;
  --panel: #f7f8fa;
  --crit: {ALERT_COLORS['critical']};
  --warn: {ALERT_COLORS['warning']};
  --ok: {ALERT_COLORS['positive']};
}}
* {{ box-sizing: border-box; }}
@page {{ size: A4; margin: 14mm 13mm 16mm 13mm; }}
html, body {{ margin: 0; padding: 0; }}
body {{
  font-family: 'Manrope', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
  color: var(--ink); background: var(--paper);
  font-size: 10.2px; line-height: 1.5; font-weight: 400;
  -webkit-font-smoothing: antialiased;
}}
h1, h2, h3 {{ margin: 0; font-weight: 800; letter-spacing: -0.02em; line-height: 1.12; }}
h2 {{ font-size: 15px; margin-bottom: 8px; }}
h3 {{ font-size: 11.5px; font-weight: 700; }}
p {{ margin: 0 0 6px; }}
.muted {{ color: var(--muted); }}
.mono {{ font-variant-numeric: tabular-nums; }}
.small {{ font-size: 8.7px; }}

/* ---- header / hero ---- */
.hero {{
  border-radius: 16px; padding: 18px 20px; margin-bottom: 14px;
  background: linear-gradient(135deg, var(--accent-soft), #ffffff 78%);
  border: 1px solid var(--line);
}}
.brandrow {{ display: flex; align-items: center; justify-content: space-between; margin-bottom: 14px; }}
.brand {{ display: flex; align-items: center; gap: 9px; }}
.logo {{ width: 22px; height: 22px; border-radius: 7px; background: var(--accent); }}
.brand .name {{ font-weight: 800; font-size: 13px; letter-spacing: -0.02em; }}
.brand .kick {{ font-size: 8.5px; color: var(--muted); letter-spacing: 0.14em; text-transform: uppercase; }}
.docref {{ text-align: right; font-size: 8.4px; color: var(--muted); line-height: 1.45; }}
.gradewrap {{ display: flex; align-items: flex-end; justify-content: space-between; gap: 18px; }}
.gradeblock .eyebrow {{ font-size: 8.6px; letter-spacing: 0.16em; text-transform: uppercase; color: var(--muted); margin-bottom: 4px; }}
.gradeblock .grade {{ font-size: 29px; font-weight: 800; letter-spacing: -0.03em; color: var(--accent); }}
.gradeblock .sub {{ font-size: 10px; color: var(--muted); margin-top: 2px; max-width: 62ch; }}
.pillrow {{ margin-top: 9px; display: flex; gap: 5px; flex-wrap: wrap; }}
.confbox {{ min-width: 168px; }}
.confhead {{ display: flex; justify-content: space-between; font-size: 8.8px; margin-bottom: 4px; }}
.confpct {{ font-weight: 800; }}
.confbar {{ height: 7px; border-radius: 99px; background: #e9ecf1; overflow: hidden; }}
.conffill {{ height: 100%; border-radius: 99px; }}
.pill {{
  display: inline-flex; align-items: center; gap: 5px; padding: 2px 9px; border-radius: 99px;
  font-size: 8.5px; font-weight: 700; border: 1px solid var(--line); background: #fff;
}}
.dot {{ width: 7px; height: 7px; border-radius: 99px; }}

/* ---- layout ---- */
.card {{ border: 1px solid var(--line); border-radius: 14px; padding: 14px 15px; margin-bottom: 12px; background: #fff; }}
.card > .cardtitle {{ display: flex; align-items: center; gap: 7px; margin-bottom: 10px; }}
.card > .cardtitle .bar {{ width: 3px; height: 15px; border-radius: 2px; background: var(--accent); }}
.card > .cardtitle .cardsub {{ margin-left: auto; font-size: 8.4px; color: var(--muted); font-weight: 600; }}
.grid {{ display: grid; gap: 10px; }}
.grid2 {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
.grid3 {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }}
.tight {{ gap: 8px; }}
.figrow {{ display: grid; grid-template-columns: 1fr 1fr; gap: 10px; }}
.figrow figure {{ margin: 0; }}
.figrow img, .fig img {{ width: 100%; border-radius: 10px; border: 1px solid var(--line); display: block; }}
figcaption {{ font-size: 8.3px; color: var(--muted); margin-top: 4px; }}

/* two-column grading layout */
.splitcol {{ display: grid; grid-template-columns: 1.12fr 1fr; gap: 15px; align-items: start; }}
.imgstack figure {{ margin: 0; }}
.imgstack .big img {{ width: 100%; border-radius: 11px; border: 1px solid var(--line); display: block; }}
.trio {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 7px; margin-top: 7px; }}
.trio figure {{ margin: 0; }}
.trio img {{ width: 100%; border-radius: 8px; border: 1px solid var(--line); display: block; }}
.mininote {{ font-size: 8.1px; letter-spacing: 0.08em; text-transform: uppercase; color: var(--muted); margin: 12px 0 4px; }}

/* follow-up verdict + gauge */
.verdictrow {{ display: grid; grid-template-columns: 1fr 208px; gap: 12px; align-items: stretch; margin-bottom: 12px; }}
.verdictrow .banner {{ margin: 0; display: flex; flex-direction: column; justify-content: center; }}
.gaugebox {{ border: 1px solid var(--line); border-radius: 14px; padding: 6px 10px; display: flex; align-items: center; justify-content: center; background: #fff; }}
.chartgrid {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
.chart {{ margin: 0; border: 1px solid var(--line); border-radius: 12px; padding: 10px 11px 5px; background: #fff; }}
.vthumb {{ width: 34px; height: 34px; border-radius: 7px; object-fit: cover; border: 1px solid var(--line); display: block; }}
/* per-visit analysis cards */
.visitgrid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 11px; }}
.visitcard {{ border: 1px solid var(--line); border-radius: 12px; padding: 10px 11px; background: var(--panel); }}
.vc-head {{ display: flex; align-items: baseline; justify-content: space-between; margin-bottom: 7px; }}
.vc-head b {{ font-size: 11px; }} .vc-head .muted {{ font-size: 8px; }}
.vc-imgs {{ display: grid; grid-template-columns: 1fr 1fr; gap: 6px; margin-bottom: 7px; }}
.vc-imgs figure {{ margin: 0; }}
.vc-imgs img {{ width: 100%; aspect-ratio: 1/1; object-fit: cover; border-radius: 7px; border: 1px solid var(--line); display: block; }}
.vc-imgs figcaption {{ font-size: 7px; color: var(--muted); margin-top: 2px; text-align: center; }}
.visitcard .findlist {{ font-size: 8.4px; }}
.visitcard .fi {{ padding: 2px 0; }}
.citegrid {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 10px; }}
.cite.col {{ margin: 0; }}

/* ---- metric tiles ---- */
.metric {{ background: var(--panel); border: 1px solid var(--line); border-radius: 11px; padding: 9px 11px; }}
.metric .k {{ font-size: 8.2px; letter-spacing: 0.06em; text-transform: uppercase; color: var(--muted); }}
.metric .v {{ font-size: 16px; font-weight: 800; letter-spacing: -0.02em; margin-top: 2px; }}
.metric .u {{ font-size: 8.6px; color: var(--muted); font-weight: 600; }}

/* ---- tissue composition bar ---- */
.tissuebar {{ display: flex; height: 20px; border-radius: 7px; overflow: hidden; border: 1px solid var(--line); }}
.tissuebar span {{ display: block; height: 100%; }}
.vsplit {{ display: flex; justify-content: space-between; font-size: 9.2px; margin-top: 7px; font-weight: 600; }}
.vsplit .good {{ color: #16A34A; }} .vsplit .bad {{ color: #B4232A; }}
.legend {{ display: flex; flex-wrap: wrap; gap: 5px 14px; margin-top: 9px; }}
.legend .item {{ display: flex; align-items: center; gap: 6px; font-size: 8.8px; }}
.legend .swatch {{ width: 9px; height: 9px; border-radius: 3px; }}

/* ---- guideline comparison: source document vs generated ---- */
.compare {{ display: grid; grid-template-columns: 1fr 1fr; gap: 12px; }}
.srcbox {{ position: relative; border: 1px solid #dfd3b8; border-radius: 12px; padding: 12px 13px 13px;
  background: repeating-linear-gradient(0deg,#fcf8ee,#fcf8ee 22px,#f7f1e2 23px); }}
.genbox {{ position: relative; border: 1px solid var(--line); border-radius: 12px; padding: 12px 13px 13px;
  background: var(--accent-soft); }}
.ribbon {{ display: inline-flex; align-items: center; gap: 5px; font-size: 7.6px; font-weight: 800;
  letter-spacing: 0.1em; text-transform: uppercase; padding: 3px 8px; border-radius: 99px; margin-bottom: 8px; }}
.ribbon.src {{ background: #7a5c12; color: #fff; }}
.ribbon.gen {{ background: var(--accent); color: #fff; }}
.cmpfig img {{ width: 100%; border-radius: 8px; border: 1px solid var(--line); display: block; }}
.cmpfig figcaption {{ font-size: 8px; color: var(--muted); margin-top: 3px; }}
.enrow {{ font-size: 8.7px; color: var(--muted); margin-top: 6px; padding-top: 6px; border-top: 1px dashed #dfd3b8; }}
.enrow b {{ color: var(--ink); }}
.findlist {{ font-size: 9px; margin-top: 8px; }}
.findlist .fi {{ display: flex; justify-content: space-between; gap: 8px; padding: 3px 0; border-bottom: 1px solid rgba(0,0,0,.05); }}
.findlist .fi:last-child {{ border-bottom: 0; }}
.findlist .fi b {{ font-variant-numeric: tabular-nums; }}

/* ---- generated-visuals gallery ---- */
.thumbrow {{ display: grid; grid-template-columns: repeat(5, 1fr); gap: 9px; margin-bottom: 11px; }}
.thumbrow figure {{ margin: 0; }}
.thumbrow img {{ width: 100%; aspect-ratio: 1/1; object-fit: cover; border-radius: 8px; border: 1px solid var(--line); display: block; }}
.thumbrow figcaption {{ font-size: 7.6px; color: var(--muted); margin-top: 4px; text-align: center; }}
.dashfig {{ margin: 0; }}
.dashfig img {{ width: 100%; border-radius: 10px; border: 1px solid var(--line); display: block; }}
.dashfig figcaption {{ font-size: 8px; color: var(--muted); margin-top: 4px; }}

/* ---- directive citation ---- */
.cite {{ border-left: 3px solid var(--accent); background: var(--accent-soft); border-radius: 0 10px 10px 0; padding: 9px 12px; }}
.cite .q {{ font-size: 9.2px; line-height: 1.5; }}
.cite .ref {{ font-size: 8.1px; color: var(--muted); margin-top: 5px; font-weight: 700; }}
.directivefig {{ display: grid; grid-template-columns: 128px 1fr; gap: 12px; align-items: start; }}
.directivefig img {{ width: 100%; border-radius: 9px; border: 1px solid var(--line); }}

/* ---- alerts / banners ---- */
.banner {{ border-radius: 14px; padding: 15px 17px; margin-bottom: 13px; color: #fff; }}
.banner .lvl {{ font-size: 8.6px; letter-spacing: 0.16em; text-transform: uppercase; opacity: 0.9; }}
.banner .hd {{ font-size: 21px; font-weight: 800; letter-spacing: -0.02em; margin: 3px 0 4px; }}
.banner .sm {{ font-size: 9.4px; opacity: 0.96; }}
.alert {{ display: grid; grid-template-columns: 4px 1fr; gap: 11px; padding: 9px 0; border-bottom: 1px solid var(--line); }}
.alert:last-child {{ border-bottom: 0; }}
.alert .stripe {{ border-radius: 3px; }}
.alert .at {{ font-weight: 700; font-size: 10px; }}
.alert .ad {{ font-size: 9px; color: var(--muted); margin-top: 1px; }}
.alert .aref {{ font-size: 8px; color: var(--muted); font-weight: 700; margin-top: 3px; }}

/* ---- timeline table ---- */
table.visits {{ width: 100%; border-collapse: collapse; font-size: 9px; }}
table.visits th {{ text-align: left; font-size: 8.1px; letter-spacing: 0.05em; text-transform: uppercase; color: var(--muted); padding: 6px 8px; border-bottom: 1px solid var(--line); }}
table.visits td {{ padding: 7px 8px; border-bottom: 1px solid var(--line); vertical-align: middle; }}
table.visits tr:last-child td {{ border-bottom: 0; }}
.chip {{ display: inline-block; padding: 1px 7px; border-radius: 99px; font-size: 8px; font-weight: 700; color: #fff; }}
.trend-up {{ color: var(--ok); font-weight: 700; }}
.trend-down {{ color: var(--crit); font-weight: 700; }}

/* ---- footer ---- */
.foot {{ margin-top: 4px; font-size: 7.8px; color: var(--muted); line-height: 1.5; border-top: 1px solid var(--line); padding-top: 8px; }}
.disc {{ background: #fff8e6; border: 1px solid #f4e3b0; border-radius: 10px; padding: 8px 11px; font-size: 8.6px; color: #7a5a12; margin-bottom: 7px; }}
/* single actionable banner directly under the grade — only when action needed */
.actbanner {{ margin-top: 12px; display: grid; grid-template-columns: 10px 1fr; gap: 10px; align-items: start; background: #fdecec; border: 1px solid #f0bcbc; border-radius: 11px; padding: 9px 12px; font-size: 9px; line-height: 1.45; color: #5a1a1a; }}
.actbanner b {{ color: #9b1c1c; }}
.actdot {{ width: 10px; height: 10px; border-radius: 99px; background: #dc2626; margin-top: 3px; }}
.warnbox {{ background: #fef2f2; border: 1px solid #f1c6c6; border-radius: 9px; padding: 8px 10px; font-size: 8.5px; line-height: 1.42; color: #7a1f1f; margin-bottom: 9px; }}
.warnbox b {{ color: #991b1b; }}
.estnote {{ display: inline-block; font-size: 7.8px; font-weight: 700; letter-spacing: 0.04em; text-transform: uppercase; color: var(--muted); background: var(--panel); border: 1px solid var(--line); border-radius: 99px; padding: 2px 9px; margin-bottom: 9px; }}
.sub .eng {{ font-size: 8.4px; color: var(--muted); font-weight: 700; text-transform: uppercase; letter-spacing: 0.05em; }}
.pagebreak {{ page-break-before: always; }}
.avoidbreak {{ page-break-inside: avoid; }}
"""

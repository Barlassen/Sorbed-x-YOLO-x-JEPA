"""Self-contained HTML report.

Rendered from the canonical :class:`WoundAnalysis` (the single source of truth)
with the annotated guide embedded as a data URI, so the file needs no external
assets. No third-party template engine is required for the core report.
"""

from __future__ import annotations

import base64
import html

from sorbed.domain.analysis import WoundAnalysis
from sorbed.domain.enums import Severity, TissueClass

_SEVERITY_COLOR = {
    Severity.INFO: "#2b6cb0",
    Severity.WARNING: "#b7791f",
    Severity.CRITICAL: "#c53030",
}


def build_html(analysis: WoundAnalysis, guide_png: bytes | None = None) -> str:
    """Return a complete standalone HTML document for the analysis."""
    d = analysis.decision
    stage = d.stage.value.replace("_", " ").title()
    conf = "withheld" if d.abstained else f"{d.confidence * 100:.0f}%"

    guide_block = ""
    if guide_png is not None:
        b64 = base64.b64encode(guide_png).decode("ascii")
        guide_block = (
            f'<img class="guide" alt="Annotated wound guide" '
            f'src="data:image/png;base64,{b64}"/>'
        )

    return _TEMPLATE.format(
        title=f"Sorbed report — {stage}",
        stage=html.escape(stage),
        conf=conf,
        review="Yes" if d.requires_clinician_review else "No",
        created=html.escape(analysis.created_at.strftime("%Y-%m-%d %H:%M UTC")),
        analysis_id=html.escape(str(analysis.analysis_id)),
        guide=guide_block,
        narrative=html.escape(d.narrative),
        metrics=_metrics_table(analysis),
        tissue=_tissue_table(analysis),
        evidence=_evidence_table(analysis),
        caveats=_caveats_block(analysis),
        provenance=_provenance_block(analysis),
        disclaimer=html.escape(analysis.disclaimer),
        year=analysis.created_at.year,
    )


def _row(label: str, value: str) -> str:
    return f"<tr><th>{html.escape(label)}</th><td>{html.escape(value)}</td></tr>"


def _metrics_table(a: WoundAnalysis) -> str:
    g = a.metrics.geometry
    rows = []
    if g.area_cm2 is not None:
        rows.append(_row("Area", f"{g.area_cm2:.2f} cm² ({g.area_px:.0f} px)"))
        rows.append(_row("Length × Width", f"{g.length_mm:.0f} × {g.width_mm:.0f} mm"))
    else:
        rows.append(_row("Area", f"{g.area_px:.0f} px (uncalibrated)"))
        rows.append(_row("Length × Width", f"{g.length_px:.0f} × {g.width_px:.0f} px"))
    rows.append(_row("Circularity", f"{g.circularity:.2f}"))
    rows.append(_row("Wound / image", f"{g.wound_fraction_of_image * 100:.1f}%"))
    rows.append(_row("Calibration", a.calibration.status.value))
    rows.append(_row("Skin tone band", a.skin_tone_band.value))
    hs = a.metrics.healing_scores
    if hs and hs.push_partial_total is not None:
        rows.append(_row("PUSH (partial)", f"{hs.push_partial_total} (size {hs.push_size_subscore},"
                                            f" tissue {hs.push_tissue_subscore})"))
    if a.metrics.depth_proxy is not None:
        rows.append(_row("Depth proxy", f"{a.metrics.depth_proxy.relative_depth_index:.2f} "
                                        "(relative shading cue, not a measurement)"))
    return "".join(rows)


def _tissue_table(a: WoundAnalysis) -> str:
    fr = a.metrics.tissue.fractions
    ordered = sorted(fr.items(), key=lambda kv: kv[1], reverse=True)
    rows = [
        _row(cls.value.replace("_", " ").title(), f"{frac * 100:.1f}%")
        for cls, frac in ordered
        if cls is not TissueClass.BACKGROUND and frac > 0
    ]
    return "".join(rows)


def _evidence_table(a: WoundAnalysis) -> str:
    rows = []
    for ev in a.decision.evidence:
        obs = "" if ev.observed is None else str(ev.observed)
        thr = "" if ev.threshold is None else f"{ev.comparison} {ev.threshold}"
        rows.append(
            f"<tr><td>{html.escape(ev.description)}</td>"
            f"<td class=mono>{html.escape(obs)}</td>"
            f"<td class=mono>{html.escape(thr)}</td>"
            f"<td class=mono>{html.escape(ev.source)}</td></tr>"
        )
    if not rows:
        return '<tr><td colspan="4">No supporting evidence (grade withheld).</td></tr>'
    return "".join(rows)


def _caveats_block(a: WoundAnalysis) -> str:
    items = []
    for c in a.decision.caveats:
        color = _SEVERITY_COLOR.get(c.severity, "#555")
        items.append(
            f'<li style="border-left:4px solid {color}">'
            f"<strong>{c.severity.value.upper()}:</strong> {html.escape(c.message)}</li>"
        )
    return "".join(items) or "<li>None</li>"


def _provenance_block(a: WoundAnalysis) -> str:
    p = a.provenance
    return (
        f"segmentation: {html.escape(p.segmentation_backend)} · "
        f"tissue: {html.escape(p.tissue_backend)} · "
        f"staging: {html.escape(p.staging_backend)} · "
        f"config {html.escape(a.config_digest[:12])}"
    )


_TEMPLATE = """<!doctype html>
<html lang="en"><head><meta charset="utf-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1"/>
<title>{title}</title>
<style>
 :root {{ color-scheme: light dark; }}
 body {{ font-family: system-ui, -apple-system, Segoe UI, Roboto, sans-serif;
        margin: 0; background: #f6f7f9; color: #1a1a1a; }}
 @media (prefers-color-scheme: dark) {{ body {{ background:#16171a; color:#e8e8e8; }}
   .card {{ background:#1f2024 !important; }} th {{ color:#aaa !important; }} }}
 header {{ background: linear-gradient(120deg,#7a1f2b,#b4404f); color:#fff; padding:20px 24px; }}
 header h1 {{ margin:0 0 4px; font-size:20px; }}
 header .sub {{ opacity:.9; font-size:13px; }}
 main {{ max-width: 920px; margin: 20px auto; padding: 0 16px; }}
 .card {{ background:#fff; border-radius:12px; padding:16px 20px; margin-bottom:16px;
         box-shadow:0 1px 3px rgba(0,0,0,.08); }}
 .card h2 {{ margin:0 0 10px; font-size:15px; text-transform:uppercase; letter-spacing:.04em;
            opacity:.7; }}
 .guide {{ max-width:100%; border-radius:8px; display:block; }}
 table {{ width:100%; border-collapse:collapse; font-size:14px; }}
 th {{ text-align:left; color:#666; font-weight:600; padding:5px 8px 5px 0; width:38%;
      vertical-align:top; }}
 td {{ padding:5px 0; }} .mono {{ font-family: ui-monospace, monospace; font-size:12.5px; }}
 .banner {{ display:flex; gap:20px; flex-wrap:wrap; }}
 .stat {{ flex:1; min-width:120px; }} .stat .k {{ font-size:12px; opacity:.7; }}
 .stat .v {{ font-size:22px; font-weight:700; }}
 ul.caveats {{ list-style:none; padding:0; margin:0; }}
 ul.caveats li {{ padding:8px 12px; margin:6px 0; background:rgba(127,127,127,.08);
                 border-radius:0 6px 6px 0; font-size:13.5px; }}
 .disclaimer {{ font-size:12.5px; opacity:.75; }}
 footer {{ text-align:center; font-size:12px; opacity:.6; padding:20px; }}
</style></head><body>
<header>
 <h1>Sorbed — Pressure Injury Analysis</h1>
 <div class="sub">Provisional, clinician-reviewed decision support · {created}</div>
</header>
<main>
 <div class="card banner">
  <div class="stat"><div class="k">Provisional grade</div><div class="v">{stage}</div></div>
  <div class="stat"><div class="k">Confidence</div><div class="v">{conf}</div></div>
  <div class="stat"><div class="k">Clinician review</div><div class="v">{review}</div></div>
 </div>
 {guide}
 <div class="card"><h2>Summary</h2><p>{narrative}</p></div>
 <div class="card"><h2>Measurements</h2><table>{metrics}</table></div>
 <div class="card"><h2>Tissue composition</h2><table>{tissue}</table></div>
 <div class="card"><h2>Evidence trace</h2><table>
   <tr><th>Finding</th><th>Observed</th><th>Threshold</th><th>Source</th></tr>
   {evidence}</table></div>
 <div class="card"><h2>Caveats</h2><ul class="caveats">{caveats}</ul></div>
 <div class="card disclaimer"><h2>Disclaimer</h2>{disclaimer}
   <p class="mono">{provenance}</p>
   <p class="mono">analysis {analysis_id}</p></div>
</main>
<footer>Generated by Sorbed · Apache-2.0 · © {year} Ariorad Moniri</footer>
</body></html>"""

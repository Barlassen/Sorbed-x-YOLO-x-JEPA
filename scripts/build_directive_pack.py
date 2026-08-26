"""Build a *clinical directive pack* from a source PDF.

A directive pack is a structured, citable distillation of an institutional
pressure-injury guideline: numbered sections (verbatim text + page reference)
plus the guideline's own figures, keyed to the topics Sorbed reports cite
(staging definitions, tissue colour model, PUSH, size, reassessment cadence,
dressing selection, Braden intervals).

The pack is *derived from the operator's own PDF at build time*. Third-party
directive text and figures are institutional content and are written to a
git-ignored location (``var/directive_packs/<slug>``); they are never committed
to this repository.

Usage::

    python scripts/build_directive_pack.py \
        --pdf /path/to/HD_T86_REV10.pdf --slug hd_t86

This module has no hard dependency on the rest of Sorbed; it only needs
``pymupdf`` (``pip install pymupdf``).
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import fitz  # PyMuPDF

# Section numbers appear on their own line in the source layout, e.g. "3.3." or
# "4.5.11.". Titles for definition sections are the leading phrase up to a colon.
_SECTION_RE = re.compile(r"^(\d+(?:\.\d+){0,3})\.?$")

# Curated topic -> (section ids, figure spec) mapping. Figure specs reference an
# embedded image by (page, xref) discovered from the source PDF; captions are
# our own neutral descriptions of the guideline's illustration.
def _t(sections: list[str], figure=None, caption: str | None = None) -> dict[str, Any]:
    return {"sections": sections, "figure": figure, "caption": caption}


_TOPIC_MAP: dict[str, dict[str, Any]] = {
    # Stage figures are the guideline's own SCHEMATIC cross-section diagrams
    # (not its realistic clinical photos) so a report never mixes a reference
    # illustration up with the patient's uploaded photograph.
    "definition": _t(["3.1"]),
    "stage.normal": _t(["3.2"]),
    "stage.stage_1": _t(["3.3"], (3, 30), "Evre 1 şeması · Stage 1 schematic"),
    "stage.stage_2": _t(["3.4"], (3, 27), "Evre 2 şeması · Stage 2 schematic"),
    "stage.stage_3": _t(["3.5"], (4, 38), "Evre 3 şeması · Stage 3 schematic"),
    "stage.stage_4": _t(["3.6"], (4, 40), "Evre 4 şeması · Stage 4 schematic"),
    "stage.unstageable": _t(["3.7"], (4, 42), "Sınıflandırılamayan şeması · Unstageable schematic"),
    "stage.deep_tissue_injury": _t(["3.8"], (4, 37), "Derin doku hasarı şeması · DTI schematic"),
    "device_related": _t(["3.9"], None, None),
    "support_surfaces": _t(["3.10"], None, None),
    "push": _t(["3.11", "4.5.8"]),
    "exudate": _t(["3.12", "4.5.10"]),
    "debridement": _t(["3.13"]),
    "risk_regions": _t(["4.2"], (6, 57), "Basınç yaralanması risk bölgeleri"),
    "braden": _t(["4.3.2"]),
    "assessment": _t(["4.5.1"]),
    "tissue.ryb": _t(["4.5.9"]),
    "size": _t(["4.5.11"]),
    "care.stage_1": _t(["4.6.2"]),
    "care.stage_2": _t(["4.6.3"]),
    "care.stage_3": _t(["4.6.3"]),
    "care.stage_4": _t(["4.6.4"]),
    "care.unstageable": _t(["4.6.4"]),
    "products": _t(["4.6.11"]),
}


def _parse_sections(doc: fitz.Document) -> dict[str, dict[str, Any]]:
    """Extract ``{section_id: {"id", "text", "page"}}`` from the PDF layout."""
    sections: dict[str, dict[str, Any]] = {}
    current: str | None = None
    buffer: list[str] = []
    start_page = 1

    def flush() -> None:
        if current and buffer:
            text = " ".join(w.strip() for w in buffer if w.strip())
            text = re.sub(r"\s+", " ", text).strip()
            if text:
                sections[current] = {"id": current, "text": text, "page": start_page}

    for pno, page in enumerate(doc, start=1):
        for raw in page.get_text("text").splitlines():
            line = raw.strip()
            m = _SECTION_RE.match(line)
            if m:
                flush()
                current = m.group(1)
                buffer = []
                start_page = pno
            elif current is not None:
                buffer.append(line)
    flush()
    return sections


def _extract_figure(doc: fitz.Document, page_no: int, xref: int, dest: Path) -> bool:
    """Save embedded image ``xref`` on ``page_no`` (1-based) as PNG at ``dest``."""
    try:
        base = doc.extract_image(xref)
    except Exception:
        return False
    from io import BytesIO

    from PIL import Image

    img = Image.open(BytesIO(base["image"])).convert("RGB")
    # Downscale oversized figures; keep them crisp for print embedding.
    img.thumbnail((900, 900), Image.LANCZOS)
    img.save(dest, format="PNG")
    return True


def build_pack(pdf_path: Path, out_dir: Path, slug: str) -> dict[str, Any]:
    doc = fitz.open(pdf_path)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "figures").mkdir(exist_ok=True)

    sections = _parse_sections(doc)

    meta = {
        "slug": slug,
        "doc_name": sections.get("3.1", {}).get("text", "")[:0] or _first_title(doc),
        "doc_no": "HD_T86",
        "source_pdf": pdf_path.name,
        "page_count": doc.page_count,
    }
    # Pull identity fields from the cover page text when present.
    cover = doc[0].get_text("text")
    meta.update(_cover_fields(cover))

    topics: dict[str, Any] = {}
    for key, spec in _TOPIC_MAP.items():
        ids = [s for s in spec["sections"] if s in sections]
        fig_rel: str | None = None
        if spec.get("figure"):
            pno, xref = spec["figure"]
            dest = out_dir / "figures" / f"{key.replace('.', '_')}.png"
            if _extract_figure(doc, pno, xref, dest):
                fig_rel = f"figures/{dest.name}"
        topics[key] = {
            "sections": ids,
            "figure": fig_rel,
            "caption": spec.get("caption"),
        }

    (out_dir / "sections.json").write_text(
        json.dumps(sections, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "topics.json").write_text(
        json.dumps(topics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "meta.json").write_text(
        json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return {"sections": len(sections), "topics": len(topics),
            "figures": sum(1 for t in topics.values() if t["figure"])}


def _first_title(doc: fitz.Document) -> str:
    lines = [x.strip() for x in doc[0].get_text("text").splitlines() if x.strip()]
    return lines[1] if len(lines) > 1 else "Clinical directive"


def _cover_fields(cover: str) -> dict[str, str]:
    fields: dict[str, str] = {}
    lines = [x.strip() for x in cover.splitlines()]
    # The cover carries colon-terminated labels ("Revizyon No:"); the revision
    # history table below repeats the bare words ("Revizyon No"). Match only the
    # labelled form and keep the first hit so the table never overwrites it.
    labels = {
        "Dokümanın Adı:": "doc_name",
        "Doküman No:": "doc_no",
        "Revizyon Tarihi:": "revision_date",
        "Revizyon No:": "revision_no",
        "İlk Yayın Tarihi:": "first_issue",
    }
    for i, line in enumerate(lines):
        key = labels.get(line)
        nxt = lines[i + 1] if i + 1 < len(lines) else ""
        if key and nxt and key not in fields:
            fields[key] = nxt
    return fields


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--pdf", required=True, type=Path)
    ap.add_argument("--slug", default="hd_t86")
    ap.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Pack output directory (default: var/directive_packs/<slug>)",
    )
    args = ap.parse_args()
    out = args.out or Path("var/directive_packs") / args.slug
    stats = build_pack(args.pdf, out, args.slug)
    print(f"directive pack '{args.slug}' -> {out}")
    print(f"  sections={stats['sections']} topics={stats['topics']} figures={stats['figures']}")


if __name__ == "__main__":
    main()

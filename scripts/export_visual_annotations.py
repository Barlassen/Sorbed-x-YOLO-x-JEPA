"""Export visually authored polygon drafts; performs no automatic diagnosis.

Input coordinates refer to a documented inspection viewport. Outputs use native
source-pixel coordinates, with binary lesion/wound-surface and indexed tissue PNGs.
All labels remain AI drafts and are excluded from ground-truth use.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

CLASSES = {
    "background": 0, "granulation": 1, "slough": 2, "eschar": 3,
    "epithelial": 4, "adipose": 5, "muscle": 6, "tendon_bone": 7,
    "intact_skin": 8, "red_surface_unspecified": 9, "unknown": 255,
}
COLORS = {
    "granulation": "#ef4444", "slough": "#facc15", "eschar": "#8b5cf6",
    "epithelial": "#f9a8d4", "adipose": "#fb923c", "muscle": "#b91c1c",
    "tendon_bone": "#cbd5e1", "intact_skin": "#38bdf8",
    "red_surface_unspecified": "#fb7185", "unknown": "#94a3b8",
}


def polygon_mask(points: list, width: int, height: int) -> np.ndarray:
    arr = np.asarray(points, dtype=np.int32)
    if len(arr) < 3 or arr.shape[1:] != (2,):
        raise ValueError("A polygon needs at least three x,y pairs")
    if np.any(arr < 0) or np.any(arr[:, 0] >= width) or np.any(arr[:, 1] >= height):
        raise ValueError("Out-of-bounds polygon")
    if cv2.contourArea(arr) <= 0:
        raise ValueError("Zero-area polygon")
    out = np.zeros((height, width), dtype=np.uint8)
    cv2.fillPoly(out, [arr], 255)
    return out


def native_polygon(points: list, viewport: list, width: int, height: int) -> list:
    vw, vh = viewport
    if abs(vw / vh - width / height) > 0.001:
        raise ValueError("Viewport and native image aspect ratios do not match")
    return [[round(x * width / vw), round(y * height / vh)] for x, y in points]


def svg_polygon(points: list, color: str, layer: str, title: str) -> str:
    pts = " ".join(f"{x},{y}" for x, y in points)
    return (f'<polygon class="{layer}" points="{pts}" fill="{color}" '
            'fill-opacity="0.18" stroke="' + color + '" stroke-width="2" '
            f'vector-effect="non-scaling-stroke"><title>{html.escape(title)}</title></polygon>')


def export(root: Path) -> dict:
    rows = json.loads((root / "inventory.json").read_text())
    inventory = {r["image_id"]: r for r in rows}
    drafts = json.loads((root / "pilot/visual_drafts.json").read_text())
    if len({d["image_id"] for d in drafts}) != len(drafts):
        raise ValueError("Duplicate draft image IDs")
    out = root / "pilot"
    for name in ("records", "lesion_masks", "wound_surface_masks", "tissue_masks", "labelme"):
        (out / name).mkdir(exist_ok=True)
    cards, records = [], []
    for draft in drafts:
        source = inventory[draft["image_id"]]
        path = Path(source["source_path"])
        if hashlib.sha256(path.read_bytes()).hexdigest() != source["sha256"]:
            raise ValueError(f"Source changed: {path.name}")
        with Image.open(path) as im:
            width, height = im.size
            if (width, height) != (source["width"], source["height"]):
                raise ValueError("Image dimension mismatch")
            if im.getexif().get(274, 1) != 1:
                raise ValueError("Nontrivial EXIF orientation: normalize coordinate protocol first")
        native = lambda p: native_polygon(p, draft["view_size"], width, height)
        lesions = [native(p) for p in draft["lesions"]]
        beds = [native(p) for p in draft["beds"]]
        lesion_mask = np.zeros((height, width), dtype=np.uint8)
        bed_mask = np.zeros_like(lesion_mask)
        for points in lesions:
            lesion_mask |= polygon_mask(points, width, height)
        for points in beds:
            bed_mask |= polygon_mask(points, width, height)
        if np.any((bed_mask > 0) & (lesion_mask == 0)):
            raise ValueError(f"Wound surface outside lesion: {path.name}")
        tissue = np.zeros_like(lesion_mask)
        tissue[lesion_mask > 0] = 255
        regions, shapes, overlay = [], [], []
        for layer, polygons, color in (("lesion", lesions, "#22d3ee"),
                                        ("wound_surface", beds, "#4ade80")):
            for idx, points in enumerate(polygons):
                overlay.append(svg_polygon(points, color, layer, f"{layer} {idx + 1}"))
                shapes.append({"label": layer, "points": points, "group_id": None,
                               "shape_type": "polygon", "flags": {"ai_draft": True}})
        clipped_total = 0
        for raw in draft["tissues"]:
            region = dict(raw)
            if "same_as_lesion" in region:
                points = lesions[region.pop("same_as_lesion")]
            elif "same_as_bed" in region:
                points = beds[region.pop("same_as_bed")]
            else:
                points = native(region.pop("polygon"))
            label = region["class_name"]
            mask = polygon_mask(points, width, height)
            clipped = int(np.count_nonzero((mask > 0) & (lesion_mask == 0)))
            clipped_total += clipped
            # Preserve authored vectors; record any raster clipping explicitly.
            tissue[(mask > 0) & (lesion_mask > 0)] = CLASSES[label]
            region.update(polygon_native=points, class_id=CLASSES[label],
                          raster_clipped_pixels=clipped)
            regions.append(region)
            overlay.append(svg_polygon(points, COLORS[label], "tissue", label))
            shapes.append({"label": f"tissue:{label}", "points": points, "group_id": None,
                           "shape_type": "polygon", "flags": {"ai_draft": True}})
        outputs = {}
        for name, mask in (("lesion_masks", lesion_mask), ("wound_surface_masks", bed_mask),
                           ("tissue_masks", tissue)):
            dest = out / name / f"{path.stem}.png"
            Image.fromarray(mask).save(dest)
            with Image.open(dest) as check:
                if check.mode != "L" or check.size != (width, height):
                    raise ValueError("Invalid mask export")
                if not np.array_equal(np.asarray(check), mask):
                    raise ValueError("PNG round-trip altered labels")
            outputs[name] = str(dest.relative_to(root))
        lesion_area = int(np.count_nonzero(lesion_mask))
        counts = {k: int(np.count_nonzero(tissue == v)) for k, v in CLASSES.items() if v != 0}
        record = {
            "schema_version": "1.0", "image_id": source["image_id"],
            "case_id": source["case_id"], "source_path": str(path),
            "source_sha256": source["sha256"], "image_width": width, "image_height": height,
            "coordinate_system": "native_pixels_top_left_xy_exif_1",
            "inspection_viewport": draft["view_size"],
            "provenance": {"author": "Codex", "method": "assistant_visual_polygon_draft",
                           "date": "2026-09-08", "automated_model_used": False,
                           "precision": "coarse_visual_polygon_requires_boundary_review"},
            "review_status": "ai_draft_pending_clinician_review", "reviewer": None,
            "eligible_for_training": False, "eligible_for_ground_truth": False,
            "split": "unassigned", "pressure_etiology_confirmed": False,
            "lesion_polygons": lesions, "wound_surface_polygons": beds,
            "open_bed_status": draft["open_bed_status"], "tissue_regions": regions,
            "stage": {"candidate": draft["stage_candidate"],
                      "alternatives": draft["stage_alternatives"],
                      "certainty": draft["stage_certainty"],
                      "rationale_tr": draft["stage_rationale"], "confirmed_label": None,
                      "prior_stage": None, "pressure_etiology_required": True,
                      "depth_mm": None, "undermining": None, "blanching": None},
            "quality_flags": draft["quality_flags"], "notes_tr": draft["notes"],
            "mask_paths": outputs,
            "geometry_summary": {"lesion_pixels": lesion_area,
                                 "wound_surface_pixels": int(np.count_nonzero(bed_mask)),
                                 "tissue_label_pixels": counts,
                                 "unknown_fraction_of_lesion": counts["unknown"] / lesion_area,
                                 "raster_clipped_pixels": clipped_total},
        }
        (out / "records" / f"{path.stem}.json").write_text(
            json.dumps(record, ensure_ascii=False, indent=2))
        labelme = {"version": "5.5.0", "flags": {"ai_draft": True}, "shapes": shapes,
                   "imagePath": str(path), "imageData": None,
                   "imageHeight": height, "imageWidth": width}
        (out / "labelme" / f"{path.stem}.json").write_text(json.dumps(labelme, indent=2))
        records.append(record)
        source["status"] = "ai_draft_pending_clinician_review"
        photo = base64.b64encode(path.read_bytes()).decode("ascii")
        flags = ", ".join(draft["quality_flags"])
        tissue_notes = "".join(f'<li><b>{html.escape(t["class_name"])}</b>: '
                               f'{html.escape(t["note"])}</li>' for t in regions)
        cards.append(f'''<article id="{path.stem}">
<h2>{html.escape(path.stem)}</h2>
<div class="badge">AI TASLAĞI · Mentor incelemesi bekliyor</div>
<div class="controls"><label><input type="checkbox" checked data-layer="lesion">Lezyon</label>
<label><input type="checkbox" checked data-layer="wound_surface">Yara yüzeyi</label>
<label><input type="checkbox" data-layer="tissue">Doku bölgeleri</label></div>
<svg viewBox="0 0 {width} {height}" xmlns="http://www.w3.org/2000/svg">
<image href="data:image/jpeg;base64,{photo}" width="{width}" height="{height}"/>
{"".join(overlay)}</svg>
<p><b>Evre adayı:</b> {html.escape(draft['stage_candidate'])} · Görsel kanaat: {draft['stage_certainty']}</p>
<p>{html.escape(draft['stage_rationale'])}</p>
<p><b>Alternatifler:</b> {html.escape(', '.join(draft['stage_alternatives']))}</p>
<p>{html.escape(draft['notes'])}</p><ul>{tissue_notes}</ul>
<p class="muted">{html.escape(flags)}</p>
<p><a href="records/{path.stem}.json">Anotasyon JSON</a> ·
<a href="labelme/{path.stem}.json">LabelMe poligonları</a> ·
<a href="lesion_masks/{path.stem}.png">Lezyon maskesi</a> ·
<a href="wound_surface_masks/{path.stem}.png">Yara yüzeyi maskesi</a> ·
<a href="tissue_masks/{path.stem}.png">Doku sınıf maskesi</a></p></article>''')
    duplicates = defaultdict(list)
    for row in rows:
        duplicates[row["sha256"]].append(row["image_id"])
    summary = {"total_images": len(rows), "case_count": len({r["case_id"] for r in rows}),
               "unique_file_hashes": len(duplicates),
               "exact_duplicate_extra_copies": len(rows) - len(duplicates),
               "exact_duplicate_groups": sum(len(v) > 1 for v in duplicates.values()),
               "visually_annotated_images": len(records), "pending_images": len(rows) - len(records),
               "confirmed_annotations": 0,
               "stage_candidates": dict(Counter(r["stage"]["candidate"] for r in records))}
    (root / "inventory.json").write_text(json.dumps(rows, indent=2))
    (root / "duplicate_groups.json").write_text(json.dumps(
        {k: v for k, v in duplicates.items() if len(v) > 1}, indent=2))
    (root / "summary.json").write_text(json.dumps(summary, indent=2))
    (root / "tissue_classes.json").write_text(json.dumps(CLASSES, indent=2))
    nav = " ".join(f'<a href="#{r["image_id"]}">{r["case_id"]}</a>' for r in records)
    legend = " ".join(f'<span style="border-bottom:3px solid {COLORS[k]}">{k}</span>'
                      for k in sorted({t["class_name"] for r in records for t in r["tissue_regions"]}))
    page = '''<!doctype html><html lang="tr"><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sorbed · Anotasyon pilotu</title><style>
body{font:16px/1.6 system-ui,sans-serif;background:#0f172a;color:#e2e8f0;margin:0}
main{max-width:1050px;margin:auto;padding:30px}h1{font-size:30px}h2{font-size:21px}
article{padding:24px;background:#1e293b;border:1px solid #334155;border-radius:16px;margin:28px 0}
svg{display:block;width:100%;max-height:760px;background:#0b1220;margin-top:18px}
a{color:#67e8f9}nav a{display:inline-block;margin:5px 12px 5px 0}
.badge{color:#fde68a}.muted{font-size:13px;color:#cbd5e1}.controls{display:flex;gap:24px;margin-top:12px}
.tissue{display:none}input{accent-color:#22d3ee}.legend span{margin-right:16px}
</style><main><h1>Yara sınırı · Doku · Evre</h1>
<p>8 görüntü / 8 vaka için görsel poligon taslakları. 732 görüntünün tamamı etiketlenmedi.
Lezyon maskesi renk değişikliğini de kapsayabilir; yara yüzeyi maskesi açık veya örtüyle kaplı
yara alanını gösterir. Örtü altındaki görünmeyen sınır çizilmedi.</p>
<p><b>Bu etiketler bağımsız ground truth değildir.</b> Evreler klinik doğrulama bekleyen
adaylardır. Belirsiz doku 255 olarak saklanır; kırmızı yüzey, granülasyonla eşanlamlı değildir.
Kutuları kapatarak değişmemiş orijinal fotoğrafı inceleyebilirsiniz.</p>
<p><a href="https://cdn.ymaws.com/npiap.com/resource/resmgr/online_store/npiap_pressure_injury_stages.pdf">NPIAP evre tanımları</a> ·
<a href="../PROTOCOL.md">Anotasyon protokolü</a></p>'''
    page += f'<nav>{nav}</nav><div class="legend">{legend}</div>' + "".join(cards)
    page += '''</main><script>
document.querySelectorAll('input[data-layer]').forEach(input=>{
input.addEventListener('change',()=>input.closest('article').querySelectorAll('.'+input.dataset.layer)
.forEach(el=>el.style.display=input.checked?'':'none'));
if(input.dataset.layer==='tissue')input.addEventListener('change',()=>input.closest('article')
.querySelectorAll('.tissue').forEach(el=>el.style.display=input.checked?'block':'none'));
});</script></html>'''
    (out / "review.html").write_text(page)
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("data/annotations/pressure_injury_v1"))
    print(json.dumps(export(parser.parse_args().root), indent=2))

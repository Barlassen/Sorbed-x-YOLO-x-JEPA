"""Versioned visual surface annotations. Run: python -m scripts.export_surface_annotations."""
from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
from pathlib import Path

import numpy as np
from PIL import Image

from scripts.export_visual_annotations import native_polygon, polygon_mask, svg_polygon
from training.surface_labels import SURFACE_CLASSES, SURFACE_COLORS, build_target


def export(parent: Path, root: Path, render: bool = False) -> dict:
    drafts = json.loads((root / 'pilot/surface_drafts.json').read_text())
    if len({d['image_id'] for d in drafts}) != len(drafts):
        raise ValueError('Duplicate image ID')
    for directory in ('records', 'surface_masks', 'validity_masks', 'labelme'):
        (root / 'pilot' / directory).mkdir(parents=True, exist_ok=True)
    records, cards = [], []
    for draft in drafts:
        image_id = draft['image_id']
        parent_path = parent / 'pilot/records' / f'{image_id}.json'
        old = json.loads(parent_path.read_text())
        source = Path(old['source_path'])
        if hashlib.sha256(source.read_bytes()).hexdigest() != old['source_sha256']:
            raise ValueError('Source hash mismatch')
        w, h = old['image_width'], old['image_height']
        with Image.open(source) as image:
            if image.size != (w, h) or image.getexif().get(274, 1) != 1:
                raise ValueError('Unexpected image geometry/orientation')
        with Image.open(parent / old['mask_paths']['lesion_masks']) as image:
            lesion = np.asarray(image) > 0
        region_masks, regions, shapes = [], [], []
        for raw in draft['regions']:
            region = dict(raw)
            keys = {'lesion_index', 'bed_index', 'tissue_region_index', 'polygon'} & region.keys()
            if len(keys) != 1:
                raise ValueError('Each surface region needs exactly one geometry source')
            key = next(iter(keys))
            value = region.pop(key)
            if key == 'lesion_index':
                points = old['lesion_polygons'][value]
            elif key == 'bed_index':
                points = old['wound_surface_polygons'][value]
            elif key == 'tissue_region_index':
                points = old['tissue_regions'][value]['polygon_native']
            else:
                points = native_polygon(value, old['inspection_viewport'], w, h)
            label = region['class_name']
            class_id = SURFACE_CLASSES[label]
            region_masks.append((class_id, polygon_mask(points, w, h) > 0))
            region.update(class_id=class_id, polygon_native=points,
                          geometry_source={key: value},
                          assignment_method='explicit_assistant_visual_reinspection')
            regions.append(region)
            shapes.append({'label': f'surface:{label}', 'points': points, 'group_id': None,
                           'shape_type': 'polygon', 'flags': {'ai_draft': True}})
        target = build_target(lesion, region_masks)
        mask_path = root / 'pilot/surface_masks' / f'{image_id}.png'
        validity_path = root / 'pilot/validity_masks' / f'{image_id}.png'
        Image.fromarray(target).save(mask_path)
        Image.fromarray((target != 255).astype(np.uint8) * 255).save(validity_path)
        with Image.open(mask_path) as image:
            if not np.array_equal(np.asarray(image), target) or image.size != (w, h):
                raise ValueError('Mask round-trip failure')
        counts = {k: int((target == v).sum()) for k, v in SURFACE_CLASSES.items()}
        record = {
            'schema_version': 'surface-2.0', 'image_id': image_id, 'case_id': old['case_id'],
            'source_path': str(source), 'source_sha256': old['source_sha256'],
            'image_width': w, 'image_height': h,
            'parent_annotation_path': str(parent_path.resolve()),
            'parent_annotation_sha256': hashlib.sha256(parent_path.read_bytes()).hexdigest(),
            'surface_regions': regions, 'surface_class_ids': SURFACE_CLASSES,
            'surface_mask_path': str(mask_path.resolve()),
            'surface_mask_sha256': hashlib.sha256(mask_path.read_bytes()).hexdigest(),
            'validity_mask_path': str(validity_path.resolve()),
            'review_status': 'ai_draft_pending_clinician_review', 'reviewer': None,
            'eligible_for_training': False, 'eligible_for_ground_truth': False,
            'patient_group_id': None, 'split': 'unassigned',
            'stage': old['stage'], 'stage_inferred_from_surface': False,
            'tissue_annotation_path': str(parent_path.resolve()),
            'notes_tr': draft['note_tr'],
            'unknown_policy': 'unassigned lesion pixels remain 255; never inferred intact',
            'background_policy': 'outside draft lesion is provisional background, not confirmed',
            'annotation_scope': 'partial visual labels; unknown pixels excluded from loss and Dice',
            'provenance': {'author': 'Codex', 'date': '2026-09-08',
                           'method': 'assistant_visual_polygon_draft',
                           'precision': 'coarse_visual_polygon_requires_boundary_review'},
            'geometry_summary': {'class_pixels': counts, 'lesion_pixels': int(lesion.sum()),
                                 'known_fraction_of_lesion': float(((target != 255) & lesion).sum()
                                                                  / lesion.sum())},
        }
        (root / 'pilot/records' / f'{image_id}.json').write_text(
            json.dumps(record, ensure_ascii=False, indent=2))
        (root / 'pilot/labelme' / f'{image_id}.json').write_text(json.dumps({
            'version': '5.5.0', 'flags': {'ai_draft': True}, 'shapes': shapes,
            'imagePath': str(source), 'imageData': None, 'imageWidth': w, 'imageHeight': h,
        }, indent=2))
        records.append(record)
        encoded = base64.b64encode(source.read_bytes()).decode('ascii')
        overlay = ''.join(svg_polygon(r['polygon_native'], SURFACE_COLORS[r['class_name']],
                                      'surface', r['class_name']) for r in regions)
        reasons = ''.join(f"<li><b>{r['class_name']}</b> ({r['certainty']}): "
                          f"{html.escape(r['reason_tr'])}</li>" for r in regions)
        cards.append(f'''<article><h2>{image_id}</h2><p>AI taslağı · Uzman onayı yok</p>
<svg viewBox="0 0 {w} {h}"><image href="data:image/jpeg;base64,{encoded}" width="{w}" height="{h}"/>
{overlay}</svg><p>{html.escape(draft['note_tr'])}</p><ul>{reasons}</ul>
<p>Lezyon içinde sınıflandırılmış alan: %{record['geometry_summary']['known_fraction_of_lesion']*100:.1f}.
Diğer lezyon pikselleri belirsiz (255); kapalı deri sayılmadı.</p>
<p>Evre adayı: {old['stage']['candidate']} — yüzeyden otomatik evre üretilmedi.</p>
<p><a href="records/{image_id}.json">JSON</a> · <a href="surface_masks/{image_id}.png">Sınıf maskesi</a>
· <a href="labelme/{image_id}.json">LabelMe</a></p></article>''')
    (root / 'surface_classes.json').write_text(json.dumps(SURFACE_CLASSES, indent=2))
    summary = {'schema_version': 'surface-2.0', 'images': len(records),
               'confirmed_annotations': 0, 'training_started': False,
               'images_containing_class': {name: sum(r['geometry_summary']['class_pixels'][name] > 0
                                                    for r in records)
                                           for name in SURFACE_COLORS},
               'per_image': {r['image_id']: r['geometry_summary'] for r in records}}
    (root / 'summary.json').write_text(json.dumps(summary, indent=2))
    page = '''<!doctype html><html lang="tr"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Sorbed yüzey anotasyonu v2</title><style>body{font:16px/1.6 system-ui;background:#0f172a;color:#e2e8f0;margin:0}
main{max-width:1050px;margin:auto;padding:25px}article{background:#1e293b;padding:24px;margin:24px 0;border-radius:12px}
svg{width:100%;max-height:720px;background:#0b1220}a{color:#67e8f9}h1{font-size:28px}</style>
<main><h1>Kapalı deri · Açık yüzey · Örtülü yüzey · Belirsiz</h1>
<p>8 pilot görüntü, uzman doğrulaması bekleyen taslaklar. Mavi: bütünlüğü korunmuş lezyonlu deri;
pembe: açık yüzey; sarı: slough/eskarla örtülü yüzey. Poligon verilmeyen lezyon alanı belirsizdir.
Sınıf maskesi 0/1/2/3/255 değerlerini taşır. Belirsiz alan kapalı deri veya negatif örnek değildir.</p>
<p><a href="../PROTOCOL.md">Protokol</a> · <a href="surface_qa.png">Belirsiz bölgeleri de gösteren kontrol figürü</a></p>'''
    (root / 'pilot/review.html').write_text(page + ''.join(cards) + '</main></html>')
    if render:
        render_qa(records, root / 'pilot/surface_qa.png')
    return summary


def render_qa(records: list, out: Path) -> None:
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba

    fig, axes = plt.subplots(4, 2, figsize=(14, 22), layout='constrained')
    for ax, r in zip(axes.flat, records, strict=True):
        with Image.open(r['source_path']) as im:
            im.thumbnail((900, 900))
            size = im.size
            ax.imshow(im)
        with Image.open(r['surface_mask_path']) as im:
            mask = np.asarray(im.resize(size, Image.Resampling.NEAREST))
        colors = np.zeros((*mask.shape, 4), dtype=np.float32)
        for name, color in SURFACE_COLORS.items():
            colors[mask == SURFACE_CLASSES[name]] = to_rgba(color, .38)
        ax.imshow(colors)
        ax.set_title(r['image_id'] + ' | AI DRAFT', fontsize=10)
        ax.axis('off')
    fig.suptitle('Surface annotation v2 — not expert ground truth\nBlue: intact lesion | Pink: open surface | Yellow: covered surface | Gray: unknown lesion pixels', fontsize=12)
    fig.savefig(out, dpi=140)
    plt.close(fig)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--parent', type=Path, default=Path('data/annotations/pressure_injury_v1'))
    parser.add_argument('--root', type=Path, default=Path('data/annotations/pressure_injury_v2'))
    parser.add_argument('--render', action='store_true')
    args = parser.parse_args()
    result = export(args.parent, args.root, args.render)
    print(json.dumps({k: v for k, v in result.items() if k != 'per_image'}, indent=2))

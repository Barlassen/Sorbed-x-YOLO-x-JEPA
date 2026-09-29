"""Evaluate class-ID surface predictions; does not relabel binary YOLO masks."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from PIL import Image

from training.surface_labels import SURFACE_CLASSES, aggregate_counts, require_review, surface_counts


def evaluate(records_dir: Path, predictions: Path, schema: Path, split: str = 'test',
             allow_draft_reference: bool = False) -> dict:
    if json.loads(schema.read_text()) != SURFACE_CLASSES:
        raise ValueError('Prediction schema must exactly match surface_classes.json')
    paths = sorted(records_dir.glob('*.json'))
    if not paths:
        raise ValueError('No annotation records')
    results = []
    for path in paths:
        record = json.loads(path.read_text())
        if split != 'all' and record['split'] != split:
            continue
        if record.get('schema_version') != 'surface-2.0':
            raise ValueError('Expected surface-2.0 annotation')
        if not allow_draft_reference:
            require_review(record, 'evaluation')
        image_id = record['image_id']
        target = Path(record['surface_mask_path'])
        if hashlib.sha256(target.read_bytes()).hexdigest() != record['surface_mask_sha256']:
            raise ValueError('Annotation mask hash mismatch')
        with Image.open(target) as im:
            reference = np.array(im)
        with Image.open(predictions / f'{image_id}.png') as im:
            pred = np.array(im)
        result = surface_counts(pred, reference)
        result.update(image_id=image_id, case_id=record['case_id'], split=record['split'])
        results.append(result)
    if not results:
        raise ValueError(f'No annotations in split {split!r}; assign reviewed patient groups first')
    return {
        'reference_kind': 'unreviewed_draft_agreement' if allow_draft_reference else 'reviewed_reference',
        'independent_test_claim': False,
        'note': 'Dataset selection and pretraining exposure require a separate audit. '
                'Draft agreement is not accuracy. Unknown-reference pixels are excluded; '
                'prediction abstention on known pixels is penalized. Both-empty is null.',
        'images': len(results), 'split': split,
        'classes': aggregate_counts(results), 'per_image': results,
    }


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--predictions', type=Path, required=True)
    parser.add_argument('--prediction-schema', type=Path, required=True)
    parser.add_argument('--split', choices=['train', 'val', 'test', 'all'], default='test')
    parser.add_argument('--allow-draft-reference', action='store_true')
    parser.add_argument('--out', type=Path, required=True)
    args = parser.parse_args()
    try:
        report = evaluate(args.records, args.predictions, args.prediction_schema,
                          args.split, args.allow_draft_reference)
    except (ValueError, FileNotFoundError) as error:
        parser.error(str(error))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(report, indent=2, allow_nan=False))
    print(json.dumps(report['classes'], indent=2))

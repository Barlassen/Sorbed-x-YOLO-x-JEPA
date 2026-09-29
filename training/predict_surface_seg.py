"""Export native-resolution class-ID masks from SurfaceNet; no stage inference."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.nn import functional as F

from training.surface_labels import SURFACE_CLASSES
from training.train_surface_seg import SurfaceNet


@torch.no_grad()
def predict(weights: Path, records_dir: Path, out: Path, split: str = 'test',
            device: str = 'cpu', confidence_threshold: float = 0.0) -> int:
    if not 0 <= confidence_threshold <= 1:
        raise ValueError('Confidence threshold must be in [0,1]')
    checkpoint = torch.load(weights, map_location='cpu', weights_only=True)
    if checkpoint.get('architecture') != 'SurfaceNet-v1' or checkpoint.get('surface_classes') != SURFACE_CLASSES:
        raise ValueError('Checkpoint architecture/class schema mismatch')
    model = SurfaceNet().to(device).eval()
    model.load_state_dict(checkpoint['state_dict'], strict=True)
    size = checkpoint['size']
    records = [json.loads(p.read_text()) for p in sorted(records_dir.glob('*.json'))]
    records = [r for r in records if split == 'all' or r['split'] == split]
    if not records:
        raise ValueError(f'No records in split {split!r}')
    if out.exists():
        raise ValueError('Prediction directory exists; use a new output directory')
    out.mkdir(parents=True)
    sources = []
    for r in records:
        source = Path(r['source_path'])
        if hashlib.sha256(source.read_bytes()).hexdigest() != r['source_sha256']:
            raise ValueError('Source hash mismatch')
        with Image.open(source) as im:
            if im.getexif().get(274, 1) != 1:
                raise ValueError('Unsupported EXIF orientation')
            w, h = im.size
            rgb = np.array(im.convert('RGB').resize((size, size), Image.Resampling.BILINEAR))
        x = torch.from_numpy(rgb).permute(2, 0, 1).float().unsqueeze(0).to(device) / 255
        logits = model(x)
        # Interpolate logits, not discrete class IDs. Return to native geometry.
        logits = F.interpolate(logits, size=(h, w), mode='bilinear', align_corners=False)
        prob, labels = logits.softmax(1).max(1)
        pred = labels[0].cpu().numpy().astype(np.uint8)
        pred[prob[0].cpu().numpy() < confidence_threshold] = 255
        Image.fromarray(pred).save(out / f"{r['image_id']}.png")
        sources.append({'image_id': r['image_id'], 'source_sha256': r['source_sha256']})
    (out / 'surface_classes.json').write_text(json.dumps(SURFACE_CLASSES, indent=2))
    (out / 'prediction_manifest.json').write_text(json.dumps({
        'weights': str(weights.resolve()), 'weights_sha256': hashlib.sha256(weights.read_bytes()).hexdigest(),
        'confidence_threshold': confidence_threshold, 'confidence_calibrated': False,
        'split': split, 'sources': sources, 'clinical_stage_inferred': False,
    }, indent=2))
    return len(sources)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--weights', type=Path, required=True)
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--split', choices=['train', 'val', 'test', 'all'], default='test')
    parser.add_argument('--device', default='cpu', choices=['cpu', 'mps', 'cuda'])
    parser.add_argument('--confidence-threshold', type=float, default=0.0)
    args = parser.parse_args()
    try:
        count = predict(args.weights, args.records, args.out, args.split,
                        args.device, args.confidence_threshold)
    except ValueError as error:
        parser.error(str(error))
    print(f'{count} native-resolution predictions written')

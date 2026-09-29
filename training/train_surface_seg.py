"""A single shared encoder with a four-channel surface segmentation head.

A small from-scratch baseline, NOT a trained replacement for the current YOLO.
Only reviewed training/validation labels are accepted; test records are not loaded.
Unknown=255 is ignored. Surface states never directly assign a clinical stage.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, Dataset

from training.surface_labels import (
    IGNORE_INDEX, NUM_CLASSES, SURFACE_CLASSES, aggregate_counts, require_review,
    surface_counts, validate_target,
)


def select_reviewed_splits(records: list[dict]) -> tuple[list[dict], list[dict]]:
    """Fail closed on ambiguous grouping or duplicate images crossing splits."""
    if not records:
        raise ValueError('No annotation records')
    owners: dict[tuple[str, str], str] = {}
    case_groups = {}
    selected = {'train': [], 'val': []}
    seen = {}
    for r in records:
        split = r.get('split')
        group = r.get('patient_group_id')
        if split not in {'train', 'val', 'test'} or not group:
            raise ValueError('Assign verified patient_group_id and train/val/test splits first')
        if r.get('schema_version') != 'surface-2.0':
            raise ValueError('Expected surface-2.0 annotation')
        case = r['case_id']
        if case in case_groups and case_groups[case] != group:
            raise ValueError('A case has conflicting patient group IDs')
        case_groups[case] = group
        for key in (('patient', group), ('case', case), ('hash', r['source_sha256'])):
            if key in owners and owners[key] != split:
                raise ValueError('Patient/case/duplicate leakage across splits')
            owners[key] = split
        if split == 'test':
            continue  # Never open test images, masks, or stage labels during training.
        require_review(r, 'training' if split == 'train' else 'evaluation')
        key = (split, r['source_sha256'])
        if key in seen:
            if seen[key] != r['surface_mask_sha256']:
                raise ValueError('Identical images have conflicting surface masks')
            continue
        seen[key] = r['surface_mask_sha256']
        selected[split].append(r)
    if not selected['train'] or not selected['val']:
        raise ValueError('At least one reviewed patient group in both train and val is required')
    return selected['train'], selected['val']


class SurfaceDataset(Dataset):
    def __init__(self, records: list[dict], size: int, purpose: str):
        if size < 16:
            raise ValueError('Image size must be at least 16')
        self.items = []
        for r in records:
            require_review(r, purpose)
            source, mask_path = Path(r['source_path']), Path(r['surface_mask_path'])
            for path, expected in ((source, r['source_sha256']),
                                   (mask_path, r['surface_mask_sha256'])):
                if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                    raise ValueError('Dataset hash mismatch')
            with Image.open(source) as im:
                if im.getexif().get(274, 1) != 1:
                    raise ValueError('EXIF orientation must be normalized first')
                native_size = im.size
                rgb = np.array(im.convert('RGB').resize((size, size), Image.Resampling.BILINEAR))
            with Image.open(mask_path) as im:
                if im.size != native_size:
                    raise ValueError('Native mask/image dimensions differ')
                validate_target(np.array(im))
                target = np.array(im.resize((size, size), Image.Resampling.NEAREST), dtype=np.int64)
            if not np.any(target != IGNORE_INDEX):
                raise ValueError('Image has no supervised pixels at training resolution')
            self.items.append((torch.from_numpy(rgb).permute(2, 0, 1).float() / 255,
                               torch.from_numpy(target)))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, index):
        return self.items[index]


def block(in_channels: int, out_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1), nn.GroupNorm(4, out_channels), nn.SiLU(),
        nn.Conv2d(out_channels, out_channels, 3, padding=1), nn.GroupNorm(4, out_channels), nn.SiLU(),
    )


class SurfaceNet(nn.Module):
    """Shared encoder, skip decoder, background/intact/open/covered softmax head."""
    def __init__(self):
        super().__init__()
        self.enc1 = block(3, 16)
        self.enc2 = block(16, 32)
        self.bottleneck = block(32, 64)
        self.dec2 = block(96, 32)
        self.dec1 = block(48, 16)
        self.head = nn.Conv2d(16, NUM_CLASSES, 1)

    def forward(self, x):
        a = self.enc1(x)
        b = self.enc2(F.max_pool2d(a, 2))
        c = self.bottleneck(F.max_pool2d(b, 2))
        c = F.interpolate(c, size=b.shape[-2:], mode='bilinear', align_corners=False)
        d = self.dec2(torch.cat([c, b], dim=1))
        d = F.interpolate(d, size=a.shape[-2:], mode='bilinear', align_corners=False)
        return self.head(self.dec1(torch.cat([d, a], dim=1)))


def surface_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    # Sum/known-count also remains finite when a batch is entirely ignored.
    losses = F.cross_entropy(logits, target, ignore_index=IGNORE_INDEX, reduction='sum')
    return losses / (target != IGNORE_INDEX).sum().clamp_min(1)


def run(args) -> None:
    if args.epochs < 1 or args.batch < 1 or args.size < 16 or args.lr <= 0:
        raise ValueError('Positive epochs, batch, lr and size >=16 required')
    records = [json.loads(p.read_text()) for p in sorted(args.records.glob('*.json'))]
    train, val = select_reviewed_splits(records)
    tr = SurfaceDataset(train, args.size, 'training')
    va = SurfaceDataset(val, args.size, 'evaluation')
    torch.manual_seed(args.seed)
    model = SurfaceNet().to(args.device)
    loader = DataLoader(tr, batch_size=args.batch, shuffle=True,
                        generator=torch.Generator().manual_seed(args.seed), num_workers=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    if args.out.exists():
        raise ValueError('Output directory already exists; choose a new experiment directory')
    args.out.mkdir(parents=True)
    best_loss, history = float('inf'), []
    for epoch in range(args.epochs):
        model.train()
        for x, y in loader:
            optimizer.zero_grad()
            loss = surface_loss(model(x.to(args.device)), y.to(args.device))
            loss.backward()
            optimizer.step()
        model.eval()
        val_loss, known_pixels, measurements = 0.0, 0, []
        with torch.no_grad():
            for x, y in DataLoader(va, batch_size=1):
                logits = model(x.to(args.device))
                yy = y.to(args.device)
                val_loss += F.cross_entropy(logits, yy, ignore_index=255, reduction='sum').item()
                known_pixels += int((y != 255).sum())
                pred = logits.argmax(1)[0].cpu().numpy()
                measurements.append(surface_counts(pred, y[0].numpy()))
        val_loss /= known_pixels
        metrics = aggregate_counts(measurements)
        history.append({'epoch': epoch + 1, 'validation_loss': val_loss,
                        'validation_dice_at_training_resolution': metrics})
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save({'state_dict': model.state_dict(), 'architecture': 'SurfaceNet-v1',
                        'surface_classes': SURFACE_CLASSES, 'size': args.size,
                        'seed': args.seed, 'pretrained': False}, args.out / 'best.pt')
        print(f'epoch={epoch+1} validation_loss={val_loss:.4f}', flush=True)
    (args.out / 'history.json').write_text(json.dumps(history, indent=2))
    (args.out / 'surface_classes.json').write_text(json.dumps(SURFACE_CLASSES, indent=2))
    (args.out / 'experiment.json').write_text(json.dumps({
        'architecture': 'SurfaceNet-v1', 'pretrained': False, 'seed': args.seed,
        'size': args.size, 'epochs': args.epochs, 'batch': args.batch, 'lr': args.lr,
        'device': args.device, 'train': train, 'validation': val,
        'test_evaluated': False, 'runtime_model_replaced': False,
    }, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--size', type=int, default=256)
    parser.add_argument('--batch', type=int, default=4)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--device', default='cpu', choices=['cpu', 'mps', 'cuda'])
    args = parser.parse_args()
    try:
        run(args)
    except ValueError as error:
        parser.error(str(error))

#!/usr/bin/env python3
"""Stage 3: a pressure-injury stage head on the JEPA encoder — and the ablation.

Stages 1-2 fine-tuned YOLO26-seg (finds the wound) and pre-trained a mask-guided
JEPA encoder on unlabeled images (learns wound representations). This stage attaches
an ordinal-ish grading head onto that encoder and trains it on stage-labelled data
(PIID, EPUAP Stage 1-4), then measures whether the JEPA pre-training actually helped.

The controlled experiment (``--init``):

* ``--init jepa``   — start from ``runs_jepa/jepa_encoder.pt`` (our pre-training).
* ``--init random`` — identical architecture, random weights (no pre-training).

Everything else — data split, head, optimizer, epochs — is held equal, so a gap in
validation quadratic-weighted kappa between the two is attributable to the JEPA
pre-training. That gap is the manuscript's core claim, in miniature.

Honest scope: the laptop encoder is tiny and was pre-trained on foot-ulcer images
(a domain gap to pressure injuries), so absolute numbers are a scaffold, not a
result. The *comparison* is what this script exists to make runnable.

    python -m training.train_grade_jepa --data data/corpus/piid_data/dataset --init jepa
    python -m training.train_grade_jepa --data data/corpus/piid_data/dataset --init random
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, cohen_kappa_score
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset

from training.train_jepa import (
    _IMAGENET_MEAN,
    _IMAGENET_STD,
    Encoder,
    JepaConfig,
    _default_device,
)

_NUM_STAGES = 4


class PiidDataset(Dataset):
    """PIID images laid out as ``<root>/<stage>/stage_<n>_*.jpg`` (stage in 1..4)."""

    def __init__(self, paths: list[Path], labels: list[int], img_size: int = 224) -> None:
        self.paths = paths
        self.labels = labels
        self.img_size = img_size

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, i: int):
        bgr = cv2.imread(str(self.paths[i]), cv2.IMREAD_COLOR)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        img = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        img = (img - _IMAGENET_MEAN) / _IMAGENET_STD
        return img, self.labels[i]


def scan_piid(root: Path) -> tuple[list[Path], list[int]]:
    paths, labels = [], []
    exts = {".jpg", ".jpeg", ".png"}
    for stage in (1, 2, 3, 4):
        for p in sorted(q for q in (root / str(stage)).iterdir() if q.suffix.lower() in exts):
            paths.append(p)
            labels.append(stage - 1)  # 0-indexed classes
    if not paths:
        raise FileNotFoundError(f"no PIID images under {root} (expected <root>/1..4/*.jpg)")
    return paths, labels


class Grader(nn.Module):
    """JEPA encoder (mean-pooled) + a linear stage head."""

    def __init__(self, cfg: JepaConfig, *, num_classes: int = _NUM_STAGES,
                 jepa_weights: str | None = None, freeze: bool = False) -> None:
        super().__init__()
        self.encoder = Encoder(cfg)
        if jepa_weights:
            state = torch.load(jepa_weights, map_location="cpu")
            missing, unexpected = self.encoder.load_state_dict(state, strict=False)
            print(f"loaded JEPA encoder ({len(state)} tensors; "
                  f"missing={len(missing)} unexpected={len(unexpected)})")
        if freeze:
            for p in self.encoder.parameters():
                p.requires_grad_(False)
        self.head = nn.Sequential(nn.LayerNorm(cfg.enc_dim), nn.Linear(cfg.enc_dim, num_classes))

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        tokens = self.encoder(images)      # (B, N, D)
        return self.head(tokens.mean(dim=1))


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, device: str) -> tuple[float, float]:
    model.eval()
    y_true, y_pred = [], []
    for images, labels in loader:
        logits = model(images.to(device))
        y_pred.extend(logits.argmax(1).cpu().tolist())
        y_true.extend(labels.tolist())
    acc = accuracy_score(y_true, y_pred)
    # Quadratic-weighted kappa: the standard staging metric (penalises far-off
    # stage errors more). Falls back to 0.0 if a run predicts a single class.
    try:
        qwk = cohen_kappa_score(y_true, y_pred, weights="quadratic", labels=list(range(_NUM_STAGES)))
    except Exception:
        qwk = 0.0
    return acc, float(qwk)


def run(args) -> None:
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    device = args.device

    paths, labels = scan_piid(Path(args.data))
    # Split seed is separate from the training seed: keep the split fixed (so a
    # JEPA encoder pre-trained on this split's train images never leaks into val)
    # while varying only the training/init seed to measure run-to-run variance.
    tr_p, va_p, tr_y, va_y = train_test_split(
        paths, labels, test_size=0.2, stratify=labels, random_state=args.split_seed
    )
    train_loader = DataLoader(PiidDataset(tr_p, tr_y, args.img_size), batch_size=args.batch,
                              shuffle=True, num_workers=0, drop_last=True)
    val_loader = DataLoader(PiidDataset(va_p, va_y, args.img_size), batch_size=args.batch,
                            shuffle=False, num_workers=0)

    cfg = JepaConfig(img_size=args.img_size)
    jepa_weights = args.jepa_weights if args.init == "jepa" else None
    model = Grader(cfg, jepa_weights=jepa_weights, freeze=args.freeze).to(device)
    print(f"init={args.init}  freeze={args.freeze}  device={device}  "
          f"train={len(tr_p)} val={len(va_p)}")

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.05)
    # Class-weighted CE to counter the mild stage imbalance.
    counts = np.bincount(tr_y, minlength=_NUM_STAGES).astype(np.float32)
    weight = torch.tensor((counts.sum() / (counts + 1e-6)), device=device)
    weight = weight / weight.mean()
    criterion = nn.CrossEntropyLoss(weight=weight)

    best = (0.0, 0.0)
    for epoch in range(args.epochs):
        model.train()
        for images, lbl in train_loader:
            images, lbl = images.to(device), lbl.to(device)
            loss = criterion(model(images), lbl)
            opt.zero_grad()
            loss.backward()
            opt.step()
        acc, qwk = evaluate(model, val_loader, device)
        best = max(best, (qwk, acc))
        print(f"epoch {epoch + 1}/{args.epochs}  val_acc={acc:.3f}  val_qwk={qwk:.3f}")

    print(f"\nRESULT[{args.init}]  best val_qwk={best[0]:.3f}  (acc at best={best[1]:.3f})")


def main() -> None:
    ap = argparse.ArgumentParser(description="Stage-3 grading head on the JEPA encoder + ablation.")
    ap.add_argument("--data", type=Path, required=True, help="PIID root with 1/ 2/ 3/ 4/ subfolders.")
    ap.add_argument("--init", choices=["jepa", "random"], default="jepa")
    ap.add_argument("--jepa-weights", default="runs_jepa/jepa_encoder.pt")
    ap.add_argument("--freeze", action="store_true", help="Linear-probe: freeze the encoder.")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0, help="Training/init seed (varies run-to-run).")
    ap.add_argument("--split-seed", type=int, default=0, help="Fixed split seed (keep constant to avoid JEPA val leakage).")
    ap.add_argument("--device", default=_default_device())
    run(ap.parse_args())


if __name__ == "__main__":
    main()

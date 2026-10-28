"""Staging step 2, arm B: a CNN that predicts the pressure-injury stage straight from the photo (no mask, no tissue).

Source-grouped folds (``petchs_groups``): trains on the other two folds, picks the epoch by balanced accuracy on a
source-disjoint internal val (every 6th training source), and writes predictions for every image of the held-out
fold. Backbone: MobileNetV3 from the cached DeepLab COCO weights (``--init coco``) or from that fold's tissue CNN
(``--init tissue``, which never saw the held-out fold either).

    python -m training.train_stage_cnn --fold 0 --init coco --out runs/staging/b_coco_fold0.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from training.petchs_groups import RF_PETCHS, source, source_folds, stage_table

SQ = 384
CLASSES = ("1", "2", "3", "4", "U")
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)


def device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


class StageNet(torch.nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
        self.head = torch.nn.Sequential(torch.nn.Dropout(0.2), torch.nn.Linear(960, len(CLASSES)))

    def forward(self, x):
        return self.head(self.backbone(x)["out"].mean((2, 3)))


def build(init: str, fold: int) -> StageNet:
    if init == "coco":
        from training.train_wound_deeplab import build_model

        return StageNet(build_model(pretrained=True).backbone)
    from training.train_tissue_cnn import load

    return StageNet(load(f"runs/tissue_cnn_fold/fold{fold}.pt").cpu().backbone)


def letterbox(bgr: np.ndarray) -> torch.Tensor:
    h, w = bgr.shape[:2]
    f = SQ / max(h, w)
    nh, nw = max(1, round(h * f)), max(1, round(w * f))
    out = np.zeros((SQ, SQ, 3), np.uint8)
    out[:nh, :nw] = cv2.resize(bgr, (nw, nh), interpolation=cv2.INTER_AREA)
    rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(((rgb - _MEAN) / _STD).transpose(2, 0, 1).copy())


def augment(img, rng):
    h, w = img.shape[:2]
    s = rng.uniform(0.5, 1.0)
    ch, cw = max(16, int(h * s)), max(16, int(w * s))
    y0, x0 = int(rng.integers(0, h - ch + 1)), int(rng.integers(0, w - cw + 1))
    img = img[y0:y0 + ch, x0:x0 + cw]
    if rng.random() < 0.5:
        img = img[:, ::-1]
    if rng.random() < 0.5:
        img = img[::-1]
    img = np.ascontiguousarray(np.rot90(img, int(rng.integers(0, 4))))
    # brightness / contrast only: colour carries the tissue signal
    return np.clip(img.astype(np.float32) * rng.uniform(0.85, 1.15) + rng.uniform(-15, 15), 0, 255).astype(np.uint8)


def balanced_accuracy(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean([(p[y == c] == c).mean() for c in np.unique(y)]))


@torch.no_grad()
def predict(model, imgs) -> np.ndarray:
    model.eval()
    dev = next(model.parameters()).device
    return np.concatenate([torch.softmax(model(torch.stack([letterbox(im) for im in imgs[i:i + 16]]).to(dev)), 1)
                           .cpu().numpy() for i in range(0, len(imgs), 16)])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--init", choices=("coco", "tissue"), default="coco")
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    folds = source_folds()
    rows = [(f, CLASSES.index(st), cv2.imread(str(RF_PETCHS / sp / f))) for sp, f, st in stage_table() if st]
    train_src = sorted({source(f) for f, _, _ in rows if folds[source(f)] != args.fold})
    val_src = set(train_src[5::6])
    train = [r for r in rows if folds[source(r[0])] != args.fold and source(r[0]) not in val_src]
    val = [r for r in rows if source(r[0]) in val_src]
    held = [r for r in rows if folds[source(r[0])] == args.fold]
    print(f"fold {args.fold}: train {len(train)} / internal val {len(val)} / held-out {len(held)}", flush=True)

    dev = device()
    model = build(args.init, args.fold).to(dev)
    counts = np.bincount([y for _, y, _ in train], minlength=len(CLASSES)).astype(np.float64)
    weight = torch.tensor((counts.sum() / (len(CLASSES) * np.maximum(counts, 1))) ** 0.5, dtype=torch.float32, device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best, best_state = (-1.0, -1), None
    yv = np.array([y for _, y, _ in val])
    for ep in range(args.epochs):
        model.train()
        order = rng.permutation(len(train))
        losses = []
        for i in range(0, len(order), args.batch):
            idx = order[i:i + args.batch]
            if len(idx) < 2:  # BatchNorm needs >1 sample
                continue
            x = torch.stack([letterbox(augment(train[j][2], rng)) for j in idx]).to(dev)
            y = torch.tensor([train[j][1] for j in idx], device=dev)
            loss = F.cross_entropy(model(x), y, weight=weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()
        ba = balanced_accuracy(yv, predict(model, [im for _, _, im in val]).argmax(1))
        if ba > best[0]:
            best = (ba, ep)
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        print(f"ep {ep:02d} loss {np.mean(losses):.3f} val-bal-acc {ba:.3f} best {best[0]:.3f}@{best[1]}", flush=True)
    model.load_state_dict(best_state)
    probs = predict(model, [im for _, _, im in held])
    out = {f: {"pred": CLASSES[int(p.argmax())], "probs": [round(float(v), 4) for v in p]} for (f, _, _), p in zip(held, probs)}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps({"fold": args.fold, "init": args.init, "best_epoch": best[1],
                                          "internal_val_bal_acc": round(best[0], 3), "pred": out}, indent=1))
    print(f"saved {args.out} (internal val balanced accuracy {best[0]:.3f}, epoch {best[1]})")


if __name__ == "__main__":
    main()

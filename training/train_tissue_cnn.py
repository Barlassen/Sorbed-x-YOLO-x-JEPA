"""B-CNN: a pretrained segmentation network for wound tissue, trained on DFUTissue only.

Answers the objection that the A/B comparison's B (a pixel random forest) was too weak. The model is
torchvision's DeepLabV3-MobileNetV3 (COCO-pretrained), re-headed for 3 tissues: fibrin, granulation,
callus. It sees the same input as the other methods: the wound crop from ``tissue_ab_compare`` (longer
side 160 px), letterboxed to a 256 px square. Pixels outside the wound are ignored by the loss.

Every 6th DFUTissue TrainVal image is held out as an internal validation set for model selection
(mean Dice over the 3 tissues); DFUTissue Test and ComplexWoundDB are never touched here.

    python -m training.train_tissue_cnn --seed 0 --out runs/tissue_cnn/seed0.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from training.tissue_ab_compare import crop_resize, dfut_items

SQ = 256
IGNORE = 255
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)
_TRAIN_LABEL = {2: 0, 1: 1, 4: 2}  # tissue_ab_compare ids (S, G, O) -> network classes (fibrin, gran, callus)
NET_TO_AB = np.array([2, 1, 4], np.uint8)  # network class -> tissue_ab_compare id


def device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def build_model(pretrained: bool = True):
    from torchvision.models.segmentation import DeepLabV3_MobileNet_V3_Large_Weights, deeplabv3_mobilenet_v3_large

    m = deeplabv3_mobilenet_v3_large(weights=DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1 if pretrained else None,
                                     weights_backbone=None, aux_loss=None)
    m.classifier[-1] = torch.nn.Conv2d(m.classifier[-1].in_channels, 3, 1)
    m.aux_classifier = None
    return m


def letterbox(img: np.ndarray, interp, fill=0):
    h, w = img.shape[:2]
    f = SQ / max(h, w)
    nh, nw = max(1, round(h * f)), max(1, round(w * f))
    r = cv2.resize(img, (nw, nh), interpolation=interp)
    out = np.full((SQ, SQ) + img.shape[2:], fill, img.dtype)
    out[:nh, :nw] = r
    return out, (nh, nw)


def to_tensor(bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(((rgb - _MEAN) / _STD).transpose(2, 0, 1).copy())


@torch.no_grad()
def predict(model, crop_bgr: np.ndarray, region: np.ndarray) -> np.ndarray:
    """Crop (as produced by crop_resize) -> tissue_ab_compare label map (G/S/O inside region)."""
    x, (nh, nw) = letterbox(crop_bgr, cv2.INTER_AREA)
    logits = model(to_tensor(x)[None].to(next(model.parameters()).device))["out"][0, :, :nh, :nw]
    cls = logits.argmax(0).cpu().numpy().astype(np.uint8)
    cls = cv2.resize(cls, (crop_bgr.shape[1], crop_bgr.shape[0]), interpolation=cv2.INTER_NEAREST)
    lbl = NET_TO_AB[cls]
    lbl[~region] = 0
    return lbl


def load(path: str):
    m = build_model(pretrained=False)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m.to(device()).eval()


def samples():
    out = []
    for name, img, gt in dfut_items("TrainVal"):
        crop, reg, (g,), _ = crop_resize(img, gt > 0, gt)
        t = np.full(g.shape, IGNORE, np.uint8)
        for k, v in _TRAIN_LABEL.items():
            t[(g == k) & reg] = v
        out.append((name, crop, t))
    return out


def augment(crop, target, rng):
    if rng.random() < 0.5:
        crop, target = crop[:, ::-1], target[:, ::-1]
    if rng.random() < 0.5:
        crop, target = crop[::-1], target[::-1]
    k = int(rng.integers(0, 4))
    crop, target = np.rot90(crop, k), np.rot90(target, k)
    s = rng.uniform(0.8, 1.2)
    h, w = target.shape
    size = (max(8, round(w * s)), max(8, round(h * s)))
    crop = cv2.resize(np.ascontiguousarray(crop), size, interpolation=cv2.INTER_LINEAR)
    target = cv2.resize(np.ascontiguousarray(target), size, interpolation=cv2.INTER_NEAREST)
    # brightness / contrast only: hue carries the tissue signal, so it is not jittered
    crop = np.clip(crop.astype(np.float32) * rng.uniform(0.85, 1.15) + rng.uniform(-15, 15), 0, 255).astype(np.uint8)
    return crop, target


def dice3(model, items) -> float:
    inter, tot = np.zeros(3), np.zeros(3)
    for _, crop, t in items:
        reg = t != IGNORE
        p = predict(model, crop, reg)
        for c, ab in enumerate(NET_TO_AB):
            a, b = (p == ab) & reg, t == c
            inter[c] += (a & b).sum()
            tot[c] += a.sum() + b.sum()
    return float(np.mean(2 * inter / np.maximum(tot, 1)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    items = samples()
    val = items[5::6]
    train = [x for i, x in enumerate(items) if i % 6 != 5]
    print(f"train {len(train)} / internal val {len(val)}")

    dev = device()
    model = build_model().to(dev)
    counts = np.bincount(np.concatenate([t[t != IGNORE] for _, _, t in train]), minlength=3).astype(np.float64)
    weight = torch.tensor((counts.sum() / (3 * counts)) ** 0.5, dtype=torch.float32, device=dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    best = (-1.0, -1)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    for ep in range(args.epochs):
        model.train()
        order = rng.permutation(len(train))
        losses = []
        for i in range(0, len(order), args.batch):
            xs, ys = [], []
            for j in order[i : i + args.batch]:
                c, t = augment(train[j][1], train[j][2], rng)
                x, _ = letterbox(c, cv2.INTER_AREA)
                y, _ = letterbox(t, cv2.INTER_NEAREST, fill=IGNORE)
                xs.append(to_tensor(x)), ys.append(torch.from_numpy(y.astype(np.int64)))
            if len(xs) < 2:  # BatchNorm needs >1 sample
                continue
            out = model(torch.stack(xs).to(dev))["out"]
            loss = F.cross_entropy(out, torch.stack(ys).to(dev), weight=weight, ignore_index=IGNORE)
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        sched.step()
        model.eval()
        d = dice3(model, val)
        if d > best[0]:
            best = (d, ep)
            torch.save(model.state_dict(), args.out)
        print(f"ep {ep:02d} loss {np.mean(losses):.3f} val-dice {d:.3f} best {best[0]:.3f}@{best[1]}", flush=True)
    print(f"saved {args.out} (internal val mean Dice {best[0]:.3f}, epoch {best[1]})")


if __name__ == "__main__":
    main()

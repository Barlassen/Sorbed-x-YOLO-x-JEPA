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
_TRAIN_LABEL = {2: 0, 1: 1, 4: 2, 3: 3}  # tissue_ab_compare ids (S, G, O, N) -> network classes
NET_TO_AB = {3: np.array([2, 1, 4], np.uint8), 4: np.array([2, 1, 4, 3], np.uint8)}  # by number of classes

LUTSEG = Path("data/corpus/lutseg")
WOUNDTISSUE = Path("data/corpus/woundtissue")
# LUTSeg ids: 0 bg, 1 epithelial, 2 slough, 3 granulation, 4 necrotic, 5 other, 255 ignore -> ab ids (255 = ignore)
_LUT_TO_AB = np.full(256, 255, np.uint8)
_LUT_TO_AB[[0, 1, 2, 3, 4]] = [0, 4, 2, 1, 3]
# WoundTissue colours (RGB): bg blue, granulation red, slough yellow, necrosis black, maceration white, bone cream,
# tendon sky blue -> ab ids (bone/tendon too rare to learn -> ignore)
_WT_PALETTE = np.array([(0, 0, 255), (255, 0, 0), (255, 255, 0), (0, 0, 0), (255, 255, 255), (255, 200, 124),
                        (135, 206, 235)], np.float32)
_WT_TO_AB = np.array([0, 1, 2, 3, 4, 255, 255], np.uint8)


def device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def build_model(pretrained: bool = True, n_cls: int = 3):
    from torchvision.models.segmentation import DeepLabV3_MobileNet_V3_Large_Weights, deeplabv3_mobilenet_v3_large

    m = deeplabv3_mobilenet_v3_large(weights=DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1 if pretrained else None,
                                     weights_backbone=None, aux_loss=None)
    m.classifier[-1] = torch.nn.Conv2d(m.classifier[-1].in_channels, n_cls, 1)
    m.aux_classifier = None
    m.n_cls = n_cls
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
    lbl = NET_TO_AB[model.n_cls][cls]
    lbl[~region] = 0
    return lbl


def load(path: str):
    sd = torch.load(path, map_location="cpu")
    m = build_model(pretrained=False, n_cls=sd["classifier.4.weight"].shape[0])
    m.load_state_dict(sd)
    return m.to(device()).eval()


def _target(crop_label, reg, n_cls):
    t = np.full(crop_label.shape, IGNORE, np.uint8)
    for ab, net in _TRAIN_LABEL.items():
        if net < n_cls:
            t[(crop_label == ab) & reg] = net
    return t


def samples(multi: bool = False):
    """(name, crop, target, is_internal_val). DFUT: every 6th TrainVal image is internal val."""
    n_cls = 4 if multi else 3
    out = []
    for i, (name, img, gt) in enumerate(dfut_items("TrainVal")):
        crop, reg, (g,), _ = crop_resize(img, gt > 0, gt)
        out.append((f"dfut/{name}", crop, _target(g, reg, n_cls), i % 6 == 5))
    if not multi:
        return out
    val = set(Path(LUTSEG / "val.txt").read_text().split())
    for line in (LUTSEG / "metadata.jsonl").read_text().splitlines():
        import json

        r = json.loads(line)
        img = cv2.imread(str(LUTSEG / r["image_file_name"]))
        raw = cv2.imread(str(LUTSEG / r["mask_file_name"]), cv2.IMREAD_UNCHANGED)
        if img is None or raw is None or raw.shape != img.shape[:2]:
            continue
        reg = (raw > 0) & (raw != 255)
        if reg.sum() < 50:
            continue
        crop, creg, (g,), _ = crop_resize(img, reg, _LUT_TO_AB[raw])
        is_val = r["split"] == "validation" or r["image_file_name"] in val
        out.append((f"lutseg/{r['image_id']}", crop, _target(g, creg, n_cls), is_val))
    for p in sorted((WOUNDTISSUE / "image").glob("*.png")):
        img, lab = cv2.imread(str(p)), cv2.imread(str(WOUNDTISSUE / "label" / p.name))
        if img is None or lab is None or lab.shape != img.shape:
            continue
        rgb = lab[..., ::-1].reshape(-1, 1, 3).astype(np.float32)
        ab = _WT_TO_AB[np.argmin(((rgb - _WT_PALETTE[None]) ** 2).sum(-1), 1)].reshape(lab.shape[:2])
        reg = ab != 0
        if reg.sum() < 50:
            continue
        crop, creg, (g,), _ = crop_resize(img, reg, ab)
        out.append((f"woundtissue/{p.stem}", crop, _target(g, creg, n_cls), False))
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


def dice_mean(model, items) -> float:
    n = model.n_cls
    inter, tot = np.zeros(n), np.zeros(n)
    for _, crop, t, _ in items:
        reg = t != IGNORE
        p = predict(model, crop, reg)
        for c, ab in enumerate(NET_TO_AB[n]):
            a, b = (p == ab) & reg, t == c
            inter[c] += (a & b).sum()
            tot[c] += a.sum() + b.sum()
    return float(np.mean((2 * inter / np.maximum(tot, 1))[tot > 0]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--out", required=True)
    ap.add_argument("--multi", action="store_true", help="add LUTSeg + WoundTissue and a 4th class (necrosis)")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)

    items = samples(args.multi)
    val = [x for x in items if x[3]]
    train = [x for x in items if not x[3]]
    n_cls = 4 if args.multi else 3
    print(f"train {len(train)} / internal val {len(val)}")

    dev = device()
    model = build_model(n_cls=n_cls).to(dev)
    counts = np.bincount(np.concatenate([t[t != IGNORE] for _, _, t, _ in train]), minlength=n_cls).astype(np.float64)
    print("pixels per class", counts.astype(int).tolist())
    weight = torch.tensor((counts.sum() / (n_cls * np.maximum(counts, 1))) ** 0.5, dtype=torch.float32, device=dev)
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
        d = dice_mean(model, val)
        if d > best[0]:
            best = (d, ep)
            torch.save(model.state_dict(), args.out)
        print(f"ep {ep:02d} loss {np.mean(losses):.3f} val-dice {d:.3f} best {best[0]:.3f}@{best[1]}", flush=True)
    print(f"saved {args.out} (internal val mean Dice {best[0]:.3f}, epoch {best[1]})")


if __name__ == "__main__":
    main()

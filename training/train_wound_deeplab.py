"""Wound mask as per-pixel classification: DeepLabV3-MobileNetV3, wound vs background, full image.

YOLO segments a wound as boxed *instances*; when the wound fills the frame the box is the whole image and the
wound breaks into many low-confidence pieces. A semantic model labels every pixel and has no box. This trains one on
the same YOLO-format dataset as the YOLO segmenter (polygons are rasterised), selects the epoch by mean Dice on the
dataset's ``val`` split (the pressure-injury validation set), and keeps a fixed 0.5 threshold.

Random crops of 45-100% of the frame are rescaled to the input size during training, so the model sees both wide
shots and close-ups.

    python -m training.train_wound_deeplab --data data/yolo_seg_v4 --out runs/wound_deeplab/v4.pt
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

SQ = 512
IGNORE = 255
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)
_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def device() -> str:
    return "mps" if torch.backends.mps.is_available() else "cpu"


def build_model(pretrained: bool = True):
    from torchvision.models.segmentation import DeepLabV3_MobileNet_V3_Large_Weights, deeplabv3_mobilenet_v3_large

    m = deeplabv3_mobilenet_v3_large(weights=DeepLabV3_MobileNet_V3_Large_Weights.COCO_WITH_VOC_LABELS_V1 if pretrained else None,
                                     weights_backbone=None, aux_loss=None)
    m.classifier[-1] = torch.nn.Conv2d(m.classifier[-1].in_channels, 2, 1)
    m.aux_classifier = None
    return m


def load(path: str):
    m = build_model(pretrained=False)
    m.load_state_dict(torch.load(path, map_location="cpu"))
    return m.to(device()).eval()


def read_item(img_path: Path):
    """Image + binary wound mask rasterised from the YOLO polygon label next to it."""
    img = cv2.imread(str(img_path))
    lbl = Path(str(img_path).replace("/images/", "/labels/")).with_suffix(".txt")
    mask = np.zeros(img.shape[:2], np.uint8)
    if lbl.exists():
        h, w = mask.shape
        for line in lbl.read_text().splitlines():
            p = line.split()[1:]
            if len(p) >= 6:
                pts = (np.array(p, np.float32).reshape(-1, 2) * [w, h]).astype(np.int32)
                cv2.fillPoly(mask, [pts], 1)
    return img, mask


def letterbox(a: np.ndarray, interp, fill=0):
    h, w = a.shape[:2]
    f = SQ / max(h, w)
    nh, nw = max(1, round(h * f)), max(1, round(w * f))
    out = np.full((SQ, SQ) + a.shape[2:], fill, a.dtype)
    out[:nh, :nw] = cv2.resize(a, (nw, nh), interpolation=interp)
    return out, (nh, nw)


def to_tensor(bgr: np.ndarray) -> torch.Tensor:
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return torch.from_numpy(((rgb - _MEAN) / _STD).transpose(2, 0, 1).copy())


@torch.no_grad()
def predict_mask(model, bgr: np.ndarray, threshold: float = 0.5) -> np.ndarray:
    """Full image -> boolean wound mask at the image's own size."""
    x, (nh, nw) = letterbox(bgr, cv2.INTER_AREA)
    prob = torch.softmax(model(to_tensor(x)[None].to(next(model.parameters()).device))["out"], 1)[0, 1, :nh, :nw]
    prob = cv2.resize(prob.cpu().numpy(), (bgr.shape[1], bgr.shape[0]), interpolation=cv2.INTER_LINEAR)
    return prob > threshold


def augment(img, mask, rng):
    h, w = mask.shape
    s = rng.uniform(0.45, 1.0)  # crop side as a fraction of the frame -> close-ups
    ch, cw = max(8, int(h * s)), max(8, int(w * s))
    y0, x0 = int(rng.integers(0, h - ch + 1)), int(rng.integers(0, w - cw + 1))
    img, mask = img[y0:y0 + ch, x0:x0 + cw], mask[y0:y0 + ch, x0:x0 + cw]
    if rng.random() < 0.5:
        img, mask = img[:, ::-1], mask[:, ::-1]
    if rng.random() < 0.5:
        img, mask = img[::-1], mask[::-1]
    k = int(rng.integers(0, 4))
    img, mask = np.ascontiguousarray(np.rot90(img, k)), np.ascontiguousarray(np.rot90(mask, k))
    img = np.clip(img.astype(np.float32) * rng.uniform(0.85, 1.15) + rng.uniform(-15, 15), 0, 255).astype(np.uint8)
    return img, mask


def dice(a: np.ndarray, b: np.ndarray) -> float:
    return float(2 * (a & b).sum() / (a.sum() + b.sum() + 1e-9))


def val_dice(model, paths) -> float:
    model.eval()
    return float(np.mean([dice(predict_mask(model, img), m > 0) for img, m in (read_item(p) for p in paths)]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, type=Path, help="YOLO seg dataset dir (images/{train,val}, labels/...)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="continue from <out>.last.pt after an interruption")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    train = sorted(p for p in (args.data / "images" / "train").iterdir() if p.suffix.lower() in _EXTS)
    val = sorted(p for p in (args.data / "images" / "val").iterdir() if p.suffix.lower() in _EXTS)
    print(f"train {len(train)} / val {len(val)}", flush=True)

    dev = device()
    model = build_model().to(dev)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    steps = args.epochs * ((len(train) + args.batch - 1) // args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.05)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    best = (-1.0, -1)
    start = 0
    last_path = Path(args.out).with_suffix(".last.pt")
    if args.resume and last_path.exists():
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"]), opt.load_state_dict(ck["opt"]), sched.load_state_dict(ck["sched"])
        start, best = ck["epoch"] + 1, tuple(ck["best"])
        rng.bit_generator.state = ck["rng"]
        torch.set_rng_state(ck["torch_rng"])
        print(f"resumed after epoch {ck['epoch']} (best {best[0]:.3f}@{best[1]})", flush=True)
    for ep in range(start, args.epochs):
        model.train()
        order = rng.permutation(len(train))
        losses = []
        for i in range(0, len(order), args.batch):
            xs, ys = [], []
            for j in order[i:i + args.batch]:
                img, m = augment(*read_item(train[j]), rng)
                x, _ = letterbox(img, cv2.INTER_AREA)
                y, _ = letterbox(m, cv2.INTER_NEAREST, fill=IGNORE)
                xs.append(to_tensor(x)), ys.append(torch.from_numpy(y.astype(np.int64)))
            if len(xs) < 2:
                continue
            out = model(torch.stack(xs).to(dev))["out"]
            y = torch.stack(ys).to(dev)
            ce = F.cross_entropy(out, y, ignore_index=IGNORE)
            valid = (y != IGNORE).float()
            p = torch.softmax(out, 1)[:, 1] * valid
            t = (y == 1).float() * valid
            soft_dice = 1 - (2 * (p * t).sum() + 1) / (p.sum() + t.sum() + 1)
            loss = ce + soft_dice
            opt.zero_grad()
            loss.backward()
            opt.step()
            sched.step()
            losses.append(loss.item())
        d = val_dice(model, val)
        if d > best[0]:
            best = (d, ep)
            torch.save(model.state_dict(), args.out)
        # full state every epoch, so a dead battery costs at most one epoch
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(), "sched": sched.state_dict(), "epoch": ep,
                    "best": list(best), "rng": rng.bit_generator.state, "torch_rng": torch.get_rng_state()}, last_path)
        print(f"ep {ep:02d} train-loss {np.mean(losses):.3f} val-dice {d:.3f} best {best[0]:.3f}@{best[1]}", flush=True)
    print(f"saved {args.out} (pressure val Dice {best[0]:.3f}, epoch {best[1]})")


if __name__ == "__main__":
    main()

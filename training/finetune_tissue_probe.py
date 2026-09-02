#!/usr/bin/env python3
"""Fine-tuning counterpart to ``validate_tissue_probe.py``: does JEPA help once the
encoder is allowed to adapt (not frozen)?

The frozen linear probe (``train_tissue_probe`` / ``validate_tissue_probe``) asks
"are JEPA's *ready-made* patch features better than random's?". That is the strict
test, and at laptop scale it comes out inconclusive. But the usual value of
self-supervised pre-training is a better *initialisation* for fine-tuning: starting
from JEPA weights should reach a better tissue classifier, with the same little
labelled data, than starting from random weights. The frozen probe cannot see that
— this script does.

What it does, keeping every comparison fair and identical to the frozen validation:

* Model = the same ``Encoder`` + a tiny linear head over the 196 patch tokens
  (4 classes: background + fibrin/granulation/callus).
* ``--unfreeze-blocks N`` unfreezes only the last ``N`` transformer blocks (plus the
  final norm and the head); the rest of the encoder stays frozen. N=0 trains the
  head only; N=6 (enc_depth) is full fine-tuning. Default 1 — conservative, because
  the Test split is ~16 images and full fine-tuning on ~110 TrainVal images can
  overfit. Both inits get the *same* N.
* Trains on the TrainVal split (~110 images ≈ 21k patches), class-balanced
  cross-entropy, evaluates macro-F1 over the tissue classes on the Test split —
  the same metric, splits, ``--crop/--depth/--relief`` preprocessing and tissue
  ids as the frozen probe.

Two uncertainty axes, mirroring ``validate_tissue_probe.py``. Note the difference:
fine-tuning is *stochastic for both inits* (head init + data order depend on the
seed), so JEPA is no longer a single point — it too becomes a seed distribution.

1. **Multi-seed (training/init uncertainty).** Fine-tune both inits under N seeds;
   report mean ± std for each and the *paired per-seed* delta (JEPA[s] − random[s]),
   which cancels the shared data-order noise seed by seed.
2. **Bootstrap (test-set uncertainty).** Resample the ~16 Test *images* with
   replacement B times; a paired image-level bootstrap on the median-scoring seed of
   each init gives the (JEPA − random) delta CI and P(delta > 0).

Only when the per-seed delta is consistently positive *and* the paired bootstrap CI
stays above 0 is a JEPA fine-tuning benefit defensible.

    python -m training.finetune_tissue_probe \
        --weights runs_jepa/jepa_best.pt --crop --depth data/rgbd_tissue_labeled/depth \
        --relief --unfreeze-blocks 1 --epochs 40 --seeds 0 1 2 3 4 --bootstrap 2000 \
        --out runs_jepa/tissue_finetune.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score

from training.train_jepa import Encoder, JepaConfig
from training.train_tissue_probe import _ROOT, _TISSUE, _prepare

_NUM_CLASSES = 4  # 0 = background, 1..3 = fibrin / granulation / callus


def _load_split(split: str, cfg: JepaConfig, depth_dir, crop: bool, relief: bool):
    """Load every annotated image in ``split`` into (input tensor, patch-label) pairs.

    Kept on CPU as a list so the training loop can batch/shuffle and the evaluation
    can score image by image (which the bootstrap needs).
    """
    img_dir = _ROOT / "Images" / split
    ann_dir = _ROOT / "Annotations" / split
    items = []
    for img_path in sorted(img_dir.glob("*.png")):
        ann = ann_dir / img_path.name
        if not ann.exists():
            continue
        t, y = _prepare(img_path, ann, cfg, depth_dir, crop, relief)
        items.append((t, torch.from_numpy(y)))
    if not items:
        raise FileNotFoundError(f"no annotated images under {img_dir}")
    return items


class TissueModel(nn.Module):
    """Encoder trunk + a linear per-patch tissue head over the 196 tokens."""

    def __init__(self, cfg: JepaConfig) -> None:
        super().__init__()
        self.encoder = Encoder(cfg)
        self.head = nn.Linear(cfg.enc_dim, _NUM_CLASSES)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        feats = self.encoder(images)          # (B, N, D)
        return self.head(feats)               # (B, N, num_classes)


def _set_trainable(model: TissueModel, unfreeze_blocks: int) -> list[nn.Parameter]:
    """Freeze the whole encoder, then re-enable the last ``unfreeze_blocks`` trunk
    blocks plus the final norm. The head is always trainable. Returns the encoder
    params left trainable (so the optimiser can give them a smaller LR than the head).
    """
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    enc_trainable: list[nn.Parameter] = []
    layers = model.encoder.trunk.blocks.layers          # ModuleList, len == enc_depth
    n = max(0, min(unfreeze_blocks, len(layers)))
    if n > 0:
        for blk in layers[-n:]:
            for p in blk.parameters():
                p.requires_grad_(True)
                enc_trainable.append(p)
        for p in model.encoder.trunk.norm.parameters():  # final LayerNorm
            p.requires_grad_(True)
            enc_trainable.append(p)
    return enc_trainable


def _class_weights(items, device: str) -> torch.Tensor:
    """Balanced class weights (n / (K * count_c)) over classes present in train."""
    counts = torch.zeros(_NUM_CLASSES, dtype=torch.float64)
    for _, y in items:
        counts += torch.bincount(y, minlength=_NUM_CLASSES).to(torch.float64)
    present = counts > 0
    w = torch.zeros(_NUM_CLASSES, dtype=torch.float32)
    w[present] = float(counts.sum()) / (int(present.sum()) * counts[present].to(torch.float32))
    return w.to(device)


def _finetune(cfg: JepaConfig, device: str, *, init: str, weights: str | None, seed: int,
              train_items, unfreeze_blocks: int, epochs: int, batch: int,
              enc_lr: float, head_lr: float, weight_decay: float) -> TissueModel:
    """Build a model (JEPA- or random-init), fine-tune it on the tissue task, return it."""
    torch.manual_seed(seed)
    model = TissueModel(cfg).to(device)
    if init == "jepa":
        state = torch.load(weights, map_location="cpu")
        model.encoder.load_state_dict(state, strict=False)
    enc_params = _set_trainable(model, unfreeze_blocks)

    groups = [{"params": model.head.parameters(), "lr": head_lr}]
    if enc_params:
        groups.append({"params": enc_params, "lr": enc_lr})
    opt = torch.optim.AdamW(groups, weight_decay=weight_decay)
    weight = _class_weights(train_items, device)

    g = torch.Generator().manual_seed(seed)      # reproducible shuffling per seed
    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(train_items), generator=g)
        for i in range(0, len(order), batch):
            idx = order[i:i + batch]
            x = torch.stack([train_items[j][0] for j in idx]).to(device)
            y = torch.stack([train_items[j][1] for j in idx]).to(device)  # (B, N)
            logits = model(x)                                            # (B, N, C)
            loss = F.cross_entropy(logits.reshape(-1, _NUM_CLASSES), y.reshape(-1), weight=weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
    model.eval()
    return model


@torch.no_grad()
def _predict_per_image(model: TissueModel, test_items, device: str):
    """Per-image (pred, label) arrays so the bootstrap can resample whole images."""
    out = []
    for t, y in test_items:
        logits = model(t.unsqueeze(0).to(device))[0]       # (N, C)
        out.append((logits.argmax(-1).cpu().numpy(), y.numpy()))
    return out


def _macro_f1(pred_imgs, tissue_ids) -> float:
    y_true = np.concatenate([y for _, y in pred_imgs])
    y_pred = np.concatenate([p for p, _ in pred_imgs])
    return float(f1_score(y_true, y_pred, labels=tissue_ids, average="macro", zero_division=0))


def _paired_delta(jepa_imgs, rand_imgs, tissue_ids, *, n_boot: int, seed: int):
    """Paired image-level bootstrap of (JEPA − random) macro-F1 on the Test split."""
    rng = np.random.default_rng(seed)
    n = len(jepa_imgs)
    deltas = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        fj = _macro_f1([jepa_imgs[i] for i in idx], tissue_ids)
        fr = _macro_f1([rand_imgs[i] for i in idx], tissue_ids)
        deltas[b] = fj - fr
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {"mean": float(deltas.mean()), "lo": float(lo), "hi": float(hi),
            "p_gt0": float(np.mean(deltas > 0))}


def run(args) -> None:
    device = args.device
    in_chans = 3 + (1 if args.depth else 0) + (1 if (args.relief and args.depth) else 0)
    cfg = JepaConfig(img_size=args.img_size, in_chans=in_chans)
    print(f"device={device}  in_chans={in_chans}  unfreeze_blocks={args.unfreeze_blocks}  "
          f"epochs={args.epochs}  seeds={args.seeds}  bootstrap={args.bootstrap}")

    train_items = _load_split("TrainVal", cfg, args.depth, args.crop, args.relief)
    test_items = _load_split("Test", cfg, args.depth, args.crop, args.relief)
    # Fixed tissue-class set from the full test split, so every seed/resample scores
    # the same classes (a dropped rare class still counts, as F1 0 — honest on 16 imgs).
    all_test_labels = np.concatenate([y.numpy() for _, y in test_items])
    tissue_ids = [c for c in _TISSUE if c in set(all_test_labels)]
    print(f"train images={len(train_items)}  test images={len(test_items)}  tissue_ids={tissue_ids}\n")

    ft = dict(unfreeze_blocks=args.unfreeze_blocks, epochs=args.epochs, batch=args.batch,
              enc_lr=args.enc_lr, head_lr=args.head_lr, weight_decay=args.weight_decay)

    jepa_f1, rand_f1 = [], []
    jepa_preds_by_seed, rand_preds_by_seed = {}, {}
    for seed in args.seeds:
        jm = _finetune(cfg, device, init="jepa", weights=args.weights, seed=seed,
                       train_items=train_items, **ft)
        jp = _predict_per_image(jm, test_items, device)
        fj = _macro_f1(jp, tissue_ids)
        jepa_f1.append(fj); jepa_preds_by_seed[seed] = jp

        rm = _finetune(cfg, device, init="random", weights=None, seed=seed,
                       train_items=train_items, **ft)
        rp = _predict_per_image(rm, test_items, device)
        fr = _macro_f1(rp, tissue_ids)
        rand_f1.append(fr); rand_preds_by_seed[seed] = rp
        print(f"seed={seed}   JEPA-init={fj:.3f}   random-init={fr:.3f}   Δ={fj - fr:+.3f}")

    j = np.array(jepa_f1); r = np.array(rand_f1)
    jm_, js = float(j.mean()), float(j.std(ddof=1) if len(j) > 1 else 0.0)
    rm_, rs = float(r.mean()), float(r.std(ddof=1) if len(r) > 1 else 0.0)
    per_seed_delta = j - r
    dsm, dss = float(per_seed_delta.mean()), float(per_seed_delta.std(ddof=1) if len(j) > 1 else 0.0)
    print(f"\nJEPA-init   : mean={jm_:.3f}  std={js:.3f}  min={j.min():.3f}  max={j.max():.3f}")
    print(f"random-init : mean={rm_:.3f}  std={rs:.3f}  min={r.min():.3f}  max={r.max():.3f}")
    print(f"paired per-seed Δ (JEPA−random): mean={dsm:+.3f}  std={dss:.3f}  "
          f"n(Δ>0)={int((per_seed_delta > 0).sum())}/{len(per_seed_delta)}")

    report = {
        "config": {
            "weights": str(args.weights), "img_size": args.img_size, "depth": str(args.depth),
            "crop": args.crop, "relief": args.relief, "seeds": list(args.seeds),
            "unfreeze_blocks": args.unfreeze_blocks, "epochs": args.epochs, "batch": args.batch,
            "enc_lr": args.enc_lr, "head_lr": args.head_lr, "weight_decay": args.weight_decay,
            "bootstrap": args.bootstrap, "tissue_ids": tissue_ids,
        },
        "jepa_seeds": {str(s): f for s, f in zip(args.seeds, jepa_f1, strict=True)},
        "random_seeds": {str(s): f for s, f in zip(args.seeds, rand_f1, strict=True)},
        "jepa_summary": {"mean": jm_, "std": js, "min": float(j.min()), "max": float(j.max())},
        "random_summary": {"mean": rm_, "std": rs, "min": float(r.min()), "max": float(r.max())},
        "paired_seed_delta": {"mean": dsm, "std": dss,
                              "n_pos": int((per_seed_delta > 0).sum()), "n": len(per_seed_delta)},
    }

    verdict = dsm > 0 and (per_seed_delta > 0).all()
    if args.bootstrap > 0:
        # Image-level bootstrap on the median-scoring seed of each init (representative).
        jseed = args.seeds[int(np.argsort(jepa_f1)[len(jepa_f1) // 2])]
        rseed = args.seeds[int(np.argsort(rand_f1)[len(rand_f1) // 2])]
        delta = _paired_delta(jepa_preds_by_seed[jseed], rand_preds_by_seed[rseed],
                              tissue_ids, n_boot=args.bootstrap, seed=args.boot_seed)
        report["bootstrap"] = {"jepa_seed": jseed, "random_seed": rseed, "paired_delta_ci95": delta}
        print(f"\nbootstrap 95% CI  (B={args.bootstrap}, image-level, JEPA s={jseed} vs random s={rseed})")
        print(f"  paired Δ : {delta['mean']:+.3f}  [{delta['lo']:+.3f}, {delta['hi']:+.3f}]  "
              f"P(Δ>0)={delta['p_gt0']:.3f}")
        verdict = verdict and (delta["lo"] > 0.0)

    print(f"\nVERDICT: JEPA fine-tuning > random is {'SUPPORTED' if verdict else 'NOT established'} "
          f"(needs every per-seed Δ>0 and the paired bootstrap Δ CI above 0).")
    report["verdict_supported"] = bool(verdict)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Fine-tuning JEPA-vs-random tissue evaluation.")
    ap.add_argument("--weights", default="runs_jepa/jepa_best.pt", help="JEPA encoder weights.")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--depth", type=Path, help="Depth-PNG dir, 2.5D input (match pretraining).")
    ap.add_argument("--crop", action="store_true", help="Crop to the wound box (match pretrain).")
    ap.add_argument("--relief", action="store_true", help="Add relief channel (match pretraining).")
    ap.add_argument("--unfreeze-blocks", type=int, default=1,
                    help="Trailing transformer blocks to unfreeze (0=head only, 6=full).")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--enc-lr", type=float, default=1e-4, help="LR for the unfrozen encoder blocks.")
    ap.add_argument("--head-lr", type=float, default=1e-3, help="LR for the linear head.")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                    help="Seeds; both inits are fine-tuned under each (fine-tuning is stochastic).")
    ap.add_argument("--bootstrap", type=int, default=2000, help="Bootstrap resamples (0 disables).")
    ap.add_argument("--boot-seed", type=int, default=12345, help="RNG seed for the bootstrap.")
    ap.add_argument("--out", type=Path, help="Write the full report as JSON.")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    run(ap.parse_args())


if __name__ == "__main__":
    main()

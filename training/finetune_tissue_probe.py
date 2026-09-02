#!/usr/bin/env python3
"""Fine-tuning counterpart to ``validate_tissue_probe.py``: does JEPA help once the
encoder is allowed to adapt (not frozen)?

The frozen linear probe (``validate_tissue_probe``) asks "are JEPA's *ready-made*
patch features better than random's?" — at laptop scale, inconclusive. The usual
value of self-supervised pre-training is instead a better *initialisation for
fine-tuning*: adapting from JEPA weights should beat adapting from random weights
on the same labelled data. This script measures that.

**Measurement instrument = the validated frozen probe, always.** Fine-tuning adapts
the encoder (a small torch head is only the vehicle that carries the task gradient
into the encoder); the reported macro-F1 is then computed by ``validate_tissue_probe``'s
*own* sklearn linear probe on the fine-tuned encoder's frozen features. This keeps the
metric identical to the validated baseline and gives a free sanity check: with
``--unfreeze-blocks 0`` the encoder never changes, so the score must reproduce the
frozen probe (JEPA≈random≈0.52). An earlier version scored with the torch head
directly and did *not* reproduce it — an unreliable instrument — which is exactly why
the metric now routes through the validated probe.

* Model = the same ``Encoder`` + a tiny linear per-patch head (4 classes: background +
  fibrin/granulation/callus), used only to fine-tune.
* ``--unfreeze-blocks N`` unfreezes the last N transformer blocks (+ final norm + head);
  the rest stays frozen. 0 = no adaptation (pure sanity check); 6 (enc_depth) = full.
  Default 1 — conservative, since Test is ~16 images. Both inits get the same N.
* Same TrainVal/Test splits, ``--crop/--depth/--relief`` preprocessing and tissue ids
  as the frozen probe.

Two uncertainty axes, mirroring ``validate_tissue_probe.py`` — note fine-tuning is
*stochastic for both inits* (head init + data order depend on the seed), so JEPA is a
seed distribution too, not a single point:

1. **Multi-seed (training/init uncertainty).** Fine-tune both inits under N seeds;
   report mean ± std and the paired per-seed delta (JEPA[s] − random[s]).
2. **Bootstrap (test-set uncertainty).** Paired image-level bootstrap on the
   median-scoring seed of each init → (JEPA − random) delta CI and P(delta>0).

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

from training.train_jepa import Encoder, JepaConfig
from training.train_tissue_probe import _ROOT, _TISSUE, _prepare
# Reuse the *validated* probe's own scoring so the metric is identical to the baseline.
from training.validate_tissue_probe import _encode_split, _fit_probe, _macro_f1, _paired_delta

_NUM_CLASSES = 4  # 0 = background, 1..3 = fibrin / granulation / callus


def _load_train(cfg: JepaConfig, depth_dir, crop: bool, relief: bool):
    """Load the TrainVal split as (input tensor, patch-label) pairs for fine-tuning."""
    img_dir = _ROOT / "Images" / "TrainVal"
    ann_dir = _ROOT / "Annotations" / "TrainVal"
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
    """Encoder trunk + a linear per-patch tissue head over the 196 tokens (fine-tune only)."""

    def __init__(self, cfg: JepaConfig) -> None:
        super().__init__()
        self.encoder = Encoder(cfg)
        self.head = nn.Linear(cfg.enc_dim, _NUM_CLASSES)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        return self.head(self.encoder(images))     # (B, N, num_classes)


def _set_trainable(model: TissueModel, unfreeze_blocks: int) -> list[nn.Parameter]:
    """Freeze the encoder, re-enable the last ``unfreeze_blocks`` trunk blocks + final
    norm. Head is always trainable. Returns the trainable encoder params (for a
    smaller LR than the head)."""
    for p in model.encoder.parameters():
        p.requires_grad_(False)
    enc_trainable: list[nn.Parameter] = []
    layers = model.encoder.trunk.blocks.layers
    n = max(0, min(unfreeze_blocks, len(layers)))
    if n > 0:
        for blk in layers[-n:]:
            for p in blk.parameters():
                p.requires_grad_(True); enc_trainable.append(p)
        for p in model.encoder.trunk.norm.parameters():
            p.requires_grad_(True); enc_trainable.append(p)
    return enc_trainable


def _class_weights(items, device: str) -> torch.Tensor:
    """Balanced class weights n / (K * count_c) over classes present in train."""
    counts = torch.zeros(_NUM_CLASSES, dtype=torch.float64)
    for _, y in items:
        counts += torch.bincount(y, minlength=_NUM_CLASSES).to(torch.float64)
    present = counts > 0
    w = torch.zeros(_NUM_CLASSES, dtype=torch.float32)
    w[present] = float(counts.sum()) / (int(present.sum()) * counts[present].to(torch.float32))
    return w.to(device)


def _finetune_encoder(cfg: JepaConfig, device: str, *, init: str, weights: str | None, seed: int,
                      train_items, unfreeze_blocks: int, epochs: int, batch: int,
                      enc_lr: float, head_lr: float, weight_decay: float) -> Encoder:
    """Build a JEPA-/random-init model, fine-tune the unfrozen parts, return the
    (adapted) encoder. With ``unfreeze_blocks == 0`` nothing is trainable in the
    encoder, so training is skipped and the encoder is returned unchanged — the
    score then reproduces the frozen probe exactly (the sanity check)."""
    torch.manual_seed(seed)
    model = TissueModel(cfg).to(device)
    if init == "jepa":
        model.encoder.load_state_dict(torch.load(weights, map_location="cpu"), strict=False)
    enc_params = _set_trainable(model, unfreeze_blocks)
    if not enc_params:                       # unfreeze_blocks == 0: no adaptation
        return model.encoder.eval()

    opt = torch.optim.AdamW(
        [{"params": model.head.parameters(), "lr": head_lr},
         {"params": enc_params, "lr": enc_lr}], weight_decay=weight_decay)
    weight = _class_weights(train_items, device)
    g = torch.Generator().manual_seed(seed)
    model.train()
    for _ in range(epochs):
        order = torch.randperm(len(train_items), generator=g)
        for i in range(0, len(order), batch):
            idx = order[i:i + batch]
            x = torch.stack([train_items[j][0] for j in idx]).to(device)
            y = torch.stack([train_items[j][1] for j in idx]).to(device)
            loss = F.cross_entropy(model(x).reshape(-1, _NUM_CLASSES), y.reshape(-1), weight=weight)
            opt.zero_grad(); loss.backward(); opt.step()
    return model.encoder.eval()


def _score(encoder: Encoder, cfg: JepaConfig, device: str, depth, crop, relief, tissue_ids):
    """Score a (fine-tuned) encoder with the validated sklearn linear probe.

    Returns (macro_f1, per-image test predictions, per-image test (feat,label)) — the
    exact same instrument, splits and metric as validate_tissue_probe.
    """
    tr = _encode_split(encoder, "TrainVal", cfg, device, depth, crop, relief)
    te = _encode_split(encoder, "Test", cfg, device, depth, crop, relief)
    preds = _fit_probe(tr, te)
    return _macro_f1(preds, te, tissue_ids), preds, te


def run(args) -> None:
    device = args.device
    in_chans = 3 + (1 if args.depth else 0) + (1 if (args.relief and args.depth) else 0)
    cfg = JepaConfig(img_size=args.img_size, in_chans=in_chans)
    print(f"device={device}  in_chans={in_chans}  unfreeze_blocks={args.unfreeze_blocks}  "
          f"epochs={args.epochs}  seeds={args.seeds}  bootstrap={args.bootstrap}")
    print("scoring: validated sklearn linear probe on the fine-tuned encoder "
          "(unfreeze_blocks=0 must reproduce the frozen probe)\n")

    train_items = _load_train(cfg, args.depth, args.crop, args.relief)
    # tissue ids: from the Test labels only (encoder-independent), matching the probe.
    te0 = _encode_split(Encoder(cfg).to(device).eval(), "Test", cfg, device, args.depth, args.crop, args.relief)
    tissue_ids = [c for c in _TISSUE if c in set(np.concatenate([y for _, y in te0]))]
    print(f"train images={len(train_items)}  test images={len(te0)}  tissue_ids={tissue_ids}\n")

    ft = dict(unfreeze_blocks=args.unfreeze_blocks, epochs=args.epochs, batch=args.batch,
              enc_lr=args.enc_lr, head_lr=args.head_lr, weight_decay=args.weight_decay)

    jepa_f1, rand_f1 = [], []
    jepa_by_seed, rand_by_seed = {}, {}
    for seed in args.seeds:
        je = _finetune_encoder(cfg, device, init="jepa", weights=args.weights, seed=seed,
                               train_items=train_items, **ft)
        fj, jpreds, jte = _score(je, cfg, device, args.depth, args.crop, args.relief, tissue_ids)
        jepa_f1.append(fj); jepa_by_seed[seed] = (jpreds, jte)

        re_ = _finetune_encoder(cfg, device, init="random", weights=None, seed=seed,
                                train_items=train_items, **ft)
        fr, rpreds, rte = _score(re_, cfg, device, args.depth, args.crop, args.relief, tissue_ids)
        rand_f1.append(fr); rand_by_seed[seed] = (rpreds, rte)
        print(f"seed={seed}   JEPA-init={fj:.3f}   random-init={fr:.3f}   Δ={fj - fr:+.3f}")

    j, r = np.array(jepa_f1), np.array(rand_f1)
    jm_, js = float(j.mean()), float(j.std(ddof=1) if len(j) > 1 else 0.0)
    rm_, rs = float(r.mean()), float(r.std(ddof=1) if len(r) > 1 else 0.0)
    d = j - r
    dsm, dss = float(d.mean()), float(d.std(ddof=1) if len(j) > 1 else 0.0)
    print(f"\nJEPA-init   : mean={jm_:.3f}  std={js:.3f}  min={j.min():.3f}  max={j.max():.3f}")
    print(f"random-init : mean={rm_:.3f}  std={rs:.3f}  min={r.min():.3f}  max={r.max():.3f}")
    print(f"paired per-seed Δ (JEPA−random): mean={dsm:+.3f}  std={dss:.3f}  "
          f"n(Δ>0)={int((d > 0).sum())}/{len(d)}")

    report = {
        "config": {
            "weights": str(args.weights), "img_size": args.img_size, "depth": str(args.depth),
            "crop": args.crop, "relief": args.relief, "seeds": list(args.seeds),
            "unfreeze_blocks": args.unfreeze_blocks, "epochs": args.epochs, "batch": args.batch,
            "enc_lr": args.enc_lr, "head_lr": args.head_lr, "weight_decay": args.weight_decay,
            "bootstrap": args.bootstrap, "tissue_ids": tissue_ids,
            "scoring": "validated sklearn linear probe on fine-tuned encoder",
        },
        "jepa_seeds": {str(s): f for s, f in zip(args.seeds, jepa_f1, strict=True)},
        "random_seeds": {str(s): f for s, f in zip(args.seeds, rand_f1, strict=True)},
        "jepa_summary": {"mean": jm_, "std": js, "min": float(j.min()), "max": float(j.max())},
        "random_summary": {"mean": rm_, "std": rs, "min": float(r.min()), "max": float(r.max())},
        "paired_seed_delta": {"mean": dsm, "std": dss, "n_pos": int((d > 0).sum()), "n": len(d)},
    }

    verdict = dsm > 0 and bool((d > 0).all())
    if args.bootstrap > 0:
        jseed = args.seeds[int(np.argsort(jepa_f1)[len(jepa_f1) // 2])]
        rseed = args.seeds[int(np.argsort(rand_f1)[len(rand_f1) // 2])]
        jpreds, jte = jepa_by_seed[jseed]
        rpreds, _ = rand_by_seed[rseed]
        # _paired_delta scores both prediction sets on the same resampled images
        # (labels are identical across encoders, so jte carries the labels).
        delta = _paired_delta(jpreds, rpreds, jte, tissue_ids, n_boot=args.bootstrap, seed=args.boot_seed)
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
                    help="Trailing transformer blocks to unfreeze (0=sanity/no adaptation, 6=full).")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--enc-lr", type=float, default=1e-4, help="LR for the unfrozen encoder blocks.")
    ap.add_argument("--head-lr", type=float, default=1e-3, help="LR for the linear head.")
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--bootstrap", type=int, default=2000, help="Bootstrap resamples (0 disables).")
    ap.add_argument("--boot-seed", type=int, default=12345, help="RNG seed for the bootstrap.")
    ap.add_argument("--out", type=Path, help="Write the full report as JSON.")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    run(ap.parse_args())


if __name__ == "__main__":
    main()

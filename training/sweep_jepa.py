#!/usr/bin/env python3
"""Reproducible hyperparameter sweep for mask-guided JEPA, scored by the tissue probe.

The handoff's tuned setting (vic=4.0, lr=3e-4 -> macro-F1 0.517) came out of trying
a handful of settings by hand and keeping the best. "Best of a few hand runs" is
neither reproducible nor auditable, and — because you keep the maximum — it is
optimistically biased. This script makes the search a single, logged artifact:

* it pre-trains a JEPA encoder for every (lr, vic) pair on a grid (``train_jepa``),
* scores each encoder with the exact tissue linear-probe used elsewhere
  (``train_tissue_probe``: frozen features -> balanced logistic -> macro-F1 over the
  tissue classes), and
* writes one row per pair to CSV + JSON, sorted best-first, next to a random-init
  reference so every number is read against the no-pretraining baseline.

Everything but (lr, vic) is held fixed, including a single ``--seed`` set before each
run, so a gap between two rows is attributable to the hyperparameters, not to run
noise. The winner is still just the grid maximum — confirm it with
``validate_tissue_probe.py`` (multi-seed + bootstrap) before believing the gap.

Preprocessing flags (``--crop``/``--depth``/``--relief``) must match between the
pre-training pool and the tissue eval, exactly as in the two scripts this one wraps.

    python -m training.sweep_jepa \
        --images data/rgbd_pool/images --masks data/rgbd_pool/masks --depth data/rgbd_pool/depth \
        --crop --relief --tissue-depth data/rgbd_tissue_labeled/depth \
        --lrs 1e-3 3e-4 1e-4 --vics 0.0 1.0 4.0 --epochs 30 \
        --out runs_jepa/sweep.json --save-best runs_jepa/jepa_best.pt
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from training.train_jepa import Encoder, JepaConfig, _default_device, build_dataloader, train
from training.train_tissue_probe import _TISSUE, _build_xy


def _tissue_macro_f1(encoder, cfg: JepaConfig, device: str,
                     tissue_depth, crop: bool, relief: bool) -> float:
    """Score a (frozen) encoder with the standard tissue linear probe → macro-F1.

    Identical to ``train_tissue_probe.run``: build patch features on TrainVal/Test,
    fit a class-balanced logistic head, and take macro-F1 over the tissue classes
    present in the test set (background dominates and is excluded).
    """
    encoder.eval()
    Xtr, ytr = _build_xy(encoder, "TrainVal", cfg, device, tissue_depth, crop, relief)
    Xte, yte = _build_xy(encoder, "Test", cfg, device, tissue_depth, crop, relief)
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xte)
    tissue_ids = [c for c in _TISSUE if c in set(yte)]
    return float(f1_score(yte, pred, labels=tissue_ids, average="macro", zero_division=0))


@torch.no_grad()
def _random_reference(cfg: JepaConfig, device: str, seed: int,
                      tissue_depth, crop: bool, relief: bool) -> float:
    """Macro-F1 of an untrained (random-init) encoder — the baseline each row beats or not."""
    torch.manual_seed(seed)
    enc = Encoder(cfg).to(device)
    return _tissue_macro_f1(enc, cfg, device, tissue_depth, crop, relief)


def run(args) -> None:
    device = args.device
    td, cr, rl = args.tissue_depth, args.crop, args.relief
    if args.depth and not td:
        raise SystemExit("--depth was given for pretraining, so --tissue-depth is required "
                         "(the probe input must have the same channels as the encoder).")
    in_chans = 3 + (1 if args.depth else 0) + (1 if (rl and args.depth) else 0)

    # The dataloader does not depend on (lr, vic); build it once and reuse it.
    loader = build_dataloader(
        args.images, masks_dir=args.masks, yolo_weights=args.yolo_weights,
        img_size=args.img_size, batch_size=args.batch, workers=args.workers,
        depth_dir=args.depth, crop=cr, relief=rl,
    )
    print(f"device={device}  in_chans={in_chans}  images={len(loader.dataset)}  "
          f"grid: lrs={args.lrs} × vics={args.vics}  epochs={args.epochs}  seed={args.seed}")

    ref_cfg = JepaConfig(img_size=args.img_size, in_chans=in_chans)
    random_f1 = _random_reference(ref_cfg, device, args.seed, td, cr, rl)
    print(f"\nrandom-init reference macro-F1 = {random_f1:.3f}\n")

    rows = []
    best = None
    for lr in args.lrs:
        for vic in args.vics:
            # Fix the seed before each run so only (lr, vic) varies between rows.
            torch.manual_seed(args.seed)
            np.random.seed(args.seed)
            cfg = JepaConfig(img_size=args.img_size, in_chans=in_chans, vic_weight=vic)
            print(f"--- training lr={lr:g} vic={vic:g} ---")
            model = train(cfg, loader, epochs=args.epochs, lr=lr, device=device)
            f1 = _tissue_macro_f1(model.encoder, cfg, device, td, cr, rl)
            delta = f1 - random_f1
            rows.append({"lr": lr, "vic": vic, "macro_f1": round(f1, 4),
                         "delta_vs_random": round(delta, 4)})
            print(f"    macro-F1={f1:.3f}  (Δ vs random {delta:+.3f})\n")
            if best is None or f1 > best[0]:
                best = (f1, lr, vic, model.encoder.state_dict())

    rows.sort(key=lambda r: r["macro_f1"], reverse=True)
    print("=== sweep results (best first) ===")
    print(f"{'lr':>8} {'vic':>6} {'macroF1':>8} {'Δrandom':>8}")
    for r in rows:
        print(f"{r['lr']:>8g} {r['vic']:>6g} {r['macro_f1']:>8.3f} {r['delta_vs_random']:>+8.3f}")
    top = rows[0]
    print(f"\nbest: lr={top['lr']:g} vic={top['vic']:g} → macro-F1={top['macro_f1']:.3f} "
          f"(random {random_f1:.3f}). Confirm with validate_tissue_probe.py before trusting it.")

    report = {
        "config": {
            "images": str(args.images), "img_size": args.img_size, "in_chans": in_chans,
            "crop": cr, "depth": str(args.depth), "relief": rl,
            "epochs": args.epochs, "batch": args.batch, "seed": args.seed,
            "lrs": args.lrs, "vics": args.vics,
        },
        "random_reference_macro_f1": round(random_f1, 4),
        "results": rows,
        "best": {k: top[k] for k in ("lr", "vic", "macro_f1", "delta_vs_random")},
    }
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        csv_path = args.out.with_suffix(".csv")
        with csv_path.open("w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=["lr", "vic", "macro_f1", "delta_vs_random"])
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.out} and {csv_path}")

    if args.save_best and best is not None:
        args.save_best.parent.mkdir(parents=True, exist_ok=True)
        torch.save(best[3], args.save_best)
        print(f"saved best encoder (lr={best[1]:g} vic={best[2]:g}) -> {args.save_best}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Reproducible JEPA (lr, vic) sweep via the probe.")
    # Pre-training data + input (mirror train_jepa).
    ap.add_argument("--images", type=Path, required=True, help="Unlabeled wound images.")
    ap.add_argument("--masks", type=Path, help="Precomputed wound-mask PNG dir (stem-matched).")
    ap.add_argument("--yolo-weights", help="YOLO26-seg weights for live masks (if no --masks).")
    ap.add_argument("--depth", type=Path, help="Depth-PNG dir for the pretraining pool (2.5D).")
    ap.add_argument("--crop", action="store_true", help="Crop to the wound box (match the probe).")
    ap.add_argument("--relief", action="store_true", help="Add the relief channel (needs --depth).")
    # Tissue-probe eval data (DFUTissue lives at a fixed _ROOT; only its depth dir varies).
    ap.add_argument("--tissue-depth", type=Path, help="Depth dir for DFUTissue (if --depth).")
    # The grid.
    ap.add_argument("--lrs", type=float, nargs="+", default=[1e-3, 3e-4, 1e-4])
    ap.add_argument("--vics", type=float, nargs="+", default=[0.0, 1.0, 4.0])
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--workers", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0, help="Fixed each run so only (lr, vic) varies.")
    ap.add_argument("--out", type=Path, help="Write the report as JSON (a .csv is written too).")
    ap.add_argument("--save-best", type=Path, help="Save the best encoder's weights here.")
    ap.add_argument("--device", default=_default_device())
    run(ap.parse_args())


if __name__ == "__main__":
    main()

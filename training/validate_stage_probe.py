#!/usr/bin/env python3
"""Pressure-injury *stage* probe on a frozen JEPA encoder — multi-seed + bootstrap.

Until now every JEPA-vs-random verdict came from the DFUTissue tissue probe: 16
**foot-ulcer** test images. Sorbed stages **pressure** injuries, and the pool
work showed domain match dominates (a pressure-heavy pool dropped the foot probe
0.545 -> 0.464). This script moves the ruler into the target domain: whole-image
NPIAP stage 1-4 on a leakage-checked pressure test split (built from roboflow
fr7kn with its own train/test near-duplicates removed, see HANDOFF).

Protocol — deliberately the same shape as ``validate_tissue_probe.py`` so the two
verdicts are comparable:

* Each image is encoded by a **frozen** encoder; its patch tokens are pooled into
  one vector (``--pool``). A standardised, class-balanced logistic regression is
  fit on the train split and scored on the test split.
* **JEPA** weights are fixed, so its score is one deterministic point. **Random**
  init is a distribution over ``--seeds``; JEPA's z-score is measured against it.
* **Bootstrap** resamples test *images* for a 95% CI, and a *paired* bootstrap
  against the median random seed gives the (JEPA - random) delta and P(delta > 0).
* Metrics: macro-F1 (primary, as in the tissue probe) and quadratic-weighted kappa
  (QWK), which respects stage order — calling a stage 1 a stage 4 costs more than
  calling it a stage 2.

Pooling matters. JEPA's patch tokens within an image are nearly orthogonal (mean
cosine 0.04-0.12 vs 0.84 for a random encoder), so a plain mean cancels much of
what they carry. ``mean`` is the pre-fixed primary protocol; ``meanstd``
(concatenated token mean and std) keeps the token spread and was chosen, on
development checkpoints only, as the secondary protocol because it gives the
*random* baseline its strongest score. ``max`` is kept for completeness.

Preprocessing must match pre-training exactly: the input is built the same way as
``train_jepa.WoundJepaDataset`` (padded wound-box crop from a precomputed YOLO mask,
ImageNet-normalised RGB, standardised depth, optional relief). An image whose mask
is missing or empty is used uncropped — again exactly as in pre-training.

    python -m training.validate_stage_probe --weights runs_jepa/jepa_pressure.pt \
        --data data/ruler_stage_fr7kn --masks data/ruler_stage_fr7kn/masks \
        --depth data/ruler_stage_fr7kn/depth --crop --relief \
        --seeds 0 1 2 3 4 --bootstrap 2000 --out runs_jepa/stage_validation.json

``--data`` holds ``train/<stage>/`` and ``test/<stage>/`` folders (stage = 1..4).
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import cohen_kappa_score, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from training.train_jepa import (
    _IMAGENET_MEAN,
    _IMAGENET_STD,
    Encoder,
    JepaConfig,
    _relief_map,
    _standardize,
    wound_box,
)

_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
_RELIEF_K = 15  # WoundJepaDataset default
_POOLS = {
    "mean": lambda t: t.mean(dim=0),
    "meanstd": lambda t: torch.cat([t.mean(dim=0), t.std(dim=0)]),
    "max": lambda t: t.max(dim=0).values,
}


def list_split(root: Path, split: str) -> list[tuple[Path, int]]:
    """``(image, stage)`` pairs from ``<root>/<split>/<stage>/``; stage dirs are ints."""
    items: list[tuple[Path, int]] = []
    split_dir = root / split
    if not split_dir.is_dir():
        raise FileNotFoundError(f"no {split}/ under {root}")
    for stage_dir in sorted(split_dir.iterdir()):
        if not stage_dir.is_dir() or not stage_dir.name.isdigit():
            continue
        for img in sorted(stage_dir.iterdir()):
            if img.suffix.lower() in _EXTS:
                items.append((img, int(stage_dir.name)))
    if not items:
        raise FileNotFoundError(f"no staged images under {split_dir}")
    return items


def prepare_image(path: Path, cfg: JepaConfig, masks_dir: Path | None, depth_dir: Path | None,
                  crop: bool, relief: bool) -> tuple[torch.Tensor, bool]:
    """Build the encoder input exactly like ``WoundJepaDataset.__getitem__``.

    Returns ``(tensor, cropped)``; ``cropped`` is False when no usable mask existed,
    in which case the full frame is used (the pre-training fallback).
    """
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"could not read image {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    h, w = rgb.shape[:2]

    mask = None
    if masks_dir is not None:
        m = cv2.imread(str(masks_dir / f"{path.stem}.png"), cv2.IMREAD_GRAYSCALE)
        if m is not None:
            if m.shape != (h, w):
                m = cv2.resize(m, (w, h), interpolation=cv2.INTER_NEAREST)
            mask = (m > 0).astype(np.uint8)

    depth = None
    if depth_dir is not None:
        d = cv2.imread(str(depth_dir / f"{path.stem}.png"), cv2.IMREAD_GRAYSCALE)
        if d is None:
            d = np.zeros((h, w), np.uint8)
        elif d.shape != (h, w):
            d = cv2.resize(d, (w, h), interpolation=cv2.INTER_LINEAR)
        depth = d

    cropped = False
    if crop and mask is not None:
        box = wound_box(mask)
        if box is not None:
            x0, y0, x1, y1 = box
            rgb = rgb[y0:y1, x0:x1]
            depth = None if depth is None else depth[y0:y1, x0:x1]
            cropped = True

    s = cfg.img_size
    rgb = cv2.resize(rgb, (s, s), interpolation=cv2.INTER_LINEAR)
    img = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
    img = (img - _IMAGENET_MEAN) / _IMAGENET_STD
    if depth is not None:
        dd = cv2.resize(depth, (s, s), interpolation=cv2.INTER_LINEAR).astype(np.float32)
        extra = [_standardize(torch.from_numpy(dd).unsqueeze(0) / 255.0)]
        if relief:
            extra.append(_standardize(_relief_map(dd, _RELIEF_K).unsqueeze(0)))
        img = torch.cat([img, *extra], dim=0)
    return img, cropped


def build_encoder(cfg: JepaConfig, device: str, *, init: str, weights: str | None,
                  seed: int) -> Encoder:
    """Frozen encoder: JEPA weights (verified to actually load) or random at ``seed``."""
    torch.manual_seed(seed)
    encoder = Encoder(cfg)
    if init == "jepa":
        state = torch.load(weights, map_location="cpu")
        result = encoder.load_state_dict(state, strict=False)
        # strict=False would otherwise let a key-name mismatch load *nothing* and
        # silently turn "JEPA" into a random encoder. Buffers aside, demand a full load.
        missing = [k for k in result.missing_keys if k != "pos_embed"]
        if missing:
            raise SystemExit(f"{weights}: {len(missing)} encoder weights missing "
                             f"(e.g. {missing[:3]}) — wrong checkpoint or in_chans")
    return encoder.to(device).eval()


@torch.no_grad()
def encode(encoder: Encoder, items, cfg: JepaConfig, device: str, masks_dir, depth_dir,
           crop: bool, relief: bool, pool: str = "mean") -> tuple[np.ndarray, np.ndarray, float]:
    """Pooled patch features per image, their stages, and the cropped fraction."""
    pool_fn = _POOLS[pool]
    feats, stages, n_crop = [], [], 0
    for path, stage in items:
        t, cropped = prepare_image(path, cfg, masks_dir, depth_dir, crop, relief)
        tokens = encoder(t.unsqueeze(0).to(device))[0]  # (num_patches, dim)
        feats.append(pool_fn(tokens).cpu().numpy())
        stages.append(stage)
        n_crop += int(cropped)
    return np.stack(feats), np.asarray(stages), n_crop / max(len(items), 1)


def fit_predict(Xtr, ytr, Xte) -> np.ndarray:
    """Standardised, class-balanced logistic probe (scaler fit on train only)."""
    clf = make_pipeline(StandardScaler(),
                        LogisticRegression(max_iter=3000, class_weight="balanced", C=1.0))
    clf.fit(Xtr, ytr)
    return clf.predict(Xte)


def scores(y_true, y_pred, labels) -> dict[str, float]:
    return {
        "macro_f1": float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        "qwk": float(cohen_kappa_score(y_true, y_pred, labels=labels, weights="quadratic")),
    }


def bootstrap_ci(y_true, y_pred, labels, *, n_boot: int, seed: int) -> dict[str, dict[str, float]]:
    """Percentile 95% CI of each metric by resampling test images with replacement."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    draws = {"macro_f1": np.empty(n_boot), "qwk": np.empty(n_boot)}
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        s = scores(y_true[idx], y_pred[idx], labels)
        for k in draws:
            draws[k][b] = s[k]
    out = {}
    for k, v in draws.items():
        v = np.nan_to_num(v)  # QWK is undefined on a degenerate resample
        lo, hi = np.percentile(v, [2.5, 97.5])
        out[k] = {"mean": float(v.mean()), "lo": float(lo), "hi": float(hi)}
    return out


def paired_delta(y_true, pred_a, pred_b, labels, *, n_boot: int, seed: int) -> dict[str, dict[str, float]]:
    """Paired bootstrap of (A - B) per metric on the same resampled images."""
    rng = np.random.default_rng(seed)
    n = len(y_true)
    deltas = {"macro_f1": np.empty(n_boot), "qwk": np.empty(n_boot)}
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sa = scores(y_true[idx], pred_a[idx], labels)
        sb = scores(y_true[idx], pred_b[idx], labels)
        for k in deltas:
            deltas[k][b] = sa[k] - sb[k]
    out = {}
    for k, v in deltas.items():
        v = np.nan_to_num(v)
        lo, hi = np.percentile(v, [2.5, 97.5])
        out[k] = {"mean": float(v.mean()), "lo": float(lo), "hi": float(hi),
                  "p_gt0": float(np.mean(v > 0))}
    return out


def run(args) -> dict:
    device = args.device
    in_chans = 3 + (1 if args.depth else 0) + (1 if (args.relief and args.depth) else 0)
    cfg = JepaConfig(img_size=args.img_size, in_chans=in_chans)
    masks_dir = Path(args.masks) if args.masks else None
    depth_dir = Path(args.depth) if args.depth else None
    if args.crop and masks_dir is None:
        raise SystemExit("--crop needs --masks (the precomputed wound masks used for the box)")

    train = list_split(args.data, "train")
    test = list_split(args.data, "test")
    labels = sorted({s for _, s in train} | {s for _, s in test})
    print(f"device={device} in_chans={in_chans} pool={args.pool} train={len(train)} "
          f"test={len(test)} stages={labels} seeds={args.seeds} bootstrap={args.bootstrap}")

    def evaluate(init: str, seed: int):
        enc = build_encoder(cfg, device, init=init, weights=args.weights, seed=seed)
        Xtr, ytr, ctr = encode(enc, train, cfg, device, masks_dir, depth_dir,
                               args.crop, args.relief, args.pool)
        Xte, yte, cte = encode(enc, test, cfg, device, masks_dir, depth_dir,
                               args.crop, args.relief, args.pool)
        pred = fit_predict(Xtr, ytr, Xte)
        return pred, yte, scores(yte, pred, labels), (ctr, cte)

    jepa_pred, y_test, jepa_s, (ctr, cte) = evaluate("jepa", 0)
    print(f"cropped fraction  train={ctr:.2%}  test={cte:.2%}")
    print(f"\nJEPA   macro-F1={jepa_s['macro_f1']:.3f}  QWK={jepa_s['qwk']:.3f}  (deterministic)")

    rand = {}
    for seed in args.seeds:
        pred, _, s, _ = evaluate("random", seed)
        rand[seed] = (pred, s)
        print(f"random seed={seed}  macro-F1={s['macro_f1']:.3f}  QWK={s['qwk']:.3f}")

    report: dict = {
        "config": {"weights": str(args.weights), "data": str(args.data), "masks": str(masks_dir),
                   "depth": str(depth_dir), "crop": args.crop, "relief": args.relief,
                   "pool": args.pool, "img_size": args.img_size, "seeds": list(args.seeds),
                   "bootstrap": args.bootstrap, "n_train": len(train), "n_test": len(test),
                   "cropped_fraction": {"train": ctr, "test": cte}},
        "jepa": jepa_s,
        "random_seeds": {str(k): v[1] for k, v in rand.items()},
    }
    summary = {}
    for metric in ("macro_f1", "qwk"):
        vals = np.array([v[1][metric] for v in rand.values()])
        mean, std = float(vals.mean()), float(vals.std(ddof=1) if len(vals) > 1 else 0.0)
        z = (jepa_s[metric] - mean) / std if std > 0 else float("inf")
        summary[metric] = {"mean": mean, "std": std, "min": float(vals.min()),
                           "max": float(vals.max()), "z_jepa": z}
        print(f"\n{metric}: random mean={mean:.3f} std={std:.3f}  "
              f"JEPA {jepa_s[metric] - mean:+.3f} (z≈{z:.2f})")
    report["random_summary"] = summary

    if args.bootstrap > 0:
        nb, bs = args.bootstrap, args.boot_seed
        order = sorted(rand, key=lambda k: rand[k][1]["macro_f1"])
        med_seed = order[len(order) // 2]
        jepa_ci = bootstrap_ci(y_test, jepa_pred, labels, n_boot=nb, seed=bs)
        delta = paired_delta(y_test, jepa_pred, rand[med_seed][0], labels, n_boot=nb, seed=bs)
        report["bootstrap"] = {"jepa_ci95": jepa_ci, "paired_delta_vs_seed": med_seed,
                               "paired_delta_ci95": delta}
        print(f"\nbootstrap 95% CI (B={nb}, image-level), paired vs random seed {med_seed}")
        for metric in ("macro_f1", "qwk"):
            c, d = jepa_ci[metric], delta[metric]
            print(f"  {metric:8s} JEPA {c['mean']:.3f} [{c['lo']:.3f}, {c['hi']:.3f}]   "
                  f"Δ {d['mean']:+.3f} [{d['lo']:+.3f}, {d['hi']:+.3f}]  P(Δ>0)={d['p_gt0']:.3f}")
        verdict = summary["macro_f1"]["z_jepa"] > 1.0 and delta["macro_f1"]["lo"] > 0.0
        report["verdict_supported"] = bool(verdict)
        print(f"\nVERDICT (macro-F1): JEPA > random is "
              f"{'SUPPORTED' if verdict else 'NOT established'} (needs z>1 and paired Δ CI above 0).")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.out}")
    return report


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-seed + bootstrap stage probe on a frozen JEPA encoder.")
    ap.add_argument("--weights", required=True, help="JEPA encoder weights.")
    ap.add_argument("--data", type=Path, required=True, help="Root with train/<stage>/ and test/<stage>/.")
    ap.add_argument("--masks", type=Path, help="Precomputed wound-mask dir (stem-matched), for --crop.")
    ap.add_argument("--depth", type=Path, help="Depth-PNG dir, 2.5D input (match pre-training).")
    ap.add_argument("--crop", action="store_true", help="Crop to the padded wound box (match pre-training).")
    ap.add_argument("--relief", action="store_true", help="Add the relief channel (match pre-training).")
    ap.add_argument("--pool", choices=sorted(_POOLS), default="mean",
                    help="Token pooling: mean (primary), meanstd (secondary), max.")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--boot-seed", type=int, default=12345)
    ap.add_argument("--out", type=Path, help="Write the full report as JSON.")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    run(ap.parse_args())


if __name__ == "__main__":
    main()

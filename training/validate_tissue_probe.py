#!/usr/bin/env python3
"""Statistically validate the tissue-probe result: is JEPA really above random?

``train_tissue_probe.py`` reports a single macro-F1 for one ``--init`` at a time.
The tuned setting (vic=4.0, lr=3e-4) hit macro-F1 0.517 and first beat the random
average (0.508) — but a single number cannot tell whether that gap is real or just
the best of several tries. This script answers that with two independent
uncertainty axes, exactly the two the handoff calls for:

1. **Multi-seed (training / init uncertainty).** The random-init encoder depends on
   the torch seed, so its probe score is a *distribution*, not a point. We train it
   under N seeds and summarise mean ± std. The JEPA encoder is loaded from fixed
   weights, so its features — and therefore the lbfgs LogisticRegression fit on them
   — are deterministic: JEPA is a single point that we ask to clear the random cloud.

2. **Bootstrap (test-set uncertainty).** The test split is ~16 images; a macro-F1 on
   so few images is itself noisy. We resample the test **images** (not patches —
   patches within one image are correlated, which would fake a tight interval) with
   replacement B times and take the 2.5/97.5 percentiles as a 95% CI. A *paired*
   bootstrap on the same resampled images yields the (JEPA − random) delta
   distribution; the fraction of resamples with delta > 0 is a bootstrap analogue of
   a one-sided p-value.

Together: axis 1 says whether JEPA beats a typical random init; axis 2 says whether
the small test set can even support that claim. Only when JEPA clears the seed cloud
*and* the paired delta CI stays above 0 is the gap defensible.

Preprocessing (``--depth``/``--crop``/``--relief``/``--img-size``) must match how the
JEPA encoder was pre-trained, and is applied identically to the random baseline —
same flags as ``train_tissue_probe.py``.

    python -m training.validate_tissue_probe \
        --weights runs_jepa/jepa_best.pt --crop --depth data/rgbd_tissue_labeled/depth \
        --relief --seeds 0 1 2 3 4 --bootstrap 2000 --out runs_jepa/tissue_validation.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score

from training.train_jepa import Encoder, JepaConfig
from training.train_tissue_probe import _ROOT, _TISSUE, _prepare


@torch.no_grad()
def _encode_split(encoder: Encoder, split: str, cfg: JepaConfig, device: str,
                  depth_dir, crop: bool, relief: bool):
    """Encode every image in ``split`` and keep the patches grouped per image.

    Returns a list of ``(feat, label)`` arrays, one per image. Grouping is what
    lets the bootstrap resample at the image level; a flat concat is recovered
    with ``np.concatenate`` when a single design matrix is needed.
    """
    img_dir = _ROOT / "Images" / split
    ann_dir = _ROOT / "Annotations" / split
    per_image = []
    for img_path in sorted(img_dir.glob("*.png")):
        ann = ann_dir / img_path.name
        if not ann.exists():
            continue
        t, y = _prepare(img_path, ann, cfg, depth_dir, crop, relief)
        feat = encoder(t.unsqueeze(0).to(device))[0].cpu().numpy()
        per_image.append((feat, y))
    if not per_image:
        raise FileNotFoundError(f"no annotated images under {img_dir}")
    return per_image


def _build_encoder(cfg: JepaConfig, device: str, *, init: str, weights: str | None,
                   seed: int) -> Encoder:
    """A frozen encoder: JEPA-pretrained (deterministic) or random at ``seed``."""
    torch.manual_seed(seed)
    encoder = Encoder(cfg).to(device).eval()
    if init == "jepa":
        state = torch.load(weights, map_location="cpu")
        encoder.load_state_dict(state, strict=False)
    return encoder


def _fit_probe(train_imgs, test_imgs):
    """Fit the balanced logistic probe on train patches; predict test patches.

    Returns per-image test predictions (a list of arrays, aligned with
    ``test_imgs``) so the bootstrap can resample whole images.
    """
    Xtr = np.concatenate([f for f, _ in train_imgs])
    ytr = np.concatenate([y for _, y in train_imgs])
    clf = LogisticRegression(max_iter=2000, class_weight="balanced", C=1.0)
    clf.fit(Xtr, ytr)
    return [clf.predict(f) for f, _ in test_imgs]


def _macro_f1(pred_imgs, test_imgs, tissue_ids) -> float:
    """Macro-F1 over the fixed tissue classes, pooled across the given images."""
    y_true = np.concatenate([y for _, y in test_imgs])
    y_pred = np.concatenate(pred_imgs)
    return float(f1_score(y_true, y_pred, labels=tissue_ids, average="macro", zero_division=0))


def _bootstrap_ci(pred_imgs, test_imgs, tissue_ids, *, n_boot: int, seed: int):
    """Percentile 95% CI of macro-F1 by resampling images with replacement.

    ``tissue_ids`` is fixed from the full test set so every resample scores the
    same class set (a resample that happens to drop a rare class still reports F1
    for it, as 0 — which is the honest behaviour on a tiny test set).
    """
    rng = np.random.default_rng(seed)
    n = len(test_imgs)
    scores = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        scores[b] = _macro_f1([pred_imgs[i] for i in idx], [test_imgs[i] for i in idx], tissue_ids)
    lo, hi = np.percentile(scores, [2.5, 97.5])
    return {"mean": float(scores.mean()), "lo": float(lo), "hi": float(hi)}


def _paired_delta(jepa_preds, rand_preds, test_imgs, tissue_ids, *, n_boot: int, seed: int):
    """Paired bootstrap of (JEPA − random) macro-F1 on the same resampled images.

    Pairing removes the shared test-set noise, so the delta CI is tighter and
    directly answers "did JEPA beat this random init". ``p_gt0`` is the fraction of
    resamples with a positive delta — a bootstrap one-sided p-value is ``1 - p_gt0``.
    """
    rng = np.random.default_rng(seed)
    n = len(test_imgs)
    deltas = np.empty(n_boot, dtype=np.float64)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        imgs = [test_imgs[i] for i in idx]
        fj = _macro_f1([jepa_preds[i] for i in idx], imgs, tissue_ids)
        fr = _macro_f1([rand_preds[i] for i in idx], imgs, tissue_ids)
        deltas[b] = fj - fr
    lo, hi = np.percentile(deltas, [2.5, 97.5])
    return {
        "mean": float(deltas.mean()),
        "lo": float(lo),
        "hi": float(hi),
        "p_gt0": float(np.mean(deltas > 0)),
    }


def run(args) -> None:
    device = args.device
    in_chans = 3 + (1 if args.depth else 0) + (1 if (args.relief and args.depth) else 0)
    cfg = JepaConfig(img_size=args.img_size, in_chans=in_chans)
    print(f"device={device}  in_chans={in_chans}  seeds={args.seeds}  bootstrap={args.bootstrap}")

    # --- JEPA: deterministic. Encode once, fit once, keep per-image predictions. ---
    jepa_enc = _build_encoder(cfg, device, init="jepa", weights=args.weights, seed=0)
    tr_j = _encode_split(jepa_enc, "TrainVal", cfg, device, args.depth, args.crop, args.relief)
    te_j = _encode_split(jepa_enc, "Test", cfg, device, args.depth, args.crop, args.relief)
    tissue_ids = [c for c in _TISSUE if c in set(np.concatenate([y for _, y in te_j]))]
    jepa_preds = _fit_probe(tr_j, te_j)
    jepa_f1 = _macro_f1(jepa_preds, te_j, tissue_ids)
    print(f"\nJEPA  macro-F1 = {jepa_f1:.3f}  (deterministic; loaded {args.weights})")

    # --- Random baseline: a distribution over seeds (training/init uncertainty). ---
    rand_scores, rand_preds_by_seed = [], {}
    for seed in args.seeds:
        enc = _build_encoder(cfg, device, init="random", weights=None, seed=seed)
        tr_r = _encode_split(enc, "TrainVal", cfg, device, args.depth, args.crop, args.relief)
        te_r = _encode_split(enc, "Test", cfg, device, args.depth, args.crop, args.relief)
        preds = _fit_probe(tr_r, te_r)
        f1 = _macro_f1(preds, te_r, tissue_ids)
        rand_scores.append(f1)
        rand_preds_by_seed[seed] = (preds, te_r)
        print(f"random seed={seed}  macro-F1 = {f1:.3f}")

    rand = np.array(rand_scores, dtype=np.float64)
    r_mean, r_std = float(rand.mean()), float(rand.std(ddof=1) if len(rand) > 1 else 0.0)
    # z of the JEPA point against the random seed cloud (how many std above typical).
    z = (jepa_f1 - r_mean) / r_std if r_std > 0 else float("inf")
    print(f"\nrandom over {len(rand)} seeds: mean={r_mean:.3f}  std={r_std:.3f}  "
          f"min={rand.min():.3f}  max={rand.max():.3f}")
    print(f"JEPA is {jepa_f1 - r_mean:+.3f} vs random mean  (z ≈ {z:.2f})")

    # --- Bootstrap (test-set uncertainty), incl. paired delta vs the median random. ---
    report = {
        "config": {
            "weights": str(args.weights), "img_size": args.img_size, "depth": str(args.depth),
            "crop": args.crop, "relief": args.relief, "seeds": list(args.seeds),
            "bootstrap": args.bootstrap, "tissue_ids": tissue_ids,
        },
        "jepa_f1": jepa_f1,
        "random_seeds": {str(s): sc for s, sc in zip(args.seeds, rand_scores, strict=True)},
        "random_summary": {"mean": r_mean, "std": r_std,
                           "min": float(rand.min()), "max": float(rand.max()), "z_jepa": z},
    }
    if args.bootstrap > 0:
        nb, bs = args.bootstrap, args.boot_seed
        jepa_ci = _bootstrap_ci(jepa_preds, te_j, tissue_ids, n_boot=nb, seed=bs)
        # Compare against the median-scoring random seed: a representative baseline.
        med_seed = args.seeds[int(np.argsort(rand_scores)[len(rand_scores) // 2])]
        rand_preds, rand_te = rand_preds_by_seed[med_seed]
        rand_ci = _bootstrap_ci(rand_preds, rand_te, tissue_ids, n_boot=nb, seed=bs)
        delta = _paired_delta(jepa_preds, rand_preds, te_j, tissue_ids, n_boot=nb, seed=bs)
        report["bootstrap"] = {
            "jepa_ci95": jepa_ci,
            "random_ci95": {"seed": med_seed, **rand_ci},
            "paired_delta_ci95": {"vs_seed": med_seed, **delta},
        }
        def _fmt(d):
            return f"{d['mean']:+.3f}  [{d['lo']:+.3f}, {d['hi']:+.3f}]"

        print(f"\nbootstrap 95% CI  (B={nb}, image-level resampling)")
        print(f"  JEPA        : {_fmt(jepa_ci)}")
        print(f"  rand s={med_seed}    : {_fmt(rand_ci)}")
        print(f"  paired Δ    : {_fmt(delta)}  P(Δ>0)={delta['p_gt0']:.3f}")
        verdict = (z > 1.0) and (delta["lo"] > 0.0)
        print(f"\nVERDICT: JEPA > random is {'SUPPORTED' if verdict else 'NOT established'} "
              f"(needs z>1 and paired Δ CI above 0).")
        report["verdict_supported"] = bool(verdict)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2))
        print(f"\nwrote {args.out}")


def main() -> None:
    ap = argparse.ArgumentParser(description="Multi-seed + bootstrap validation of the probe.")
    ap.add_argument("--weights", default="runs_jepa/jepa_best.pt", help="JEPA encoder weights.")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--depth", type=Path, help="Depth-PNG dir, 2.5D input (match pretraining).")
    ap.add_argument("--crop", action="store_true", help="Crop to the wound box (match pretrain).")
    ap.add_argument("--relief", action="store_true", help="Add relief channel (match pretraining).")
    ap.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4],
                    help="Seeds for the random-init baseline distribution.")
    ap.add_argument("--bootstrap", type=int, default=2000, help="Bootstrap resamples (0 disables).")
    ap.add_argument("--boot-seed", type=int, default=12345, help="RNG seed for the bootstrap.")
    ap.add_argument("--out", type=Path, help="Write the full report as JSON.")
    ap.add_argument("--device", default="mps" if torch.backends.mps.is_available() else "cpu")
    run(ap.parse_args())


if __name__ == "__main__":
    main()

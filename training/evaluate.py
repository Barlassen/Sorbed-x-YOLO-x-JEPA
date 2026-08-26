#!/usr/bin/env python3
"""Publication-grade evaluation metrics for wound segmentation and stage grading.

This is the shared metrics harness referenced by ``training/EVALUATE.md``. It has
two independent sections, selected by subcommand:

* ``seg`` — segmentation: **Dice**, **IoU**, and boundary quality **HD95**/**ASSD**
  between predicted masks and ground-truth masks paired by filename stem, with
  bootstrap confidence intervals over images.
* ``grade`` — stage grading: **balanced accuracy**, **quadratic-weighted Cohen's
  κ** (the ordinal primary), **per-stage sensitivity**, all with patient-clustered
  bootstrap CIs, plus **calibration** (ECE, Brier, and a reliability table) when
  per-class probabilities are supplied.

Dependency-light on purpose: only ``numpy`` and ``scipy`` (both Sorbed core deps).
No torch, no plotting requirement — the reliability table is emitted as data so any
downstream tool can render it. Nothing here fabricates values; empty-vs-empty masks
score as perfect, one-sided-empty boundary distances are reported as ``NaN`` and
excluded from the boundary means (never silently zeroed).

HD95/ASSD are in **pixels**. Convert to millimetres only with a real per-image
scale from a fiducial — this harness does not guess physical scale.
"""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage

IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"})

# Ordinal-then-side NPIAP class order, mirroring
# ``sorbed.domain.enums.PressureInjuryStage`` / ``training.data.GRADE_CLASSES``.
# Hard-coded here so this module stays importable without pulling in torch.
DEFAULT_GRADE_CLASSES: tuple[str, ...] = (
    "stage_1",
    "stage_2",
    "stage_3",
    "stage_4",
    "unstageable",
    "deep_tissue_injury",
)


# --------------------------------------------------------------------------- #
# Segmentation metrics
# --------------------------------------------------------------------------- #
def dice_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Dice overlap of two boolean masks (empty-vs-empty is 1.0)."""
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    denom = float(pred_b.sum() + gt_b.sum())
    if denom == 0.0:
        return 1.0
    return 2.0 * float(np.logical_and(pred_b, gt_b).sum()) / denom


def iou_score(pred: np.ndarray, gt: np.ndarray) -> float:
    """Intersection-over-union of two boolean masks (empty-vs-empty is 1.0)."""
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    union = float(np.logical_or(pred_b, gt_b).sum())
    if union == 0.0:
        return 1.0
    return float(np.logical_and(pred_b, gt_b).sum()) / union


def _surface_distances(pred: np.ndarray, gt: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Symmetric boundary distances (pixels) between two boolean masks.

    Returns ``(d_pred_to_gt, d_gt_to_pred)``: for every predicted-boundary pixel
    its distance to the nearest GT-boundary pixel, and vice versa. Empty when a
    boundary does not exist.
    """
    pred_b = pred.astype(bool)
    gt_b = gt.astype(bool)
    pred_border = pred_b ^ ndimage.binary_erosion(pred_b)
    gt_border = gt_b ^ ndimage.binary_erosion(gt_b)
    if not pred_border.any() or not gt_border.any():
        return np.array([]), np.array([])
    dt_to_gt = ndimage.distance_transform_edt(~gt_border)
    dt_to_pred = ndimage.distance_transform_edt(~pred_border)
    return dt_to_gt[pred_border], dt_to_pred[gt_border]


def hd95(pred: np.ndarray, gt: np.ndarray) -> float:
    """95th-percentile symmetric Hausdorff distance in pixels.

    ``0.0`` when both masks are empty; ``NaN`` when exactly one is empty (the
    distance is undefined and must not be counted as a good or bad score).
    """
    pred_empty = not pred.any()
    gt_empty = not gt.any()
    if pred_empty and gt_empty:
        return 0.0
    if pred_empty or gt_empty:
        return float("nan")
    d_pg, d_gp = _surface_distances(pred, gt)
    if d_pg.size == 0 or d_gp.size == 0:
        return float("nan")
    return float(max(np.percentile(d_pg, 95), np.percentile(d_gp, 95)))


def assd(pred: np.ndarray, gt: np.ndarray) -> float:
    """Average symmetric surface distance in pixels (same edge cases as HD95)."""
    pred_empty = not pred.any()
    gt_empty = not gt.any()
    if pred_empty and gt_empty:
        return 0.0
    if pred_empty or gt_empty:
        return float("nan")
    d_pg, d_gp = _surface_distances(pred, gt)
    if d_pg.size == 0 or d_gp.size == 0:
        return float("nan")
    return float(np.concatenate([d_pg, d_gp]).mean())


def _read_mask(path: Path) -> np.ndarray:
    import cv2

    raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise RuntimeError(f"could not read mask {path}")
    return raw > 0


def _index_by_stem(directory: Path) -> dict[str, Path]:
    return {
        p.stem: p
        for p in sorted(directory.iterdir())
        if p.is_file() and p.suffix.lower() in IMAGE_SUFFIXES
    }


def _mean_ignoring_nan(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    return float(finite.mean()) if finite.size else float("nan")


def _bootstrap_ci(
    values: Sequence[float],
    *,
    n_boot: int,
    alpha: float,
    seed: int,
) -> tuple[float, float]:
    """Percentile CI of the mean of ``values`` (ignoring NaNs) by resampling."""
    arr = np.asarray(values, dtype=float)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=float)
    n = finite.size
    for b in range(n_boot):
        sample = finite[rng.integers(0, n, size=n)]
        means[b] = sample.mean()
    lo = float(np.percentile(means, 100 * alpha / 2))
    hi = float(np.percentile(means, 100 * (1 - alpha / 2)))
    return lo, hi


def evaluate_segmentation(
    pred_dir: Path,
    gt_dir: Path,
    *,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> dict[str, Any]:
    """Score every predicted mask against its GT counterpart (paired by stem)."""
    preds = _index_by_stem(pred_dir)
    gts = _index_by_stem(gt_dir)
    stems = sorted(set(preds) & set(gts))
    if not stems:
        raise SystemExit(f"no common mask stems between {pred_dir} and {gt_dir}")

    dices: list[float] = []
    ious: list[float] = []
    hd95s: list[float] = []
    assds: list[float] = []
    for stem in stems:
        pred = _read_mask(preds[stem])
        gt = _read_mask(gts[stem])
        if pred.shape != gt.shape:
            import cv2

            pred = (
                cv2.resize(pred.astype(np.uint8), (gt.shape[1], gt.shape[0]),
                           interpolation=cv2.INTER_NEAREST) > 0
            )
        dices.append(dice_score(pred, gt))
        ious.append(iou_score(pred, gt))
        hd95s.append(hd95(pred, gt))
        assds.append(assd(pred, gt))

    def summarize(values: list[float]) -> dict[str, float]:
        lo, hi = _bootstrap_ci(values, n_boot=n_boot, alpha=alpha, seed=seed)
        return {"mean": _mean_ignoring_nan(values), "ci_low": lo, "ci_high": hi}

    return {
        "n_images": len(stems),
        "dice": summarize(dices),
        "iou": summarize(ious),
        "hd95_px": summarize(hd95s),
        "assd_px": summarize(assds),
        "note": "HD95/ASSD in pixels; NaN (single-sided empty) excluded from means.",
    }


# --------------------------------------------------------------------------- #
# Grading metrics
# --------------------------------------------------------------------------- #
def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> np.ndarray:
    """Integer confusion matrix ``M[t, p]`` (rows = truth, cols = prediction)."""
    matrix = np.zeros((n_classes, n_classes), dtype=np.int64)
    for t, p in zip(y_true, y_pred, strict=True):
        matrix[int(t), int(p)] += 1
    return matrix


def per_class_sensitivity(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> list[float]:
    """Recall for each class (``NaN`` for classes absent from the truth)."""
    matrix = confusion_matrix(y_true, y_pred, n_classes)
    out: list[float] = []
    for c in range(n_classes):
        support = matrix[c].sum()
        out.append(float(matrix[c, c] / support) if support > 0 else float("nan"))
    return out


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, n_classes: int) -> float:
    """Mean per-class recall over the classes that appear in the truth."""
    sens = per_class_sensitivity(y_true, y_pred, n_classes)
    present = [s for s in sens if np.isfinite(s)]
    return float(np.mean(present)) if present else float("nan")


def quadratic_weighted_kappa(
    y_true: np.ndarray, y_pred: np.ndarray, n_classes: int
) -> float:
    """Cohen's κ with quadratic weights — the ordinal grading primary.

    Penalises distant mis-grades (calling Stage 4 a Stage 1) far more than
    adjacent ones, matching the clinical cost of under-staging.
    """
    observed = confusion_matrix(y_true, y_pred, n_classes).astype(float)
    total = observed.sum()
    if total == 0:
        return float("nan")
    weights = np.zeros((n_classes, n_classes), dtype=float)
    for i in range(n_classes):
        for j in range(n_classes):
            weights[i, j] = ((i - j) ** 2) / ((n_classes - 1) ** 2)
    row = observed.sum(axis=1)
    col = observed.sum(axis=0)
    expected = np.outer(row, col) / total
    denom = float((weights * expected).sum())
    if denom == 0.0:
        return float("nan")
    return 1.0 - float((weights * observed).sum()) / denom


def _bootstrap_metric_by_group(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    groups: np.ndarray,
    metric: Callable[[np.ndarray, np.ndarray], float],
    *,
    n_boot: int,
    alpha: float,
    seed: int,
) -> tuple[float, float]:
    """Patient-clustered bootstrap CI: resample whole patients, recompute metric."""
    unique = np.unique(groups)
    index_by_group = {g: np.nonzero(groups == g)[0] for g in unique}
    rng = np.random.default_rng(seed)
    stats = np.empty(n_boot, dtype=float)
    for b in range(n_boot):
        chosen = rng.choice(unique, size=unique.size, replace=True)
        rows = np.concatenate([index_by_group[g] for g in chosen])
        value = metric(y_true[rows], y_pred[rows])
        stats[b] = value
    finite = stats[np.isfinite(stats)]
    if finite.size == 0:
        return float("nan"), float("nan")
    lo = float(np.percentile(finite, 100 * alpha / 2))
    hi = float(np.percentile(finite, 100 * (1 - alpha / 2)))
    return lo, hi


def expected_calibration_error(
    confidences: np.ndarray, correct: np.ndarray, *, n_bins: int = 15
) -> tuple[float, list[dict[str, float]]]:
    """Equal-width ECE and a reliability table over confidence bins."""
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    n = confidences.size
    ece = 0.0
    table: list[dict[str, float]] = []
    for b in range(n_bins):
        lo, hi = edges[b], edges[b + 1]
        in_bin = (confidences > lo) & (confidences <= hi) if b > 0 else (confidences <= hi)
        count = int(in_bin.sum())
        if count == 0:
            table.append({"bin_low": float(lo), "bin_high": float(hi), "count": 0,
                          "confidence": float("nan"), "accuracy": float("nan")})
            continue
        acc = float(correct[in_bin].mean())
        conf = float(confidences[in_bin].mean())
        ece += (count / n) * abs(acc - conf)
        table.append({"bin_low": float(lo), "bin_high": float(hi), "count": count,
                      "confidence": conf, "accuracy": acc})
    return float(ece), table


def brier_score(probs: np.ndarray, y_true: np.ndarray, n_classes: int) -> float:
    """Multiclass Brier score (mean squared error against one-hot truth)."""
    one_hot = np.zeros_like(probs)
    one_hot[np.arange(y_true.size), y_true] = 1.0
    return float(((probs - one_hot) ** 2).sum(axis=1).mean())


def evaluate_grading(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    *,
    groups: np.ndarray,
    classes: Sequence[str],
    probs: np.ndarray | None = None,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
    n_bins: int = 15,
) -> dict[str, Any]:
    """Full grading report: κ, balanced accuracy, sensitivity, calibration."""
    n_classes = len(classes)

    def qwk(a: np.ndarray, b: np.ndarray) -> float:
        return quadratic_weighted_kappa(a, b, n_classes)

    def bacc(a: np.ndarray, b: np.ndarray) -> float:
        return balanced_accuracy(a, b, n_classes)

    kappa = qwk(y_true, y_pred)
    kappa_ci = _bootstrap_metric_by_group(
        y_true, y_pred, groups, qwk, n_boot=n_boot, alpha=alpha, seed=seed
    )
    bal = bacc(y_true, y_pred)
    bal_ci = _bootstrap_metric_by_group(
        y_true, y_pred, groups, bacc, n_boot=n_boot, alpha=alpha, seed=seed
    )
    sens = per_class_sensitivity(y_true, y_pred, n_classes)

    report: dict[str, Any] = {
        "n_samples": int(y_true.size),
        "n_patients": int(np.unique(groups).size),
        "classes": list(classes),
        "quadratic_weighted_kappa": {
            "value": kappa, "ci_low": kappa_ci[0], "ci_high": kappa_ci[1]
        },
        "balanced_accuracy": {"value": bal, "ci_low": bal_ci[0], "ci_high": bal_ci[1]},
        "per_stage_sensitivity": {classes[c]: sens[c] for c in range(n_classes)},
        "confusion_matrix": confusion_matrix(y_true, y_pred, n_classes).tolist(),
    }

    if probs is not None:
        confidences = probs.max(axis=1)
        predicted = probs.argmax(axis=1)
        correct = (predicted == y_true).astype(float)
        ece, table = expected_calibration_error(confidences, correct, n_bins=n_bins)
        report["calibration"] = {
            "ece": ece,
            "brier": brier_score(probs, y_true, n_classes),
            "n_bins": n_bins,
            "reliability": table,
        }
    return report


# --------------------------------------------------------------------------- #
# Grading CSV loading
# --------------------------------------------------------------------------- #
def _label_to_index(value: str, classes: Sequence[str]) -> int:
    value = value.strip()
    if value in classes:
        return classes.index(value)
    try:
        idx = int(value)
    except ValueError as exc:
        raise SystemExit(f"unknown label {value!r}; expected one of {list(classes)}") from exc
    if not 0 <= idx < len(classes):
        raise SystemExit(f"label index {idx} out of range for {len(classes)} classes")
    return idx


def load_grading_csv(
    path: Path, classes: Sequence[str]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Load ``true,pred[,patient_id][,prob_<class>...]`` into arrays.

    Returns ``(y_true, y_pred, groups, probs)``. ``probs`` is ``None`` unless a
    full set of ``prob_<class>`` columns is present, in which case calibration is
    computed. ``patient_id`` defaults to the row index (each sample its own group).
    """
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        if "true" not in fieldnames or "pred" not in fieldnames:
            raise SystemExit(f"{path} must have at least 'true' and 'pred' columns")
        prob_cols = [f"prob_{c}" for c in classes]
        has_probs = all(col in fieldnames for col in prob_cols)
        rows = list(reader)

    y_true = np.array([_label_to_index(r["true"], classes) for r in rows], dtype=np.int64)
    y_pred = np.array([_label_to_index(r["pred"], classes) for r in rows], dtype=np.int64)
    groups = np.array(
        [r.get("patient_id", "").strip() or str(i) for i, r in enumerate(rows)],
        dtype=object,
    )
    probs = None
    if has_probs:
        probs = np.array(
            [[float(r[col]) for col in prob_cols] for r in rows], dtype=np.float64
        )
        row_sums = probs.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1.0
        probs = probs / row_sums
    return y_true, y_pred, groups, probs


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Wound seg / grading evaluation metrics.")
    sub = parser.add_subparsers(dest="command", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--out", type=Path, default=None, help="Write the metrics JSON here.")
    common.add_argument("--n-boot", type=int, default=1000, help="Bootstrap resamples for CIs.")
    common.add_argument(
        "--alpha", type=float, default=0.05, help="CI significance level (0.05 gives a 95pct CI)."
    )
    common.add_argument("--seed", type=int, default=0, help="Bootstrap RNG seed.")

    seg_p = sub.add_parser("seg", parents=[common], help="Segmentation Dice/IoU/HD95/ASSD.")
    seg_p.add_argument("--pred-dir", type=Path, required=True, help="Predicted masks directory.")
    seg_p.add_argument("--gt-dir", type=Path, required=True, help="Ground-truth masks directory.")

    grade_p = sub.add_parser("grade", parents=[common], help="Grading κ / balanced-acc / ECE.")
    grade_p.add_argument("--csv", type=Path, required=True, help="Predictions CSV (see docstring).")
    grade_p.add_argument(
        "--classes",
        type=str,
        default=",".join(DEFAULT_GRADE_CLASSES),
        help="Comma-separated ordinal class order (used for weighted κ).",
    )
    grade_p.add_argument("--n-bins", type=int, default=15, help="Calibration reliability bins.")

    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    if args.command == "seg":
        report: dict[str, Any] = {
            "task": "segmentation",
            "pred_dir": str(args.pred_dir),
            "gt_dir": str(args.gt_dir),
            **evaluate_segmentation(
                args.pred_dir, args.gt_dir,
                n_boot=args.n_boot, alpha=args.alpha, seed=args.seed,
            ),
        }
        for key in ("dice", "iou", "hd95_px", "assd_px"):
            stat = report[key]
            print(
                f"  {key:9s} mean={stat['mean']:.4f}  "
                f"95% CI [{stat['ci_low']:.4f}, {stat['ci_high']:.4f}]"
            )
    elif args.command == "grade":
        classes = tuple(c.strip() for c in args.classes.split(",") if c.strip())
        y_true, y_pred, groups, probs = load_grading_csv(args.csv, classes)
        report = {
            "task": "grading",
            "csv": str(args.csv),
            **evaluate_grading(
                y_true, y_pred, groups=groups, classes=classes, probs=probs,
                n_boot=args.n_boot, alpha=args.alpha, seed=args.seed, n_bins=args.n_bins,
            ),
        }
        kappa = report["quadratic_weighted_kappa"]
        bal = report["balanced_accuracy"]
        print(
            f"  quadratic-weighted kappa = {kappa['value']:.4f}  "
            f"95% CI [{kappa['ci_low']:.4f}, {kappa['ci_high']:.4f}]"
        )
        print(
            f"  balanced accuracy        = {bal['value']:.4f}  "
            f"95% CI [{bal['ci_low']:.4f}, {bal['ci_high']:.4f}]"
        )
        for name, value in report["per_stage_sensitivity"].items():
            print(f"    sensitivity[{name}] = {value:.4f}")
        if "calibration" in report:
            cal = report["calibration"]
            print(f"  ECE = {cal['ece']:.4f}   Brier = {cal['brier']:.4f}")
    else:  # pragma: no cover - argparse enforces the choices
        raise SystemExit(f"unknown command {args.command!r}")

    if args.out is not None:
        args.out.expanduser().parent.mkdir(parents=True, exist_ok=True)
        args.out.expanduser().write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

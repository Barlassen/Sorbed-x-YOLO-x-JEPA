"""Surface-state targets and metrics, separate from tissue type and clinical stage.

One segmentation head predicts background + three observed surface states.
Unknown is an annotation ignore value / inference abstention, not intact skin.
"""

from __future__ import annotations

import numpy as np

SURFACE_CLASSES = {
    "background": 0,
    "intact_lesion": 1,
    "open_surface": 2,
    "covered_surface": 3,
    "unknown": 255,
}
SURFACE_COLORS = {
    "intact_lesion": "#38bdf8",
    "open_surface": "#fb7185",
    "covered_surface": "#facc15",
    "unknown": "#94a3b8",
}
NUM_CLASSES = 4
IGNORE_INDEX = 255


def validate_target(target: np.ndarray) -> None:
    if target.ndim != 2 or not np.issubdtype(target.dtype, np.integer):
        raise ValueError("Surface mask must be a 2D integer class-ID array")
    if not set(np.unique(target)).issubset(SURFACE_CLASSES.values()):
        raise ValueError("Unknown surface class ID; do not pass tissue or RGB masks")


def build_target(lesion: np.ndarray, regions: list[tuple[int, np.ndarray]]) -> np.ndarray:
    """Only explicitly labeled regions become known. Lesion remainder stays 255.

    Different known labels may not overlap; same-class polygons may overlap.
    Unknown polygons can additionally mark occluded regions outside the lesion.
    No complement operation produces an intact-skin label.
    """
    target = np.zeros(lesion.shape, dtype=np.uint8)
    target[lesion.astype(bool)] = IGNORE_INDEX
    assigned = np.zeros(lesion.shape, dtype=bool)
    for class_id, mask in regions:
        if class_id not in {1, 2, 3, IGNORE_INDEX} or mask.shape != lesion.shape:
            raise ValueError("Invalid surface region")
        mask = mask.astype(bool)
        if class_id != IGNORE_INDEX and np.any(mask & ~lesion.astype(bool)):
            raise ValueError("Known surface polygon lies outside the lesion")
        if np.any(mask & assigned & (target != class_id)):
            raise ValueError("Conflicting surface classes overlap; resolve the annotation")
        target[mask] = class_id
        assigned[mask] = True
    return target


def surface_counts(pred: np.ndarray, reference: np.ndarray) -> dict:
    """Per-class overlap on known reference pixels; predicted abstentions count as FN.

    Both-empty class pairs yield None, never a fabricated perfect score.
    Reference-unknown pixels are excluded from every class including background.
    """
    validate_target(pred)
    validate_target(reference)
    if pred.shape != reference.shape:
        raise ValueError("Prediction/reference shape mismatch; evaluation never resizes masks")
    valid = reference != IGNORE_INDEX
    result = {
        "known_reference_pixels": int(valid.sum()),
        "ignored_reference_pixels": int((~valid).sum()),
        "abstained_known_pixels": int(((pred == IGNORE_INDEX) & valid).sum()),
        "classes": {},
    }
    for name, class_id in SURFACE_CLASSES.items():
        if class_id in {0, IGNORE_INDEX}:
            continue
        p, g = (pred == class_id) & valid, (reference == class_id) & valid
        tp, ps, gs = int((p & g).sum()), int(p.sum()), int(g.sum())
        denominator = ps + gs
        result["classes"][name] = {
            "tp": tp, "fp": ps - tp, "fn": gs - tp,
            "reference_pixels": gs, "predicted_pixels": ps,
            "dice": 2 * tp / denominator if denominator else None,
        }
    return result


def aggregate_counts(results: list[dict]) -> dict:
    output = {}
    for name in ("intact_lesion", "open_surface", "covered_surface"):
        values = [r["classes"][name] for r in results]
        tp = sum(v["tp"] for v in values)
        fp = sum(v["fp"] for v in values)
        fn = sum(v["fn"] for v in values)
        dice = [v["dice"] for v in values if v["dice"] is not None]
        denominator = 2 * tp + fp + fn
        output[name] = {
            "pooled_dice": 2 * tp / denominator if denominator else None,
            "mean_image_dice": float(np.mean(dice)) if dice else None,
            "n_scored": len(dice), "n_both_empty": len(values) - len(dice),
            "tp": tp, "fp": fp, "fn": fn,
        }
    return output


def require_review(record: dict, purpose: str) -> None:
    flag = {"training": "eligible_for_training", "evaluation": "eligible_for_ground_truth"}[purpose]
    reviewer = record.get("reviewer")
    if (record.get("review_status") != "clinician_reviewed"
            or record.get(flag) is not True
            or not isinstance(reviewer, dict)
            or not reviewer.get("id") or not reviewer.get("date")):
        raise ValueError(f"{record.get('image_id')}: clinician-reviewed {purpose} labels required")

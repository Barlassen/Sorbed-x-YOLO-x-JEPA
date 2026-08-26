#!/usr/bin/env python3
"""Continual (human-in-the-loop) fine-tune of the deployed grader — with a gate.

This is the *second teacher* in the Sorbed pipeline. The rule teacher
(:class:`training.teacher_student.DirectiveTeacher`) abstains under uncertainty
and structurally under-labels the depth-dependent stages; nurse/HQ corrections
harvested by the runtime feedback store are the only ground truth for exactly
those cases. This module folds a small batch of those corrections into the
currently-promoted ConvNeXt-V2 grader, using a **small learning rate plus a
replay buffer** so the update nudges the model without forgetting the distilled
base, and then **refuses to promote** the result unless it demonstrably does not
regress balanced accuracy, quadratic-weighted kappa, or calibration (ECE) on a
frozen hold-out.

Where it sits in the chain
--------------------------
1. Runtime emits ``inferences.jsonl`` + ``feedback.jsonl`` (the feedback store).
2. ``sorbed feedback export`` JOINs feedback onto inferences and writes a training
   manifest CSV with columns ``image,mask,stage,patient_id,sample_weight,source``
   (label = corrected stage when present else predicted; human weight > rule).
3. **This module** consumes that manifest plus a replay manifest of
   previously-confirmed rows, fine-tunes, gates, and — on promotion — exports ONNX
   and appends the round's confirmed rows to the replay manifest.
4. :func:`training.fl_client.package_client_update` turns the promoted round into a
   federated client delta for a future cross-hospital FedAvg/FedProx server.

torch is imported lazily inside functions, so this module (and its manifest
readers, weighting, and replay logic) imports and runs even when torch is absent.

Example
-------
    python -m training.continual --config training/configs/continual.yaml \\
        --manifest round7/new.csv --replay-manifest store/replay.csv \\
        --holdout-manifest eval/frozen_holdout.csv --base-ckpt artifacts/grade/best.pt
"""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

from sorbed.domain.enums import PressureInjuryStage
from training.evaluate import expected_calibration_error

if TYPE_CHECKING:  # torch/timm are training-only; imported lazily where used.
    import torch

# Canonical grade axis — mirrors :data:`training.data.GRADE_CLASSES` but declared
# here from the torch-free enum so manifest parsing needs no training stack.
GRADE_CLASSES: tuple[str, ...] = (
    PressureInjuryStage.STAGE_1.value,
    PressureInjuryStage.STAGE_2.value,
    PressureInjuryStage.STAGE_3.value,
    PressureInjuryStage.STAGE_4.value,
    PressureInjuryStage.UNSTAGEABLE.value,
    PressureInjuryStage.DEEP_TISSUE.value,
)
_STAGE_TO_INDEX: dict[str, int] = {name: i for i, name in enumerate(GRADE_CLASSES)}
_N_STAGES = len(GRADE_CLASSES)
# Columns written to a manifest / replay buffer. ``grader_sha`` carries the
# provenance of the predicted label a row is built on (feedback export sets it),
# so the trainer can bound self-generated labels.
_MANIFEST_COLUMNS = (
    "image", "mask", "stage", "patient_id", "sample_weight", "source", "grader_sha"
)
# Only the original six are REQUIRED to read a manifest; grader_sha is optional so
# manifests produced before provenance plumbing still load (with an empty sha).
_REQUIRED_COLUMNS = ("image", "mask", "stage", "patient_id", "sample_weight", "source")

# Sources eligible for the replay buffer: only INDEPENDENT human corrections. A
# bare "agree" (``human_confirmed``) is a vote on the model's own output, not an
# independent label, so it is deliberately excluded — admitting it would let the
# model's predictions re-enter its own training set and compound in replay.
_HUMAN_SOURCES = frozenset({"human_corrected", "correction"})


@dataclass(frozen=True)
class ContinualRow:
    """One joined training row from the feedback export (or the replay buffer)."""

    image: Path
    mask: Path | None
    stage_index: int
    patient_id: str
    sample_weight: float
    source: str
    grader_sha: str = ""

    @property
    def is_human(self) -> bool:
        """Whether this row is an independent human correction (replay-eligible).

        A bare confirmation (``human_confirmed``) is intentionally NOT human here:
        it votes on the model's own output and must not seed the replay buffer.
        """
        return self.source.strip().lower() in _HUMAN_SOURCES


@dataclass(frozen=True)
class GradeMetrics:
    """Hold-out grading metrics used by the promotion gate."""

    balanced_accuracy: float
    qwk: float
    ece: float

    def as_dict(self) -> dict[str, float]:
        return {
            "balanced_accuracy": self.balanced_accuracy,
            "qwk": self.qwk,
            "ece": self.ece,
        }


# ---------------------------------------------------------------------------
# Manifest I/O (torch-free)
# ---------------------------------------------------------------------------
def read_continual_manifest(manifest: str | Path) -> list[ContinualRow]:
    """Parse a joined feedback manifest into :class:`ContinualRow` records.

    Columns: ``image,mask,stage,patient_id,sample_weight,source``. ``image`` and
    ``mask`` may be absolute or relative to the manifest's directory; an empty
    ``mask`` means grade-only (no segmentation supervision for that row).
    """
    path = Path(manifest)
    base = path.parent
    rows: list[ContinualRow] = []
    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or not set(_REQUIRED_COLUMNS).issubset(reader.fieldnames):
            raise SystemExit(
                f"manifest {path} must have columns {list(_REQUIRED_COLUMNS)}; "
                f"got {reader.fieldnames}"
            )
        for line_no, row in enumerate(reader, start=2):
            stage = row["stage"].strip()
            if stage not in _STAGE_TO_INDEX:
                raise SystemExit(
                    f"{path}:{line_no}: unknown stage {stage!r}; expected one of "
                    f"{list(GRADE_CLASSES)}"
                )
            image = _resolve(base, row["image"])
            mask_field = (row.get("mask") or "").strip()
            mask = _resolve(base, mask_field) if mask_field else None
            rows.append(
                ContinualRow(
                    image=image,
                    mask=mask,
                    stage_index=_STAGE_TO_INDEX[stage],
                    patient_id=row["patient_id"].strip(),
                    sample_weight=_parse_float(row.get("sample_weight"), default=1.0),
                    source=(row.get("source") or "rule").strip() or "rule",
                    grader_sha=(row.get("grader_sha") or "").strip(),
                )
            )
    if not rows:
        raise SystemExit(f"manifest {path} contained no rows")
    return rows


def append_replay_rows(replay_manifest: str | Path, rows: list[ContinualRow]) -> int:
    """Append confirmed rows to the replay manifest (creating it with a header).

    Paths are written absolute so the replay buffer resolves regardless of the
    working directory of a later round. Returns the number of rows appended.
    """
    path = Path(replay_manifest)
    path.parent.mkdir(parents=True, exist_ok=True)
    new_file = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        if new_file:
            writer.writerow(_MANIFEST_COLUMNS)
        for row in rows:
            writer.writerow(
                [
                    str(row.image.resolve()),
                    str(row.mask.resolve()) if row.mask is not None else "",
                    GRADE_CLASSES[row.stage_index],
                    row.patient_id,
                    f"{row.sample_weight:.6g}",
                    row.source,
                    row.grader_sha,
                ]
            )
    return len(rows)


def _resolve(base: Path, raw: str) -> Path:
    p = Path(raw.strip())
    return p if p.is_absolute() else (base / p)


def _parse_float(value: str | None, *, default: float) -> float:
    if value is None or not value.strip():
        return default
    try:
        return float(value)
    except ValueError:
        return default


# ---------------------------------------------------------------------------
# Weighting + replay assembly (torch-free)
# ---------------------------------------------------------------------------
def effective_weight(row: ContinualRow, cfg: dict[str, Any]) -> float:
    """Per-row training weight: human corrections dominate rule-teacher labels.

    When ``respect_manifest_weight`` and the export supplied a positive
    ``sample_weight`` (already HQ/physician > nurse > rule), that value is used
    directly; otherwise the ``source_weights`` map provides the fallback.
    """
    source_weights: dict[str, float] = cfg.get("source_weights", {}) or {}
    key = row.source.strip().lower()
    fallback = float(source_weights.get(key, 1.0))
    if bool(cfg.get("respect_manifest_weight", True)) and row.sample_weight > 0.0:
        return float(row.sample_weight)
    return fallback


def stratified_sample(
    rows: list[ContinualRow], target: int, rng: np.random.Generator
) -> list[ContinualRow]:
    """Draw ``target`` rows stratified by stage (round-robin across classes).

    Rare stages (Stage 4, unstageable, DTI) stay represented instead of being
    averaged away by the majority class — the property the replay buffer needs.
    """
    if target <= 0 or not rows:
        return []
    by_class: dict[int, list[ContinualRow]] = defaultdict(list)
    for row in rows:
        by_class[row.stage_index].append(row)
    order = sorted(by_class)
    queues = {c: list(rng.permutation(len(by_class[c]))) for c in order}
    picked: list[ContinualRow] = []
    while len(picked) < min(target, len(rows)):
        progressed = False
        for c in order:
            if queues[c]:
                picked.append(by_class[c][queues[c].pop()])
                progressed = True
                if len(picked) >= min(target, len(rows)):
                    break
        if not progressed:
            break
    return picked


def build_training_rows(
    new_rows: list[ContinualRow],
    replay_rows: list[ContinualRow],
    cfg: dict[str, Any],
    rng: np.random.Generator,
) -> list[ContinualRow]:
    """Combine new corrections with a stratified replay sample, capped per class.

    Replay is sampled up to ``replay_ratio × len(new_rows)`` (bounded by
    ``replay_size``) so a tiny weekly batch cannot overfit the last round. A final
    per-class cap (``max_class_fraction``) prevents any single stage from dominating
    the round.
    """
    n_new = len(new_rows)
    replay_target = min(
        len(replay_rows),
        round(float(cfg.get("replay_ratio", 2.0)) * max(1, n_new)),
        int(cfg.get("replay_size", 4000)),
    )
    replay_sample = stratified_sample(replay_rows, replay_target, rng)
    combined = list(new_rows) + replay_sample
    return _apply_class_cap(combined, float(cfg.get("max_class_fraction", 0.35)), rng)


def _apply_class_cap(
    rows: list[ContinualRow], max_fraction: float, rng: np.random.Generator
) -> list[ContinualRow]:
    if not rows or not 0.0 < max_fraction < 1.0:
        return rows
    by_class: dict[int, list[ContinualRow]] = defaultdict(list)
    for row in rows:
        by_class[row.stage_index].append(row)
    cap = max(1, int(len(rows) * max_fraction))
    kept: list[ContinualRow] = []
    for members in by_class.values():
        if len(members) <= cap:
            kept.extend(members)
            continue
        idx = rng.permutation(len(members))[:cap]
        kept.extend(members[i] for i in idx)
    return kept


def label_histogram(rows: list[ContinualRow]) -> dict[str, int]:
    """Count of rows per stage name — a non-PHI stat for FL data-quality gating."""
    counts: dict[str, int] = dict.fromkeys(GRADE_CLASSES, 0)
    for row in rows:
        counts[GRADE_CLASSES[row.stage_index]] += 1
    return counts


# ---------------------------------------------------------------------------
# Datasets (map-style; torch imported lazily inside __getitem__)
# ---------------------------------------------------------------------------
class _WeightedGradeDataset:
    """Map-style grade dataset yielding ``(image_tensor, stage_index, weight)``.

    Not a ``torch.utils.data.Dataset`` subclass on purpose — ``DataLoader`` accepts
    any object with ``__len__``/``__getitem__``, which keeps this module importable
    without torch. torch/data are imported lazily per item.
    """

    def __init__(
        self, rows: list[ContinualRow], weights: list[float], input_size: int, *, augment: bool
    ) -> None:
        self._rows = rows
        self._weights = weights
        self._size = int(input_size)
        self._augment = augment

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int, float]:
        from training import data as data_mod

        row = self._rows[index]
        rgb = data_mod.read_rgb(row.image, self._size)
        if self._augment:
            rgb = data_mod.strong_augment(rgb)
        return data_mod.to_input_tensor(rgb), row.stage_index, float(self._weights[index])


class _MaskedSegDataset:
    """Map-style seg dataset over rows that carry a mask: ``(image, mask)`` float."""

    def __init__(self, rows: list[ContinualRow], input_size: int, *, augment: bool) -> None:
        self._rows = [r for r in rows if r.mask is not None]
        self._size = int(input_size)
        self._augment = augment

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        import torch

        from training import data as data_mod

        row = self._rows[index]
        rgb = data_mod.read_rgb(row.image, self._size)
        mask = data_mod.read_mask(row.mask, self._size)
        if self._augment and float(torch.rand(1).item()) < 0.5:
            rgb = np.ascontiguousarray(rgb[:, ::-1, :])
            mask = np.ascontiguousarray(mask[:, ::-1])
        binary = (mask > 0).astype(np.float32)[np.newaxis, ...]
        return data_mod.to_input_tensor(rgb), torch.from_numpy(np.ascontiguousarray(binary))


# ---------------------------------------------------------------------------
# Grader: load, fine-tune, evaluate
# ---------------------------------------------------------------------------
def _load_grader_state(ckpt_path: Path) -> dict[str, Any]:
    """Load a grader state dict from a checkpoint, preferring the EMA student."""
    import torch

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if isinstance(state, dict) and "ema_state" in state:
        return state["ema_state"]
    if isinstance(state, dict) and "model_state" in state:
        return state["model_state"]
    return state


def _build_and_init_grader(cfg: dict[str, Any], device: torch.device) -> torch.nn.Module:
    from training.models import build_grader

    model = build_grader(
        model_name=str(cfg["model_name"]),
        num_classes=_N_STAGES,
        pretrained=bool(cfg.get("pretrained", True)) and not cfg.get("base_ckpt"),
    )
    base_ckpt = cfg.get("base_ckpt")
    if base_ckpt:
        model.load_state_dict(_load_grader_state(Path(base_ckpt)), strict=True)
    return model.to(device)


def fine_tune_grader(
    model: torch.nn.Module,
    rows: list[ContinualRow],
    cfg: dict[str, Any],
    device: torch.device,
) -> torch.nn.Module:
    """Small-LR, replay-mixed fine-tune with per-sample weighted cross-entropy.

    Returns the EMA student module — the mean-teacher copy is what the promotion
    gate scores and what gets exported, matching the base training recipe.
    """
    import torch
    from torch.utils.data import DataLoader

    from training import memory, utils

    weights = [effective_weight(r, cfg) for r in rows]
    loader = DataLoader(
        _WeightedGradeDataset(rows, weights, int(cfg["input_size"]), augment=True),
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=memory.safe_num_workers(int(cfg.get("num_workers", 0))),
        drop_last=False,
        pin_memory=device.type == "cuda",
    )
    ema = utils.ModelEma(model, decay=float(cfg.get("ema_decay", 0.999)))
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=float(cfg["lr"]), weight_decay=float(cfg.get("weight_decay", 0.05))
    )
    steps_per_epoch = max(1, len(loader))
    scheduler = utils.cosine_warmup_scheduler(
        optimizer,
        total_steps=steps_per_epoch * int(cfg["epochs"]),
        warmup_steps=steps_per_epoch * int(cfg.get("warmup_epochs", 1)),
    )
    scaler = utils.make_grad_scaler(device, enabled=bool(cfg.get("amp", True)))

    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        meter = utils.AverageMeter()
        for images, targets, sample_w in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            sample_w = sample_w.to(device, non_blocking=True).float()
            # OOM-safe: empty the cache and retry the step once (the weighted-mean
            # loss must be computed over the whole batch, so we retry rather than
            # micro-batch); a persistent OOM raises a clear, actionable error.
            for attempt in range(2):
                try:
                    optimizer.zero_grad(set_to_none=True)
                    with utils.autocast_context(device, enabled=bool(cfg.get("amp", True))):
                        logits = model(images)
                        per_sample = torch.nn.functional.cross_entropy(
                            logits, targets, reduction="none")
                        loss = (per_sample * sample_w).sum() / sample_w.sum().clamp_min(1e-8)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                    break
                except RuntimeError as exc:
                    if not memory.is_oom_error(exc) or attempt == 1:
                        if memory.is_oom_error(exc):
                            raise MemoryError(
                                "CUDA OOM during continual fine-tune; lower batch_size/"
                                "input_size in the config or set vram_fraction."
                            ) from exc
                        raise
                    memory.empty_cache()
            scheduler.step()
            ema.update(model)
            meter.update(float(loss.item()), images.size(0))
        print(f"  [grader] epoch {epoch:02d}/{cfg['epochs']}  weighted_ce={meter.average:.4f}")
    return ema.module


@dataclass
class _GradeEval:
    y_true: np.ndarray
    y_pred: np.ndarray
    confidence: np.ndarray


def _infer_grader(
    model: torch.nn.Module, rows: list[ContinualRow], cfg: dict[str, Any], device: torch.device
) -> _GradeEval:
    import torch
    from torch.utils.data import DataLoader

    from training import memory

    loader = DataLoader(
        _WeightedGradeDataset(rows, [1.0] * len(rows), int(cfg["input_size"]), augment=False),
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        num_workers=memory.safe_num_workers(int(cfg.get("num_workers", 0))),
    )
    model.eval()
    trues: list[int] = []
    preds: list[int] = []
    confs: list[float] = []
    with torch.no_grad():
        for images, targets, _weight in loader:
            images = images.to(device, non_blocking=True)
            probs = torch.softmax(memory.safe_infer(model, images), dim=1)
            conf, pred = probs.max(dim=1)
            trues.extend(int(t) for t in targets.tolist())
            preds.extend(int(p) for p in pred.cpu().tolist())
            confs.extend(float(c) for c in conf.cpu().tolist())
    return _GradeEval(np.asarray(trues), np.asarray(preds), np.asarray(confs, dtype=np.float64))


def evaluate_grader(
    model: torch.nn.Module, rows: list[ContinualRow], cfg: dict[str, Any], device: torch.device
) -> GradeMetrics:
    """Balanced accuracy, QWK, and ECE on a hold-out set (torch inference)."""
    from training.train_grade import balanced_accuracy, quadratic_weighted_kappa

    ev = _infer_grader(model, rows, cfg, device)
    bacc = balanced_accuracy(ev.y_true, ev.y_pred, _N_STAGES)
    qwk = quadratic_weighted_kappa(ev.y_true, ev.y_pred, _N_STAGES)
    correct = (ev.y_pred == ev.y_true).astype(np.float64)
    ece, _table = expected_calibration_error(ev.confidence, correct, n_bins=10)
    return GradeMetrics(balanced_accuracy=bacc, qwk=qwk, ece=ece)


# ---------------------------------------------------------------------------
# Segmenter (optional joint fine-tune on rows that carry a mask)
# ---------------------------------------------------------------------------
def fine_tune_segmenter(
    rows: list[ContinualRow], cfg: dict[str, Any], device: torch.device
) -> torch.nn.Module:
    """Fine-tune the binary wound segmenter on rows that carry a mask (Dice+BCE)."""
    import torch
    from torch.utils.data import DataLoader

    from training import utils
    from training.losses import DiceBCELoss
    from training.models import build_segmenter

    model = build_segmenter(
        arch=str(cfg.get("seg_arch", "segformer")),
        encoder_name=str(cfg.get("seg_encoder", "mit_b3")),
        encoder_weights="imagenet" if bool(cfg.get("pretrained", True)) else None,
        classes=1,
    ).to(device)
    if cfg.get("base_seg_ckpt"):
        state = torch.load(Path(cfg["base_seg_ckpt"]), map_location="cpu", weights_only=False)
        model.load_state_dict(state.get("model_state", state), strict=True)

    dataset = _MaskedSegDataset(rows, int(cfg["input_size"]), augment=True)
    if len(dataset) == 0:
        raise SystemExit("train_segmenter=true but no rows in the manifest carry a mask")
    loader = DataLoader(
        dataset,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        num_workers=int(cfg.get("num_workers", 0)),
    )
    criterion = DiceBCELoss().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg.get("seg_lr", 1e-5)))
    for epoch in range(1, int(cfg.get("seg_epochs", 4)) + 1):
        model.train()
        meter = utils.AverageMeter()
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images), masks)
            loss.backward()
            optimizer.step()
            meter.update(float(loss.item()), images.size(0))
        print(f"  [seg] epoch {epoch:02d}/{cfg.get('seg_epochs', 4)}  dice_bce={meter.average:.4f}")
    return model


def evaluate_segmenter(
    model: torch.nn.Module, rows: list[ContinualRow], cfg: dict[str, Any], device: torch.device
) -> float:
    """Mean soft-Dice of the segmenter over masked hold-out rows."""
    import torch
    from torch.utils.data import DataLoader

    from training.losses import soft_dice_coefficient

    dataset = _MaskedSegDataset(rows, int(cfg["input_size"]), augment=False)
    if len(dataset) == 0:
        return 0.0
    loader = DataLoader(dataset, batch_size=int(cfg["batch_size"]), shuffle=False)
    model.eval()
    dice_sum = 0.0
    count = 0
    with torch.no_grad():
        for images, masks in loader:
            images = images.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            probs = torch.sigmoid(model(images))
            dice = soft_dice_coefficient(probs, masks, dims=(1, 2, 3)).mean()
            dice_sum += float(dice) * images.size(0)
            count += images.size(0)
    return dice_sum / max(1, count)


def _load_base_segmenter(cfg: dict[str, Any], device: torch.device) -> torch.nn.Module:
    """Build the incumbent segmenter and load its base checkpoint for scoring."""
    import torch

    from training.models import build_segmenter

    model = build_segmenter(
        arch=str(cfg.get("seg_arch", "segformer")),
        encoder_name=str(cfg.get("seg_encoder", "mit_b3")),
        encoder_weights="imagenet" if bool(cfg.get("pretrained", True)) else None,
        classes=1,
    ).to(device)
    state = torch.load(Path(cfg["base_seg_ckpt"]), map_location="cpu", weights_only=False)
    model.load_state_dict(state.get("model_state", state), strict=True)
    return model


# ---------------------------------------------------------------------------
# Promotion gate
# ---------------------------------------------------------------------------
def promotion_decision(
    base: GradeMetrics, candidate: GradeMetrics, cfg: dict[str, Any]
) -> tuple[bool, list[str]]:
    """Decide promotion: no regression in balanced accuracy, QWK, or ECE.

    Returns ``(promote, reasons)``. Each check is one-sided and ties go to the
    incumbent — a candidate must be at least as good (within tolerance) on every
    axis, and calibration may not worsen.
    """
    bacc_tol = float(cfg.get("bacc_tolerance", 0.02))
    qwk_tol = float(cfg.get("qwk_tolerance", 0.02))
    ece_tol = float(cfg.get("ece_tolerance", 0.01))
    reasons: list[str] = []
    ok_bacc = candidate.balanced_accuracy >= base.balanced_accuracy - bacc_tol
    ok_qwk = candidate.qwk >= base.qwk - qwk_tol
    ok_ece = candidate.ece <= base.ece + ece_tol
    reasons.append(
        f"balanced_accuracy {candidate.balanced_accuracy:.4f} vs base "
        f"{base.balanced_accuracy:.4f} (tol {bacc_tol}) -> {'pass' if ok_bacc else 'FAIL'}"
    )
    reasons.append(
        f"qwk {candidate.qwk:.4f} vs base {base.qwk:.4f} (tol {qwk_tol}) "
        f"-> {'pass' if ok_qwk else 'FAIL'}"
    )
    reasons.append(
        f"ece {candidate.ece:.4f} vs base {base.ece:.4f} (tol {ece_tol}) "
        f"-> {'pass' if ok_ece else 'FAIL'}"
    )
    return (ok_bacc and ok_qwk and ok_ece), reasons


# ---------------------------------------------------------------------------
# Round orchestration
# ---------------------------------------------------------------------------
def run_round(cfg: dict[str, Any]) -> dict[str, Any]:
    """Run one continual round end-to-end and return a JSON-able manifest.

    Steps: read manifests → assemble weighted training rows with replay → score
    the incumbent base on the frozen hold-out → fine-tune → score the candidate →
    gate. On promotion: export ONNX and append the round's confirmed rows to the
    replay manifest; otherwise leave the deployed model and buffer untouched.
    """
    import torch

    from training import memory, utils

    utils.set_seed(int(cfg.get("seed", 1234)))
    device = utils.resolve_device(str(cfg.get("device", "auto")))
    memory.configure(device, vram_fraction=cfg.get("vram_fraction"))
    rng = np.random.default_rng(int(cfg.get("seed", 1234)))
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    new_rows = read_continual_manifest(cfg["manifest"])
    replay_rows: list[ContinualRow] = []
    if cfg.get("replay_manifest"):
        replay_rows = read_continual_manifest(cfg["replay_manifest"])
    holdout_rows = read_continual_manifest(cfg["holdout_manifest"])
    train_rows = build_training_rows(new_rows, replay_rows, cfg, rng)
    print(
        f"round: new={len(new_rows)} replay_pool={len(replay_rows)} "
        f"train_rows={len(train_rows)} holdout={len(holdout_rows)} device={device}"
    )

    model = _build_and_init_grader(cfg, device)
    base_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    base_metrics = evaluate_grader(model, holdout_rows, cfg, device)
    print(f"base   : {base_metrics.as_dict()}")

    student = fine_tune_grader(model, train_rows, cfg, device)
    cand_metrics = evaluate_grader(student, holdout_rows, cfg, device)
    print(f"cand   : {cand_metrics.as_dict()}")

    promote, reasons = promotion_decision(base_metrics, cand_metrics, cfg)
    for line in reasons:
        print(f"  gate: {line}")

    seg_result: dict[str, Any] | None = None
    if bool(cfg.get("train_segmenter", False)):
        base_seg_dice = 0.0
        if cfg.get("base_seg_ckpt"):
            base_seg = _load_base_segmenter(cfg, device)
            base_seg_dice = evaluate_segmenter(base_seg, holdout_rows, cfg, device)
        seg_model = fine_tune_segmenter(train_rows, cfg, device)
        seg_dice = evaluate_segmenter(seg_model, holdout_rows, cfg, device)
        seg_ok = seg_dice >= base_seg_dice - float(cfg.get("seg_dice_tolerance", 0.02))
        seg_result = {"candidate_dice": seg_dice, "base_dice": base_seg_dice, "pass": seg_ok}
        promote = promote and seg_ok
        print(f"  gate: seg dice {seg_dice:.4f} vs base {base_seg_dice:.4f} "
              f"-> {'pass' if seg_ok else 'FAIL'}")

    candidate_path = out_dir / "candidate.pt"
    torch.save(
        {
            "model_state": student.state_dict(),
            "model_name": cfg["model_name"],
            "classes": list(GRADE_CLASSES),
            "input_size": int(cfg["input_size"]),
        },
        candidate_path,
    )

    manifest: dict[str, Any] = {
        "promoted": bool(promote),
        "base_metrics": base_metrics.as_dict(),
        "candidate_metrics": cand_metrics.as_dict(),
        "gate_reasons": reasons,
        "seg": seg_result,
        "counts": {
            "new_rows": len(new_rows),
            "replay_pool": len(replay_rows),
            "train_rows": len(train_rows),
            "holdout": len(holdout_rows),
        },
        "label_histogram": label_histogram(train_rows),
        "candidate_ckpt": str(candidate_path),
        "seed": int(cfg.get("seed", 1234)),
    }

    if promote:
        promoted_path = out_dir / "promoted.pt"
        torch.save(
            {
                "model_state": student.state_dict(),
                "model_name": cfg["model_name"],
                "classes": list(GRADE_CLASSES),
                "input_size": int(cfg["input_size"]),
            },
            promoted_path,
        )
        manifest["promoted_ckpt"] = str(promoted_path)
        manifest["base_state_sha256"] = _state_sha256(base_state)
        if bool(cfg.get("export_onnx", True)):
            from training.models import export_onnx

            onnx_path = export_onnx(
                student.to(torch.device("cpu")),
                out_dir / "model.onnx",
                input_size=int(cfg["input_size"]),
                output_names=("logits",),
                device=torch.device("cpu"),
            )
            manifest["onnx"] = str(onnx_path)
            manifest["onnx_sha256"] = utils.sha256_file(onnx_path)
        if cfg.get("replay_manifest"):
            confirmed = [r for r in new_rows if r.is_human]
            appended = append_replay_rows(cfg["replay_manifest"], confirmed)
            manifest["replay_appended"] = appended
        print(f"PROMOTED -> {promoted_path}")
    else:
        print("REJECTED (candidate regresses vs base) — deployed model unchanged")

    (out_dir / "round_report.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return manifest


def _state_sha256(state: dict[str, Any]) -> str:
    """Deterministic sha256 over a state dict's names + raw bytes (provenance)."""
    import hashlib

    digest = hashlib.sha256()
    for key in sorted(state):
        digest.update(key.encode("utf-8"))
        digest.update(np.ascontiguousarray(state[key].numpy()).tobytes())
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None, help="NEW joined rows (required)")
    parser.add_argument("--replay-manifest", type=Path, default=None,
                        help="previously-confirmed rows; appended on promotion")
    parser.add_argument("--holdout-manifest", type=Path, default=None,
                        help="frozen hold-out for the promotion gate (required)")
    parser.add_argument("--base-ckpt", type=Path, default=None, help="grader base .pt")
    parser.add_argument("--base-seg-ckpt", type=Path, default=None, help="segmenter base .pt")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--replay-ratio", type=float, default=None)
    parser.add_argument("--replay-size", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--pretrained", dest="pretrained", action="store_true", default=None)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--train-segmenter", dest="train_segmenter",
                        action="store_true", default=None)
    parser.add_argument("--no-onnx", dest="export_onnx", action="store_false", default=None)
    return parser.parse_args(argv)


_DEFAULTS: dict[str, Any] = {
    "model_name": "convnextv2_base.fcmae_ft_in22k_in1k_384",
    "pretrained": True,
    "input_size": 384,
    "lr": 2e-5,
    "weight_decay": 0.05,
    "epochs": 4,
    "warmup_epochs": 1,
    "batch_size": 16,
    "ema_decay": 0.999,
    "num_workers": 4,
    "amp": True,
    "seed": 1234,
    "device": "auto",
    "replay_ratio": 2.0,
    "replay_size": 4000,
    "max_class_fraction": 0.35,
    "respect_manifest_weight": True,
    "source_weights": {"hq": 3.0, "physician": 3.0, "human": 2.0, "nurse": 1.5, "rule": 1.0},
    "bacc_tolerance": 0.02,
    "qwk_tolerance": 0.02,
    "ece_tolerance": 0.01,
    "seg_dice_tolerance": 0.02,
    "train_segmenter": False,
    "seg_arch": "segformer",
    "seg_encoder": "mit_b3",
    "seg_epochs": 4,
    "seg_lr": 1e-5,
    "export_onnx": True,
    "out_dir": "artifacts/continual",
}


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    from training import utils

    file_config = utils.load_yaml_config(args.config) if args.config else {}
    cli = {k: v for k, v in vars(args).items() if k != "config"}
    merged = utils.merge_cli_over_config({**_DEFAULTS, **file_config}, cli)
    for key in ("manifest", "holdout_manifest"):
        if merged.get(key) is None:
            raise SystemExit(f"--{key.replace('_', '-')} is required (via CLI or config)")
    for key in (
        "manifest", "replay_manifest", "holdout_manifest",
        "base_ckpt", "base_seg_ckpt", "out_dir",
    ):
        if merged.get(key) is not None:
            merged[key] = Path(merged[key])
    return merged


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = build_config(args)
    result = run_round(cfg)
    return 0 if result["promoted"] else 3


if __name__ == "__main__":
    raise SystemExit(main())

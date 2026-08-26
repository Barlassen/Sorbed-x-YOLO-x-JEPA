"""Datasets, augmentation, and patient-level cross-validation splitting.

Data layout
-----------
Segmentation expects paired directories keyed by file stem::

    <root>/images/0001.png   <root>/masks/0001.png   # wound = nonzero

Grading and semi-supervision consume a **prepared manifest** — a CSV the operator
builds locally, since the only staged photo sets (e.g. PIID for NPIAP stages,
DFUC for infection/ischaemia) are either unlicensed or gated behind a data-use
agreement and must not be auto-downloaded. See ``training/configs`` for the
documented manual step. The manifest columns are::

    image,stage,patient_id[,skin_tone,site]

``image`` is a path (absolute, or relative to the manifest's directory).
``stage`` is one of the :class:`~sorbed.domain.enums.PressureInjuryStage` string
values in :data:`GRADE_CLASSES`. ``patient_id`` groups rows so no patient leaks
across a train/val fold. ``skin_tone`` and ``site`` are optional stratifiers.
"""

from __future__ import annotations

import csv
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset

from sorbed.domain.enums import PressureInjuryStage

IMAGE_SUFFIXES = frozenset({".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"})
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

# Sentinel class index written for the foreground of a *binary* (wound-vs-
# background) mask in a multiclass tissue run: the pixel is known to be wound
# but of unknown tissue class. ``MulticlassDiceCELoss(superset_index=...)``
# excludes these pixels from cross-entropy/Dice and instead pushes their
# foreground probability up. It is negative so it can never collide with a real
# tissue-class index read from a mask (those are >= 0).
SUPERSET_SENTINEL: int = -2

# The photo-stageable NPIAP classes, in ordinal-then-side order. Stages 1–4 form
# the ordered axis; Unstageable and Deep-Tissue-Injury are non-ordinal sides.
GRADE_CLASSES: tuple[str, ...] = (
    PressureInjuryStage.STAGE_1.value,
    PressureInjuryStage.STAGE_2.value,
    PressureInjuryStage.STAGE_3.value,
    PressureInjuryStage.STAGE_4.value,
    PressureInjuryStage.UNSTAGEABLE.value,
    PressureInjuryStage.DEEP_TISSUE.value,
)
GRADE_CLASS_TO_INDEX: dict[str, int] = {name: i for i, name in enumerate(GRADE_CLASSES)}


def _mean_std() -> tuple[np.ndarray, np.ndarray]:
    mean = np.asarray(IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray(IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)
    return mean, std


def read_rgb(path: Path, size: int | None = None) -> np.ndarray:
    """Read an image as an ``(H, W, 3)`` uint8 RGB array, optionally resized."""
    bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if bgr is None:
        raise RuntimeError(f"could not read image {path}")
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    if size is not None:
        rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)
    return rgb


def read_mask(path: Path, size: int | None = None) -> np.ndarray:
    """Read a single-channel mask as uint8 (nearest-resized), values preserved."""
    raw = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if raw is None:
        raise RuntimeError(f"could not read mask {path}")
    if size is not None:
        raw = cv2.resize(raw, (size, size), interpolation=cv2.INTER_NEAREST)
    return raw


def to_input_tensor(rgb_uint8: np.ndarray) -> torch.Tensor:
    """ImageNet-normalize an ``(H, W, 3)`` uint8 RGB array to a CHW float tensor."""
    mean, std = _mean_std()
    img = rgb_uint8.astype(np.float32) / 255.0
    img = np.transpose(img, (2, 0, 1))
    img = (img - mean) / std
    return torch.from_numpy(np.ascontiguousarray(img))


def _rand() -> float:
    return float(torch.rand(1).item())


def weak_augment(rgb_uint8: np.ndarray) -> np.ndarray:
    """Weak view: horizontal flip with probability 0.5 (label-preserving)."""
    if _rand() < 0.5:
        return np.ascontiguousarray(rgb_uint8[:, ::-1, :])
    return rgb_uint8


def strong_augment(rgb_uint8: np.ndarray) -> np.ndarray:
    """Strong view: flip + photometric jitter, for FixMatch/consistency.

    Photometric-only jitter (brightness/contrast in RGB, hue/saturation in HSV)
    keeps the wound geometry intact while forcing colour invariance — the
    property that makes staging robust across devices and skin tones.
    """
    out = rgb_uint8
    if _rand() < 0.5:
        out = np.ascontiguousarray(out[:, ::-1, :])
    # Brightness / contrast in RGB.
    brightness = 0.7 + 0.6 * _rand()
    contrast = 0.7 + 0.6 * _rand()
    out = out.astype(np.float32)
    mean = out.mean(axis=(0, 1), keepdims=True)
    out = (out - mean) * contrast + mean * brightness
    out = np.clip(out, 0.0, 255.0).astype(np.uint8)
    # Hue / saturation in HSV.
    hsv = cv2.cvtColor(out, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[..., 0] = (hsv[..., 0] + (_rand() - 0.5) * 20.0) % 180.0
    hsv[..., 1] = np.clip(hsv[..., 1] * (0.8 + 0.4 * _rand()), 0.0, 255.0)
    out = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)
    if _rand() < 0.3:
        out = cv2.GaussianBlur(out, (0, 0), sigmaX=1.0 + _rand())
    return np.ascontiguousarray(out)


def pair_by_stem(images_dir: Path, masks_dir: Path) -> list[tuple[Path, Path]]:
    """Pair each image with the mask sharing its stem; error if none match."""
    masks_by_stem: dict[str, Path] = {
        m.stem: m for m in masks_dir.iterdir() if m.suffix.lower() in IMAGE_SUFFIXES
    }
    pairs: list[tuple[Path, Path]] = []
    for image in sorted(images_dir.iterdir()):
        if image.suffix.lower() not in IMAGE_SUFFIXES:
            continue
        mask = masks_by_stem.get(image.stem)
        if mask is not None:
            pairs.append((image, mask))
    if not pairs:
        raise SystemExit(f"no image/mask pairs found between {images_dir} and {masks_dir}")
    return pairs


def list_images(images_dir: Path) -> list[Path]:
    """All image files in a directory, sorted by name."""
    return sorted(p for p in images_dir.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def group_kfold_indices(
    groups: Sequence[str],
    *,
    n_splits: int,
    seed: int = 0,
) -> list[tuple[list[int], list[int]]]:
    """Patient-level group K-fold: no group appears in both train and val.

    Groups are shuffled deterministically by ``seed`` then round-robin assigned to
    folds, so fold sizes stay balanced without splitting any patient. Returns a
    list of ``(train_indices, val_indices)`` per fold.
    """
    if n_splits < 2:
        raise ValueError("n_splits must be >= 2")
    unique = sorted(set(groups))
    if len(unique) < n_splits:
        raise ValueError(f"{len(unique)} groups < {n_splits} splits; cannot split cleanly")
    rng = np.random.default_rng(seed)
    order = rng.permutation(len(unique))
    fold_of_group: dict[str, int] = {}
    for rank, gi in enumerate(order):
        fold_of_group[unique[int(gi)]] = rank % n_splits
    folds: list[tuple[list[int], list[int]]] = []
    for fold in range(n_splits):
        train_idx = [i for i, g in enumerate(groups) if fold_of_group[g] != fold]
        val_idx = [i for i, g in enumerate(groups) if fold_of_group[g] == fold]
        folds.append((train_idx, val_idx))
    return folds


class SegmentationDataset(Dataset):
    """Paired image/mask dataset for binary or multiclass segmentation.

    For ``num_classes == 1`` masks are binarized (wound = nonzero) and returned as
    float ``(1, H, W)``. For ``num_classes > 1`` mask pixel values are taken as
    class indices and returned as long ``(H, W)`` for cross-entropy.

    **Partial labels.** In a multiclass run, sources whose masks are only
    wound-vs-background (``binary_flags[i]`` true) cannot supply a tissue class.
    When ``superset_index`` is set, such a mask's foreground is written as that
    sentinel (see :data:`SUPERSET_SENTINEL`) so the loss supervises localization
    without fabricating a tissue label; its background stays class 0. Sources with
    real tissue-class masks (flag false) pass their pixel values through unchanged.
    """

    def __init__(
        self,
        pairs: Sequence[tuple[Path, Path]],
        *,
        input_size: int,
        num_classes: int = 1,
        augment: bool = False,
        superset_index: int | None = None,
        binary_flags: Sequence[bool] | None = None,
    ) -> None:
        self._pairs = list(pairs)
        self._size = int(input_size)
        self._num_classes = int(num_classes)
        self._augment = augment
        self._superset_index = superset_index
        if binary_flags is None:
            self._binary = [False] * len(self._pairs)
        else:
            self._binary = [bool(flag) for flag in binary_flags]
            if len(self._binary) != len(self._pairs):
                raise ValueError(
                    f"binary_flags length {len(self._binary)} != "
                    f"pairs length {len(self._pairs)}"
                )

    def __len__(self) -> int:
        return len(self._pairs)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        image_path, mask_path = self._pairs[index]
        rgb = read_rgb(image_path, self._size)
        mask = read_mask(mask_path, self._size)
        if self._augment and _rand() < 0.5:
            rgb = np.ascontiguousarray(rgb[:, ::-1, :])
            mask = np.ascontiguousarray(mask[:, ::-1])
        image_tensor = to_input_tensor(rgb)
        if self._num_classes == 1:
            binary = (mask > 0).astype(np.float32)[np.newaxis, ...]
            return image_tensor, torch.from_numpy(np.ascontiguousarray(binary))
        if self._superset_index is not None and self._binary[index]:
            # Wound-vs-background mask: foreground -> superset sentinel, bg -> 0.
            target = np.where(mask > 0, self._superset_index, 0).astype(np.int64)
        else:
            target = mask.astype(np.int64)
        return image_tensor, torch.from_numpy(np.ascontiguousarray(target))


@dataclass(frozen=True)
class GradeRecord:
    """One manifest row for stage grading."""

    image: Path
    stage_index: int
    patient_id: str
    skin_tone: str | None = None
    site: str | None = None


def read_grade_manifest(manifest: Path) -> list[GradeRecord]:
    """Parse a grading manifest CSV into validated :class:`GradeRecord` rows."""
    manifest = Path(manifest)
    base = manifest.parent
    records: list[GradeRecord] = []
    with manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"image", "stage", "patient_id"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise SystemExit(f"manifest {manifest} must have columns {sorted(required)}")
        for line_no, row in enumerate(reader, start=2):
            stage = row["stage"].strip()
            if stage not in GRADE_CLASS_TO_INDEX:
                raise SystemExit(
                    f"{manifest}:{line_no}: unknown stage {stage!r}; "
                    f"expected one of {list(GRADE_CLASSES)}"
                )
            image_path = Path(row["image"].strip())
            if not image_path.is_absolute():
                image_path = base / image_path
            records.append(
                GradeRecord(
                    image=image_path,
                    stage_index=GRADE_CLASS_TO_INDEX[stage],
                    patient_id=row["patient_id"].strip(),
                    skin_tone=(row.get("skin_tone") or "").strip() or None,
                    site=(row.get("site") or "").strip() or None,
                )
            )
    if not records:
        raise SystemExit(f"manifest {manifest} contained no rows")
    return records


class GradingDataset(Dataset):
    """Stage-grading dataset over manifest records (single normalized view)."""

    def __init__(
        self,
        records: Sequence[GradeRecord],
        *,
        input_size: int,
        augment: bool = False,
    ) -> None:
        self._records = list(records)
        self._size = int(input_size)
        self._augment = augment

    def __len__(self) -> int:
        return len(self._records)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        record = self._records[index]
        rgb = read_rgb(record.image, self._size)
        if self._augment:
            rgb = strong_augment(rgb)
        return to_input_tensor(rgb), record.stage_index


class UnlabeledViewsDataset(Dataset):
    """Unlabeled images returning a weak and a strong view for FixMatch.

    Also returns the resized uint8 RGB so the rule teacher can compute
    colour/tissue features on the same pixels the student sees.
    """

    def __init__(self, images: Sequence[Path], *, input_size: int) -> None:
        self._images = list(images)
        self._size = int(input_size)

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        rgb = read_rgb(self._images[index], self._size)
        weak = to_input_tensor(weak_augment(rgb))
        strong = to_input_tensor(strong_augment(rgb))
        raw = torch.from_numpy(np.ascontiguousarray(rgb))
        return weak, strong, raw

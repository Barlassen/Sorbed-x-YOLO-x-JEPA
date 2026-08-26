"""Manifest schema, dataset adapters, patient-level splits, and augmentation.

This module is the data layer for Sorbed's segmentation/staging training. It is
deliberately dependency-light *at import time*: heavy stacks (``torch``,
``albumentations``) are imported lazily inside the functions that need them, so
this file — and the ``training/data_prep.py`` CLI built on top of it — imports
and lints cleanly on a machine that only has the Sorbed core installed.

The public surface is:

* :class:`ManifestRecord` — one row of the training manifest.
* :func:`write_manifest` / :func:`read_manifest` — CSV **and** JSONL round-trip.
* :func:`scan_azh_fuseg` — adapter for the AZH / FUSeg foot-ulcer layout
  distributed from github.com/uwm-bigdata/wound-segmentation.
* :func:`scan_generic` — adapter for a generic ``images/`` + ``masks/`` layout.
* :func:`split_by_patient` — leakage-free train/val/test split grouped on
  ``patient_id`` (no patient, hence no visit, appears in two splits).
* :func:`build_train_transform` / :func:`build_val_transform` — albumentations
  pipelines including colour normalization.
* :class:`WoundSegmentationDataset` — a ``torch.utils.data.Dataset`` yielding
  ``(image, mask)`` tensors ready for the ``scripts/train_segmenter.py`` loop.

Nothing here fabricates data or download URLs. Where a dataset is gated, the
adapter consumes a locally prepared directory; see ``training/DATA_README.md``.
"""

from __future__ import annotations

import csv
import json
import random
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover - typing only, no runtime import
    import numpy as np
    import torch

# We subclass ``object`` rather than ``torch.utils.data.Dataset`` so that importing
# this module never pulls in torch; ``WoundSegmentationDataset`` still satisfies the
# duck-typed Dataset protocol (``__len__`` + ``__getitem__``) a DataLoader needs.
_TorchDataset = object

# Image/mask file extensions we recognize when pairing by stem.
IMAGE_SUFFIXES: frozenset[str] = frozenset(
    {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff", ".webp"}
)

# ImageNet statistics — the encoders in segmentation-models-pytorch expect them.
IMAGENET_MEAN: tuple[float, float, float] = (0.485, 0.456, 0.406)
IMAGENET_STD: tuple[float, float, float] = (0.229, 0.224, 0.225)

# Manifest split names.
SPLIT_TRAIN = "train"
SPLIT_VAL = "val"
SPLIT_TEST = "test"
SPLITS: tuple[str, str, str] = (SPLIT_TRAIN, SPLIT_VAL, SPLIT_TEST)


# --------------------------------------------------------------------------- #
# Manifest schema
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class ManifestRecord:
    """One training sample.

    Attributes
    ----------
    image_path:
        Path to the RGB photo, relative to the manifest's ``root`` (or absolute).
    mask_path:
        Path to a single-channel binary mask (wound = nonzero). Empty string when
        the sample has no segmentation mask (e.g. a classification-only source).
    body_part:
        Coarse anatomical site, e.g. ``"foot"``, ``"sacrum"``, ``"mixed"``.
    stage_label:
        Optional pressure-injury / severity label (e.g. ``"stage_2"``), or empty
        string when the source carries no grade. Kept as a free string so the
        column survives datasets with differing label vocabularies; validate it
        against :class:`sorbed.domain.enums.PressureInjuryStage` before use.
    patient_id:
        Grouping key for leakage-free splitting. When a source does not track
        patients, adapters fall back to the per-image stem so each image forms its
        own singleton group (documented, never silently merged).
    visit_index:
        0-based longitudinal visit index for the patient. ``0`` when a source is
        not longitudinal.
    source:
        Dataset identifier, e.g. ``"azh"``, ``"fuseg"``, ``"piid"``.
    license:
        Short license/usage tag copied from the source (e.g. ``"research-only"``).
        This is metadata for hygiene tracking, not legal advice.
    mask_kind:
        How ``mask_path`` should be interpreted in a multiclass tissue run:
        ``"tissue"`` when pixel values are unified tissue-class indices, or
        ``"binary"`` when the mask is only wound-vs-background (foreground pixels
        of unknown tissue class). Binary masks supply *partial-label* supervision:
        their foreground drives localization via the segmentation loss's superset
        term without inventing a tissue class. Defaults to ``"binary"`` — the
        clinically safe assumption, since a wound mask never implies a class.
    """

    image_path: str
    mask_path: str
    body_part: str
    stage_label: str
    patient_id: str
    visit_index: int
    source: str
    license: str
    mask_kind: str = "binary"

    @classmethod
    def column_names(cls) -> list[str]:
        """Return the manifest column order (stable for CSV headers)."""
        return [f.name for f in fields(cls)]

    def to_row(self) -> dict[str, str]:
        """Serialize to a flat string dict for CSV writing."""
        row = asdict(self)
        row["visit_index"] = str(self.visit_index)
        return {key: str(value) for key, value in row.items()}

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> ManifestRecord:
        """Parse a CSV/JSONL row back into a record (tolerant of missing cols)."""
        return cls(
            image_path=str(row.get("image_path", "")),
            mask_path=str(row.get("mask_path", "")),
            body_part=str(row.get("body_part", "") or "unknown"),
            stage_label=str(row.get("stage_label", "") or ""),
            patient_id=str(row.get("patient_id", "")),
            visit_index=int(row.get("visit_index", 0) or 0),
            source=str(row.get("source", "") or "unknown"),
            license=str(row.get("license", "") or "unknown"),
            mask_kind=str(row.get("mask_kind", "") or "binary"),
        )


def write_manifest(
    records: Sequence[ManifestRecord],
    path: Path,
    *,
    also_csv: bool = True,
) -> list[Path]:
    """Write ``records`` to ``path`` as JSONL and (optionally) a sibling CSV.

    ``path`` should end in ``.jsonl``; the CSV is written next to it with a
    ``.csv`` suffix. Returns the list of files written.
    """
    path = path.expanduser()
    path.parent.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(json.dumps(asdict(record), ensure_ascii=False) + "\n")
    written.append(path)

    if also_csv:
        csv_path = path.with_suffix(".csv")
        with csv_path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=ManifestRecord.column_names())
            writer.writeheader()
            for record in records:
                writer.writerow(record.to_row())
        written.append(csv_path)

    return written


def read_manifest(path: Path) -> list[ManifestRecord]:
    """Read a manifest from a ``.jsonl`` or ``.csv`` file into records."""
    path = path.expanduser()
    if not path.is_file():
        raise FileNotFoundError(f"manifest not found: {path}")

    records: list[ManifestRecord] = []
    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                records.append(ManifestRecord.from_row(row))
    else:
        with path.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if line:
                    records.append(ManifestRecord.from_row(json.loads(line)))
    return records


# --------------------------------------------------------------------------- #
# Pairing helpers
# --------------------------------------------------------------------------- #
def _index_by_stem(directory: Path) -> dict[str, Path]:
    """Map file stem -> path for image-like files directly under ``directory``."""
    index: dict[str, Path] = {}
    if not directory.is_dir():
        return index
    for item in sorted(directory.iterdir()):
        if item.is_file() and item.suffix.lower() in IMAGE_SUFFIXES:
            # First writer wins; deterministic because iterdir() is sorted.
            index.setdefault(item.stem, item)
    return index


def _relativize(path: Path, root: Path | None) -> str:
    """Return ``path`` relative to ``root`` when possible, else its POSIX string."""
    resolved = path.resolve()
    if root is not None:
        try:
            return resolved.relative_to(root.resolve()).as_posix()
        except ValueError:
            pass
    return resolved.as_posix()


# --------------------------------------------------------------------------- #
# Adapters
# --------------------------------------------------------------------------- #
# Directory names commonly used for images / masks across public wound sets.
_IMAGE_DIR_NAMES: tuple[str, ...] = ("images", "image", "img", "imgs")
_MASK_DIR_NAMES: tuple[str, ...] = ("labels", "label", "masks", "mask", "annotations", "gt")


def _find_child(parent: Path, candidates: Iterable[str]) -> Path | None:
    """Return the first existing child directory whose name matches (case-insensitive)."""
    if not parent.is_dir():
        return None
    lowered = {c.lower() for c in candidates}
    for child in sorted(parent.iterdir()):
        if child.is_dir() and child.name.lower() in lowered:
            return child
    return None


def scan_generic(
    root: Path,
    *,
    source: str,
    license: str,
    body_part: str = "unknown",
    root_for_paths: Path | None = None,
    require_masks: bool = True,
    mask_kind: str = "binary",
) -> list[ManifestRecord]:
    """Adapter for a flat ``root/images`` + ``root/masks`` layout.

    Images and masks are paired by filename stem. Because a generic dump carries
    no patient tracking, each image becomes its own singleton ``patient_id`` (the
    stem), so :func:`split_by_patient` cannot leak — but note there is then no
    real cross-visit grouping to protect. Longitudinal sources should use a
    dedicated adapter or a pre-built manifest that fills ``patient_id`` properly.
    """
    root = root.expanduser()
    images_dir = _find_child(root, _IMAGE_DIR_NAMES)
    masks_dir = _find_child(root, _MASK_DIR_NAMES)
    if images_dir is None:
        raise FileNotFoundError(
            f"no image directory (one of {_IMAGE_DIR_NAMES}) under {root}"
        )

    masks_by_stem = _index_by_stem(masks_dir) if masks_dir is not None else {}
    records: list[ManifestRecord] = []
    for stem, image_path in _index_by_stem(images_dir).items():
        mask_path = masks_by_stem.get(stem)
        if mask_path is None and require_masks:
            continue
        records.append(
            ManifestRecord(
                image_path=_relativize(image_path, root_for_paths),
                mask_path=_relativize(mask_path, root_for_paths) if mask_path else "",
                body_part=body_part,
                stage_label="",
                patient_id=f"{source}:{stem}",
                visit_index=0,
                source=source,
                license=license,
                mask_kind=mask_kind,
            )
        )
    return records


def scan_azh_fuseg(
    root: Path,
    *,
    source: str = "azh_fuseg",
    license: str = "research-only",
    root_for_paths: Path | None = None,
) -> list[ManifestRecord]:
    """Adapter for the AZH / FUSeg foot-ulcer layout.

    The github.com/uwm-bigdata/wound-segmentation release organizes data as
    per-split folders, each holding an image and a label directory::

        root/
          train/       (images/ + labels/)
          validation/  (images/ + labels/)   # a.k.a. "val"
          test/        (images/ + labels/)    # FUSeg test labels are held out

    This adapter walks whichever of those split folders exist and also tolerates a
    flat ``root/images`` + ``root/labels`` layout (falling back to
    :func:`scan_generic`). All samples are ``body_part="foot"``. AZH/FUSeg do not
    publish per-patient identifiers, so each image is its own singleton group; the
    original ``train``/``validation``/``test`` partition is recorded via a
    ``source`` suffix (e.g. ``azh_fuseg/train``) so it can be honoured or ignored
    downstream.
    """
    root = root.expanduser()
    split_dir_names = {
        SPLIT_TRAIN: ("train", "training"),
        SPLIT_VAL: ("validation", "val", "valid"),
        SPLIT_TEST: ("test", "testing"),
    }

    records: list[ManifestRecord] = []
    found_split = False
    for split, names in split_dir_names.items():
        split_dir = _find_child(root, names)
        if split_dir is None:
            continue
        found_split = True
        records.extend(
            scan_generic(
                split_dir,
                source=f"{source}/{split}",
                license=license,
                body_part="foot",
                root_for_paths=root_for_paths,
                # FUSeg's public test split ships images without labels.
                require_masks=split != SPLIT_TEST,
            )
        )

    if not found_split:
        # Flat layout fallback.
        records = scan_generic(
            root,
            source=source,
            license=license,
            body_part="foot",
            root_for_paths=root_for_paths,
            require_masks=True,
        )

    if not records:
        raise FileNotFoundError(
            f"no image/mask pairs discovered under {root}; expected AZH/FUSeg "
            "'train|validation|test' folders each with images/ + labels/, or a "
            "flat images/ + labels/ layout"
        )
    return records


ADAPTERS = {
    "azh_fuseg": scan_azh_fuseg,
    "generic": scan_generic,
}


# --------------------------------------------------------------------------- #
# Patient-level splitting (no leakage across visits)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True, slots=True)
class SplitRatios:
    """Fractional sizes for train/val/test. Must sum to 1 (within tolerance)."""

    train: float = 0.7
    val: float = 0.15
    test: float = 0.15

    def validate(self) -> None:
        total = self.train + self.val + self.test
        if not (0.999 <= total <= 1.001):
            raise ValueError(f"split ratios must sum to 1.0, got {total:.4f}")
        if min(self.train, self.val, self.test) < 0:
            raise ValueError("split ratios must be non-negative")


def _patient_stratum(records: Sequence[ManifestRecord], key: str) -> str:
    """Derive one stratification bucket for a patient from its records.

    Uses the most common non-empty value of ``key`` across the patient's records
    (e.g. dominant ``stage_label`` or ``body_part``); ``""`` when all are empty.
    """
    counts: dict[str, int] = defaultdict(int)
    for record in records:
        value = getattr(record, key, "") or ""
        if value:
            counts[value] += 1
    if not counts:
        return ""
    return max(counts, key=lambda value: (counts[value], value))


def split_by_patient(
    records: Sequence[ManifestRecord],
    *,
    ratios: SplitRatios | None = None,
    seed: int = 1234,
    stratify_by: str | None = None,
) -> dict[str, list[ManifestRecord]]:
    """Split ``records`` into train/val/test grouped by ``patient_id``.

    A patient (and therefore all its visits) is assigned to exactly one split, so
    no visit of a patient leaks across the train/val/test boundary. When
    ``stratify_by`` is given (``"stage_label"`` or ``"body_part"``), patients are
    bucketed by their dominant value and each bucket is split independently so the
    class balance is preserved across splits.

    The assignment is deterministic for a fixed ``seed`` and record set.
    """
    ratios = ratios or SplitRatios()
    ratios.validate()

    # Group records by patient.
    by_patient: dict[str, list[ManifestRecord]] = defaultdict(list)
    for record in records:
        by_patient[record.patient_id].append(record)

    # Bucket patients by stratum (single "" bucket when not stratifying).
    strata: dict[str, list[str]] = defaultdict(list)
    for patient_id, patient_records in by_patient.items():
        bucket = _patient_stratum(patient_records, stratify_by) if stratify_by else ""
        strata[bucket].append(patient_id)

    out: dict[str, list[ManifestRecord]] = {split: [] for split in SPLITS}
    rng = random.Random(seed)

    for _bucket, patient_ids in sorted(strata.items()):
        ordered = sorted(patient_ids)
        rng.shuffle(ordered)
        n = len(ordered)
        n_train = round(n * ratios.train)
        n_val = round(n * ratios.val)
        # Guard rounding so the three counts always cover every patient.
        n_train = min(n_train, n)
        n_val = min(n_val, n - n_train)
        assignment = (
            [SPLIT_TRAIN] * n_train
            + [SPLIT_VAL] * n_val
            + [SPLIT_TEST] * (n - n_train - n_val)
        )
        for patient_id, split in zip(ordered, assignment, strict=True):
            out[split].extend(by_patient[patient_id])

    return out


def assert_no_patient_leakage(splits: dict[str, list[ManifestRecord]]) -> None:
    """Raise ``AssertionError`` if any ``patient_id`` appears in two splits."""
    seen: dict[str, str] = {}
    for split, records in splits.items():
        for record in records:
            prior = seen.get(record.patient_id)
            if prior is not None and prior != split:
                raise AssertionError(
                    f"patient {record.patient_id!r} leaks across "
                    f"{prior!r} and {split!r}"
                )
            seen[record.patient_id] = split


# --------------------------------------------------------------------------- #
# Augmentation + colour normalization (albumentations)
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class AugmentationConfig:
    """Knobs for the albumentations pipelines.

    Defaults follow the architecture brief: 768-px inputs, ImageNet normalization,
    and photometric jitter that is safe for wound colour semantics (mild — heavy
    hue shifts would corrupt the very tissue colours downstream heads read).
    """

    input_size: int = 768
    hflip_prob: float = 0.5
    vflip_prob: float = 0.5
    rotate_limit: int = 20
    rotate_prob: float = 0.5
    brightness_contrast_prob: float = 0.5
    brightness_limit: float = 0.2
    contrast_limit: float = 0.2
    hue_shift_limit: int = 6
    sat_shift_limit: int = 12
    val_shift_limit: int = 8
    hue_sat_prob: float = 0.3
    gauss_noise_prob: float = 0.15
    # Colour normalization: gray-world white-balance before ImageNet standardize.
    gray_world: bool = True
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any] | None) -> AugmentationConfig:
        """Build from a config dict, ignoring unknown keys."""
        if not mapping:
            return cls()
        known = {f.name for f in fields(cls)}
        kwargs = {k: v for k, v in mapping.items() if k in known}
        for tup in ("mean", "std"):
            if tup in kwargs and kwargs[tup] is not None:
                kwargs[tup] = tuple(float(x) for x in kwargs[tup])
        return cls(**kwargs)


def _gray_world_uint8(image: np.ndarray, **_: Any) -> np.ndarray:
    """Gray-world white-balance for a uint8 RGB image (albumentations Lambda).

    Reuses :func:`sorbed.preprocess.color_norm.gray_world_normalize` when Sorbed is
    importable, and otherwise applies the identical gray-world rule locally so this
    module never hard-depends on the package being installed.
    """
    import numpy as np

    rgb = image.astype(np.float32) / 255.0
    try:
        from sorbed.preprocess.color_norm import gray_world_normalize

        balanced = gray_world_normalize(rgb)
    except Exception:
        means = rgb.reshape(-1, 3).mean(axis=0)
        gray = float(means.mean())
        scale = np.where(means > 1e-4, gray / means, 1.0).astype(np.float32)
        balanced = np.clip(rgb * scale, 0.0, 1.0).astype(np.float32)
    out: np.ndarray = np.clip(balanced * 255.0, 0, 255).astype(np.uint8)
    return out


def build_train_transform(cfg: AugmentationConfig) -> Any:
    """Return the training albumentations ``Compose`` (image + mask)."""
    import albumentations as A  # noqa: N812 - conventional albumentations alias

    steps: list[Any] = [
        A.Resize(cfg.input_size, cfg.input_size),
        A.HorizontalFlip(p=cfg.hflip_prob),
        A.VerticalFlip(p=cfg.vflip_prob),
        A.Rotate(limit=cfg.rotate_limit, p=cfg.rotate_prob, border_mode=0),
        A.RandomBrightnessContrast(
            brightness_limit=cfg.brightness_limit,
            contrast_limit=cfg.contrast_limit,
            p=cfg.brightness_contrast_prob,
        ),
        A.HueSaturationValue(
            hue_shift_limit=cfg.hue_shift_limit,
            sat_shift_limit=cfg.sat_shift_limit,
            val_shift_limit=cfg.val_shift_limit,
            p=cfg.hue_sat_prob,
        ),
        A.GaussNoise(p=cfg.gauss_noise_prob),
    ]
    if cfg.gray_world:
        steps.append(A.Lambda(image=_gray_world_uint8, name="gray_world"))
    steps.append(A.Normalize(mean=cfg.mean, std=cfg.std))
    return A.Compose(steps)


def build_val_transform(cfg: AugmentationConfig) -> Any:
    """Return the deterministic val/test albumentations ``Compose``.

    No geometric/photometric jitter — only resize, the same colour normalization
    used at train time, and ImageNet standardization. Keeping normalization
    identical across train and inference is what makes the numbers comparable.
    """
    import albumentations as A  # noqa: N812

    steps: list[Any] = [A.Resize(cfg.input_size, cfg.input_size)]
    if cfg.gray_world:
        steps.append(A.Lambda(image=_gray_world_uint8, name="gray_world"))
    steps.append(A.Normalize(mean=cfg.mean, std=cfg.std))
    return A.Compose(steps)


# --------------------------------------------------------------------------- #
# Torch dataset
# --------------------------------------------------------------------------- #
class WoundSegmentationDataset(_TorchDataset):
    """A ``torch.utils.data.Dataset`` of ``(image, mask)`` tensors.

    ``records`` are manifest rows; ``root`` is prepended to relative paths. The
    ``transform`` is an albumentations ``Compose`` (see :func:`build_train_transform`);
    when ``None``, a deterministic val transform is built from ``aug_config``.

    Each item is a CHW float32 image tensor (normalized) and a ``(1, H, W)``
    float32 binary mask tensor (wound = 1.0) — the exact shapes the U-Net loop in
    ``scripts/train_segmenter.py`` consumes. Records without a mask raise at access
    time; filter them out (``require_masks``) before constructing the dataset.
    """

    def __init__(
        self,
        records: Sequence[ManifestRecord],
        *,
        root: Path | None = None,
        transform: Any | None = None,
        aug_config: AugmentationConfig | None = None,
    ) -> None:
        self._records = list(records)
        self._root = root.expanduser() if root is not None else None
        self._cfg = aug_config or AugmentationConfig()
        self._transform = transform if transform is not None else build_val_transform(self._cfg)

    def __len__(self) -> int:
        return len(self._records)

    def _resolve(self, rel: str) -> Path:
        path = Path(rel)
        if not path.is_absolute() and self._root is not None:
            path = self._root / path
        return path

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        import cv2
        import numpy as np
        import torch

        record = self._records[index]
        image_path = self._resolve(record.image_path)
        bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"could not read image {image_path}")
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)

        if not record.mask_path:
            raise RuntimeError(
                f"record {record.image_path!r} has no mask; filter unlabeled "
                "records out before building a segmentation dataset"
            )
        mask_path = self._resolve(record.mask_path)
        mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask_raw is None:
            raise RuntimeError(f"could not read mask {mask_path}")
        mask = (mask_raw > 0).astype(np.uint8)

        out = self._transform(image=rgb, mask=mask)
        image_np = np.ascontiguousarray(np.transpose(out["image"], (2, 0, 1)))
        mask_np = np.ascontiguousarray(out["mask"].astype(np.float32))[np.newaxis, ...]
        return torch.from_numpy(image_np), torch.from_numpy(mask_np)


# --------------------------------------------------------------------------- #
# Source registry (for config-driven scanning)
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class SourceSpec:
    """One dataset source declared in a data-prep config file."""

    name: str
    root: Path
    adapter: str = "generic"
    body_part: str = "unknown"
    license: str = "unknown"
    require_masks: bool = True
    mask_kind: str = "binary"

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> SourceSpec:
        if "name" not in mapping or "root" not in mapping:
            raise ValueError("each source needs at least 'name' and 'root'")
        adapter = str(mapping.get("adapter", "generic"))
        if adapter not in ADAPTERS:
            raise ValueError(
                f"unknown adapter {adapter!r} for source {mapping['name']!r}; "
                f"choose one of {sorted(ADAPTERS)}"
            )
        mask_kind = str(mapping.get("mask_kind", "binary")).lower()
        if mask_kind not in ("binary", "tissue"):
            raise ValueError(
                f"source {mapping['name']!r}: mask_kind must be 'binary' or "
                f"'tissue', got {mask_kind!r}"
            )
        return cls(
            name=str(mapping["name"]),
            root=Path(str(mapping["root"])).expanduser(),
            adapter=adapter,
            body_part=str(mapping.get("body_part", "unknown")),
            license=str(mapping.get("license", "unknown")),
            require_masks=bool(mapping.get("require_masks", True)),
            mask_kind=mask_kind,
        )

    def scan(self, root_for_paths: Path | None) -> list[ManifestRecord]:
        """Run the adapter for this source."""
        if self.adapter == "azh_fuseg":
            # AZH/FUSeg is foot-ulcer wound-vs-background only (always binary).
            return scan_azh_fuseg(
                self.root,
                source=self.name,
                license=self.license,
                root_for_paths=root_for_paths,
            )
        return scan_generic(
            self.root,
            source=self.name,
            license=self.license,
            body_part=self.body_part,
            root_for_paths=root_for_paths,
            require_masks=self.require_masks,
            mask_kind=self.mask_kind,
        )


@dataclass(slots=True)
class DataPrepConfig:
    """Parsed ``training/configs/*.yaml`` for :mod:`training.data_prep`."""

    sources: list[SourceSpec] = field(default_factory=list)
    ratios: SplitRatios = field(default_factory=SplitRatios)
    seed: int = 1234
    stratify_by: str | None = None
    augmentation: AugmentationConfig = field(default_factory=AugmentationConfig)

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> DataPrepConfig:
        sources = [SourceSpec.from_mapping(s) for s in mapping.get("sources", [])]
        split_map = mapping.get("split", {}) or {}
        ratios = SplitRatios(
            train=float(split_map.get("train", 0.7)),
            val=float(split_map.get("val", 0.15)),
            test=float(split_map.get("test", 0.15)),
        )
        stratify = split_map.get("stratify_by")
        if stratify in ("", None, "none"):
            stratify = None
        return cls(
            sources=sources,
            ratios=ratios,
            seed=int(mapping.get("seed", split_map.get("seed", 1234))),
            stratify_by=stratify,
            augmentation=AugmentationConfig.from_mapping(mapping.get("augmentation")),
        )

    @classmethod
    def load(cls, path: Path | str) -> DataPrepConfig:
        """Load a YAML config (PyYAML is a Sorbed core dependency)."""
        import yaml

        with Path(path).expanduser().open("r", encoding="utf-8") as handle:
            data = yaml.safe_load(handle) or {}
        if not isinstance(data, dict):
            raise ValueError(f"config root must be a mapping, got {type(data).__name__}")
        return cls.from_mapping(data)

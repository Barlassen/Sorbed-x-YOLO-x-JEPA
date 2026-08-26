"""Append-only, locked JSONL store for inferences and human feedback.

:class:`FeedbackStore` owns the on-disk layout (two JSONL logs plus a mask/
correction PNG tree) and the atomic-append primitives. All writes go through a
cross-process advisory lock and are ``fsync``-ed; readers tolerate a torn last
line (skip-on-parse-error) so a reader concurrent with a partial write never
crashes. Nothing here imports torch or any training code — this ships in the
CPU-only inference container.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import TypeVar

import numpy as np
from PIL import Image

from sorbed.feedback.records import FeedbackRecord, InferenceRecord

_RecordT = TypeVar("_RecordT", InferenceRecord, FeedbackRecord)

try:  # POSIX advisory locking; degrades gracefully where unavailable.
    import fcntl

    _HAVE_FCNTL = True
except ImportError:  # pragma: no cover - non-POSIX fallback
    _HAVE_FCNTL = False

_ENV_ROOT = "SORBED_FEEDBACK_DIR"
_DEFAULT_ROOT = "feedback"

# Reviewer seniority for conflict resolution. Physician and HQ are the senior
# tier (a wound-care lead / headquarters adjudicator); nurse is the front-line
# tier. A higher rank is a hard floor — a later review from a lower tier cannot
# overwrite it. Unknown roles rank below all named tiers.
_ROLE_RANK: dict[str, int] = {"physician": 3, "hq": 3, "nurse": 1}


def _feedback_precedence(fb: FeedbackRecord) -> tuple[int, str]:
    """Sort key for conflict resolution: (role rank, then recency within a tier)."""
    rank = _ROLE_RANK.get(fb.reviewer_role.strip().lower(), 0)
    return (rank, fb.created_at)


class FeedbackStore:
    """Filesystem-backed feedback store rooted at a single directory."""

    def __init__(self, root: str | Path | None = None) -> None:
        """Resolve the store root from ``root``, ``$SORBED_FEEDBACK_DIR``, or the default.

        The root is treated as a PHI zone: the JSONL logs are a low-sensitivity
        index while the mask/correction PNGs are the sensitive payload.
        """
        resolved = root or os.environ.get(_ENV_ROOT) or _DEFAULT_ROOT
        self.root = Path(resolved)
        self.inferences_path = self.root / "inferences.jsonl"
        self.feedback_path = self.root / "feedback.jsonl"
        self.masks_dir = self.root / "masks"
        self.corrections_dir = self.root / "corrections"
        self._lock_path = self.root / ".lock"
        self._ensure_dirs()

    def _ensure_dirs(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        self.masks_dir.mkdir(parents=True, exist_ok=True)
        self.corrections_dir.mkdir(parents=True, exist_ok=True)

    @contextlib.contextmanager
    def _locked(self) -> Iterator[None]:
        """Hold an exclusive advisory lock over the store for the block."""
        with open(self._lock_path, "w", encoding="utf-8") as handle:
            if _HAVE_FCNTL:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                if _HAVE_FCNTL:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _append_line(self, path: Path, line: str) -> None:
        with self._locked(), open(path, "a", encoding="utf-8") as handle:
            handle.write(line + "\n")
            handle.flush()
            os.fsync(handle.fileno())

    def append_inference(self, record: InferenceRecord) -> None:
        """Append one :class:`InferenceRecord` to ``inferences.jsonl``."""
        self._append_line(self.inferences_path, record.to_jsonl_line())

    def append_feedback(self, record: FeedbackRecord) -> None:
        """Append one :class:`FeedbackRecord` to ``feedback.jsonl``."""
        self._append_line(self.feedback_path, record.to_jsonl_line())

    def save_mask_png(
        self,
        mask: np.ndarray,
        record_id: str,
        *,
        when: datetime | None = None,
    ) -> str:
        """Persist a predicted mask as a ``{0,255}`` L-mode PNG.

        The file is partitioned ``masks/<yyyy>/<mm>/<record_id>.png``; the
        returned path is *relative to the store root* (what goes in the record).
        """
        stamp = when or datetime.now(UTC)
        rel = Path("masks") / f"{stamp.year:04d}" / f"{stamp.month:02d}" / f"{record_id}.png"
        self._write_binary_mask(mask, self.root / rel)
        return rel.as_posix()

    def save_correction_png(self, mask: np.ndarray, record_id: str) -> str:
        """Persist a human-fixed mask as ``corrections/<record_id>.png``; return rel path."""
        rel = Path("corrections") / f"{record_id}.png"
        self._write_binary_mask(mask, self.root / rel)
        return rel.as_posix()

    @staticmethod
    def _write_binary_mask(mask: np.ndarray, dest: Path) -> None:
        binary = (np.asarray(mask) > 0).astype(np.uint8) * 255
        if binary.ndim != 2:
            raise ValueError(f"mask must be 2-D (H, W); got shape {binary.shape}")
        dest.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(binary, mode="L").save(dest, format="PNG")

    def iter_inferences(self) -> Iterator[InferenceRecord]:
        """Yield every parseable inference row, skipping a torn final line."""
        yield from _iter_records(self.inferences_path, InferenceRecord)

    def iter_feedback(self) -> Iterator[FeedbackRecord]:
        """Yield every parseable feedback row, skipping a torn final line."""
        yield from _iter_records(self.feedback_path, FeedbackRecord)

    def latest_feedback_by_record(self) -> dict[str, FeedbackRecord]:
        """Map ``record_id`` to its authoritative feedback row by ROLE precedence.

        A senior reviewer's label is a hard floor: a physician/HQ correction is
        never overwritten by a later nurse review (recency only breaks ties within
        the same role tier). This matches the contract in ``records.py`` — resolve
        conflicts by role precedence — and prevents a late low-tier review from
        silently corrupting an adjudicated label.
        """
        latest: dict[str, FeedbackRecord] = {}
        for fb in self.iter_feedback():
            current = latest.get(fb.record_id)
            if current is None or _feedback_precedence(fb) >= _feedback_precedence(current):
                latest[fb.record_id] = fb
        return latest

    def join_pairs(self) -> Iterator[tuple[InferenceRecord, FeedbackRecord | None]]:
        """Yield ``(inference, newest_feedback_or_None)`` for each inference row."""
        latest = self.latest_feedback_by_record()
        for inf in self.iter_inferences():
            yield inf, latest.get(inf.record_id)


def _iter_records(path: Path, model: type[_RecordT]) -> Iterator[_RecordT]:
    if not path.exists():
        return
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            stripped = line.strip()
            if not stripped:
                continue
            try:
                yield model.from_jsonl_line(stripped)
            except ValueError:
                # A torn last line from a concurrent write, or a legacy off-schema
                # row: skip it rather than fail the whole read.
                continue

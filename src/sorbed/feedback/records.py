"""On-disk feedback contract — the single source of truth for both logs.

These pydantic v2 models mirror the canonical append-only JSONL schema exactly.
They are imported by the runtime emitter (``sorbed.feedback``) that *writes* the
rows and, later, by the offline continual trainer that *reads* them, so the
contract cannot drift between producer and consumer. The module depends only on
``pydantic`` — no torch, numpy, or PIL — so it stays importable inside the
CPU-only inference wheel.

``File A`` is :class:`InferenceRecord` (``inferences.jsonl``); ``File B`` is
:class:`FeedbackRecord` (``feedback.jsonl``). Both forbid extra fields so a typo
in a producer fails loudly instead of silently minting an off-schema row.
"""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field

SCHEMA_VERSION = "1.0.0"


class ModelVersions(BaseModel):
    """Which checkpoints produced a row — ties every label to its origin.

    ``segmenter``/``grader`` are checkpoint sha256s, or the sentinels
    ``"classical"``/``"rule"`` when a weight-free backend produced the output.
    This provenance is what lets the continual trainer detect and cap
    self-generated labels (the feedback-loop-bias guard).
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    segmenter: str
    grader: str


class InferenceRecord(BaseModel):
    """File A — one analysis result, appended to ``inferences.jsonl``.

    Contains no raw PHI: ``image_ref``/``patient_ref``/``body_part`` are opaque
    caller-supplied tokens, and the pixels live out-of-band under the store's
    PHI zone. ``image_sha256`` is a content hash for dedupe and audit only.
    """

    model_config = ConfigDict(extra="forbid")

    record_id: str = Field(description="uuid4 hex, generated at emit time (primary key).")
    created_at: str = Field(description="ISO-8601 UTC timestamp.")
    image_ref: str = Field(description="Opaque path or stable id of the analysed photo.")
    image_sha256: str = Field(description="SHA-256 of the image bytes, for dedupe/audit.")
    patient_ref: str | None = None
    body_part: str | None = None
    predicted_stage: str = Field(description="PressureInjuryStage value, or '' when abstained.")
    confidence: float
    abstained: bool
    mask_path: str = Field(description="Store-relative path to the predicted mask PNG (0/255).")
    tissue_fractions: dict[str, float] = Field(default_factory=dict)
    area_cm2: float | None = None
    push_total: int | None = None
    stats: dict[str, Any] = Field(default_factory=dict, description="Free-form numeric stats blob.")
    model_versions: ModelVersions

    def to_jsonl_line(self) -> str:
        """Serialize to a single JSON line (no trailing newline)."""
        return self.model_dump_json()

    @classmethod
    def from_jsonl_line(cls, line: str) -> InferenceRecord:
        """Parse one JSON line into an :class:`InferenceRecord`."""
        return cls.model_validate_json(line)


class FeedbackRecord(BaseModel):
    """File B — one human review, appended to ``feedback.jsonl``.

    ``record_id`` is a foreign key into ``inferences.jsonl``. Multiple reviews of
    the same analysis are allowed (e.g. nurse then HQ); rows are never rewritten,
    and the harvester resolves conflicts by role precedence.
    """

    model_config = ConfigDict(extra="forbid")

    record_id: str = Field(description="FK to an inferences.jsonl row.")
    created_at: str = Field(description="ISO-8601 UTC timestamp.")
    reviewer_role: str = Field(description="'nurse' | 'hq' | 'physician'.")
    reviewer_id: str = Field(description="Pseudonymous staff id, never a name.")
    agree: bool = Field(description="Did the human accept the model's stage?")
    corrected_stage: str | None = Field(default=None, description="Human stage if disagreeing.")
    corrected_mask_path: str | None = Field(default=None, description="Optional human-fixed mask.")
    notes: str = ""
    source: str = Field(default="human", description="'human' here; room for 'rule'/'fl-peer'.")

    def to_jsonl_line(self) -> str:
        """Serialize to a single JSON line (no trailing newline)."""
        return self.model_dump_json()

    @classmethod
    def from_jsonl_line(cls, line: str) -> FeedbackRecord:
        """Parse one JSON line into a :class:`FeedbackRecord`."""
        return cls.model_validate_json(line)

"""Turn a completed analysis into one persisted :class:`InferenceRecord`.

``emit_inference`` reads a pipeline :class:`~sorbed.pipeline.analyzer.AnalysisBundle`,
writes its predicted mask PNG, and appends a single inference row. It derives
every field from the real analysis object (no fabricated values) and populates a
free-form ``stats`` blob for features that must not force a schema migration.

Emission is meant to be *best-effort* at the call site — a feedback-store failure
must never fail a clinical analysis — so callers should wrap this in
try/except/log-and-continue. This function itself performs the work and returns
the new ``record_id``.
"""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from uuid import uuid4

from sorbed.feedback.records import SCHEMA_VERSION, InferenceRecord, ModelVersions
from sorbed.feedback.store import FeedbackStore

if TYPE_CHECKING:
    from sorbed.domain.analysis import WoundAnalysis
    from sorbed.pipeline.analyzer import AnalysisBundle


def emit_inference(
    bundle: AnalysisBundle,
    store: FeedbackStore,
    *,
    patient_ref: str | None = None,
    body_part: str | None = None,
    image_ref: str | None = None,
    image_bytes: bytes | None = None,
) -> str:
    """Persist ``bundle`` as an :class:`InferenceRecord` and return its ``record_id``.

    ``image_sha256`` is computed from ``image_bytes`` when supplied (the source
    photo the pipeline decoded); otherwise it falls back to the normalized-pixel
    content hash already carried in the analysis metadata. ``image_ref`` and
    ``patient_ref`` must be opaque tokens, never raw PHI.
    """
    analysis: WoundAnalysis = bundle.analysis
    decision = analysis.decision
    record_id = uuid4().hex
    now = datetime.now(UTC)

    mask_path = store.save_mask_png(bundle.wound_mask, record_id, when=now)

    if image_bytes is not None:
        image_sha256 = hashlib.sha256(image_bytes).hexdigest()
    else:
        image_sha256 = analysis.image.pixel_sha256

    record = InferenceRecord(
        record_id=record_id,
        created_at=now.isoformat(),
        image_ref=image_ref or "",
        image_sha256=image_sha256,
        patient_ref=patient_ref,
        body_part=body_part,
        predicted_stage="" if decision.abstained else decision.stage.value,
        confidence=decision.confidence,
        abstained=decision.abstained,
        mask_path=mask_path,
        tissue_fractions=_tissue_fractions(analysis),
        area_cm2=analysis.metrics.geometry.area_cm2,
        push_total=_push_total(analysis),
        stats=_stats_blob(analysis),
        model_versions=_model_versions(analysis),
    )
    store.append_inference(record)
    return record_id


def _tissue_fractions(analysis: WoundAnalysis) -> dict[str, float]:
    """Wound-bed area fractions keyed by :class:`TissueClass` value strings."""
    return {cls.value: float(frac) for cls, frac in analysis.metrics.tissue.fractions.items()}


def _push_total(analysis: WoundAnalysis) -> int | None:
    healing = analysis.metrics.healing_scores
    return healing.push_partial_total if healing is not None else None


def _stats_blob(analysis: WoundAnalysis) -> dict[str, object]:
    """Free-form numeric blob — the extension point that avoids schema churn."""
    metrics = analysis.metrics
    healing = metrics.healing_scores
    blob: dict[str, object] = {
        "schema_version": SCHEMA_VERSION,
        "skin_tone_band": analysis.skin_tone_band.value,
        "config_digest": analysis.config_digest,
        "stage_timings_ms": dict(analysis.stage_timings_ms),
        "color_cues": metrics.color_cues.model_dump(mode="json"),
    }
    if healing is not None:
        blob["push_size_subscore"] = healing.push_size_subscore
        blob["push_tissue_subscore"] = healing.push_tissue_subscore
        blob["granulation_percent"] = healing.granulation_percent
    if metrics.depth_proxy is not None:
        blob["depth_proxy"] = metrics.depth_proxy.model_dump(mode="json")
    if metrics.periwound is not None:
        blob["periwound"] = metrics.periwound.model_dump(mode="json")
    return blob


def _model_versions(analysis: WoundAnalysis) -> ModelVersions:
    """Resolve segmenter/grader checkpoint shas, or the weight-free sentinels."""
    provenance = analysis.provenance
    weights = provenance.weights_sha256
    segmenter = weights.get(provenance.segmentation_backend) or "classical"
    grader = weights.get(provenance.staging_backend) or "rule"
    return ModelVersions(segmenter=segmenter, grader=grader)

"""Runtime feedback-store tests — round-trip, store join, emit, and export.

All data is real: records are built and re-parsed, and ``emit_inference`` runs on
the shared ``granulating_analysis`` fixture. No network, no torch.
"""

from __future__ import annotations

import csv
import hashlib
from types import SimpleNamespace

import numpy as np
from typer.testing import CliRunner

from sorbed.cli.feedback_cmd import feedback_app
from sorbed.feedback import (
    FeedbackRecord,
    FeedbackStore,
    InferenceRecord,
    ModelVersions,
    emit_inference,
)

runner = CliRunner()


def _inference(record_id: str = "rec1", stage: str = "stage_3") -> InferenceRecord:
    return InferenceRecord(
        record_id=record_id,
        created_at="2026-07-11T00:00:00+00:00",
        image_ref=f"opaque/{record_id}",
        image_sha256="a" * 64,
        patient_ref="mrn-token",
        body_part="sacrum",
        predicted_stage=stage,
        confidence=0.71,
        abstained=False,
        mask_path="masks/2026/07/rec1.png",
        tissue_fractions={"granulation": 0.8, "slough": 0.2},
        area_cm2=3.4,
        push_total=6,
        stats={"schema_version": "1.0.0", "skin_tone_band": "fitzpatrick_i_iii"},
        model_versions=ModelVersions(segmenter="classical", grader="rule"),
    )


def _feedback(record_id: str = "rec1", **overrides: object) -> FeedbackRecord:
    base: dict[str, object] = {
        "record_id": record_id,
        "created_at": "2026-07-11T01:00:00+00:00",
        "reviewer_role": "hq",
        "reviewer_id": "staff-42",
        "agree": False,
        "corrected_stage": "stage_4",
        "corrected_mask_path": None,
        "notes": "muscle visible",
        "source": "human",
    }
    base.update(overrides)
    return FeedbackRecord(**base)  # type: ignore[arg-type]


def test_inference_record_jsonl_round_trip() -> None:
    rec = _inference()
    reparsed = InferenceRecord.from_jsonl_line(rec.to_jsonl_line())
    assert reparsed == rec
    assert reparsed.model_versions.grader == "rule"


def test_feedback_record_jsonl_round_trip() -> None:
    rec = _feedback()
    reparsed = FeedbackRecord.from_jsonl_line(rec.to_jsonl_line())
    assert reparsed == rec


def test_store_append_and_join(tmp_path) -> None:
    store = FeedbackStore(tmp_path)
    store.append_inference(_inference("recA"))
    store.append_inference(_inference("recB"))
    store.append_feedback(_feedback("recA"))

    pairs = {inf.record_id: fb for inf, fb in store.join_pairs()}
    assert set(pairs) == {"recA", "recB"}
    assert pairs["recA"] is not None
    assert pairs["recA"].corrected_stage == "stage_4"
    assert pairs["recB"] is None


def test_store_join_resolves_by_role_then_recency(tmp_path) -> None:
    # A senior HQ correction supersedes an earlier front-line nurse review.
    store = FeedbackStore(tmp_path)
    store.append_inference(_inference("recA"))
    store.append_feedback(_feedback("recA", created_at="2026-07-11T01:00:00+00:00",
                                    reviewer_role="nurse", corrected_stage=None, agree=True))
    store.append_feedback(_feedback("recA", created_at="2026-07-11T02:00:00+00:00"))

    (_, fb), = list(store.join_pairs())
    assert fb is not None
    assert fb.reviewer_role == "hq"
    assert fb.corrected_stage == "stage_4"


def test_store_tolerates_torn_line(tmp_path) -> None:
    store = FeedbackStore(tmp_path)
    store.append_inference(_inference("recA"))
    with open(store.inferences_path, "a", encoding="utf-8") as handle:
        handle.write('{"record_id": "torn"')  # no newline, truncated
    ids = [inf.record_id for inf in store.iter_inferences()]
    assert ids == ["recA"]


def test_emit_inference_writes_record_and_mask(tmp_path, granulating_analysis) -> None:
    store = FeedbackStore(tmp_path)
    mask = np.zeros((220, 300), dtype=bool)
    mask[80:140, 120:200] = True
    bundle = SimpleNamespace(analysis=granulating_analysis, wound_mask=mask)

    record_id = emit_inference(
        bundle,  # type: ignore[arg-type]
        store,
        patient_ref="mrn-token",
        image_ref="photo-1",
        image_bytes=b"raw-image-bytes",
    )

    records = list(store.iter_inferences())
    assert len(records) == 1
    rec = records[0]
    assert rec.record_id == record_id
    assert rec.patient_ref == "mrn-token"
    assert rec.image_sha256 == hashlib.sha256(b"raw-image-bytes").hexdigest()
    assert rec.tissue_fractions  # derived from the real analysis
    assert abs(sum(rec.tissue_fractions.values()) - 1.0) < 1e-3
    assert rec.model_versions.grader == "rule"
    assert rec.stats["schema_version"] == "1.0.0"
    assert (store.root / rec.mask_path).exists()


def test_export_human_label_overrides_model(tmp_path) -> None:
    store_root = tmp_path / "store"
    store = FeedbackStore(store_root)
    store.append_inference(_inference("recA", stage="stage_3"))
    # HQ disagrees and corrects up to Stage 4 — the informative gradient.
    store.append_feedback(_feedback("recA"))
    # A rule-only inference with no human review keeps model label + rule weight.
    store.append_inference(_inference("recB", stage="stage_2"))

    out = tmp_path / "manifest.csv"
    result = runner.invoke(
        feedback_app, ["export", "--root", str(store_root), "--out", str(out)]
    )
    assert result.exit_code == 0, result.output

    with open(out, encoding="utf-8") as handle:
        rows = {r["image"]: r for r in csv.DictReader(handle)}

    human = next(r for r in rows.values() if r["source"] == "human_corrected")
    rule = next(r for r in rows.values() if r["source"] == "rule")
    assert human["stage"] == "stage_4"  # human label overrode the model's stage_3
    assert float(human["sample_weight"]) > float(rule["sample_weight"])
    assert rule["stage"] == "stage_2"
    # Provenance of the model that produced each predicted label is carried through.
    assert human["grader_sha"] == "rule"
    assert rule["grader_sha"] == "rule"


def test_role_precedence_beats_recency(tmp_path) -> None:
    # A physician correction must not be overwritten by a LATER nurse review.
    store = FeedbackStore(tmp_path)
    store.append_inference(_inference("recA", stage="stage_3"))
    store.append_feedback(_feedback("recA", created_at="2026-07-11T01:00:00+00:00",
                                    reviewer_role="physician", corrected_stage="stage_4"))
    store.append_feedback(_feedback("recA", created_at="2026-07-11T09:00:00+00:00",
                                    reviewer_role="nurse", corrected_stage="stage_2"))
    (_, fb), = list(store.join_pairs())
    assert fb is not None
    assert fb.reviewer_role == "physician"      # senior tier wins despite being older
    assert fb.corrected_stage == "stage_4"


def test_agree_is_confirmed_not_human_weighted(tmp_path) -> None:
    # A bare agree (no correction) must be 'human_confirmed' at <= rule weight,
    # never laundered in at the reviewer's role weight.
    store_root = tmp_path / "store"
    store = FeedbackStore(store_root)
    store.append_inference(_inference("recA", stage="stage_2"))
    store.append_feedback(_feedback("recA", reviewer_role="physician",
                                    agree=True, corrected_stage=None))
    out = tmp_path / "m.csv"
    result = runner.invoke(feedback_app, ["export", "--root", str(store_root), "--out", str(out)])
    assert result.exit_code == 0, result.output
    with open(out, encoding="utf-8") as handle:
        row = next(csv.DictReader(handle))
    assert row["source"] == "human_confirmed"
    assert row["stage"] == "stage_2"            # still the model's own label
    assert float(row["sample_weight"]) <= 1.0   # capped at the rule tier

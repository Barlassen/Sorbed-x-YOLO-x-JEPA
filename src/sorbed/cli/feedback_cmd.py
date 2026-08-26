"""``sorbed feedback`` — emit inferences and ingest human review.

Three commands close the runtime feedback loop:

* ``record`` analyses an image through the shared pipeline and appends one
  :class:`~sorbed.feedback.records.InferenceRecord` (+ its mask PNG).
* ``submit`` attaches a human :class:`~sorbed.feedback.records.FeedbackRecord`
  (agree/disagree, optional corrected stage and mask) to a prior record.
* ``export`` joins the two logs into a training manifest CSV for the offline
  continual trainer, with the human label overriding the model's and a
  role-weighted ``sample_weight``.

No torch, no network — this is the CPU-only runtime surface.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from PIL import Image
from rich.console import Console

from sorbed.domain.enums import PressureInjuryStage
from sorbed.feedback import FeedbackRecord, FeedbackStore, emit_inference
from sorbed.pipeline import AnalyzeOptions, analyze_image

feedback_app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Emit analysis results and ingest human feedback for continual learning.",
)
console = Console()
err_console = Console(stderr=True)

# Role precedence for the training manifest: human HQ/physician corrections
# dominate nurse review, which dominates rule-teacher labels (see the continual
# trainer's weighting policy — this is the runtime-side default).
_ROLE_WEIGHTS: dict[str, float] = {
    "hq": 3.0,
    "physician": 3.0,
    "nurse": 1.5,
}
_RULE_WEIGHT = 1.0
_OVERRIDE_BOOST = 1.3

_MANIFEST_COLUMNS = [
    "image", "mask", "stage", "patient_id", "sample_weight", "source", "grader_sha"
]


@feedback_app.command()
def record(
    image: Annotated[Path, typer.Argument(help="Path to the wound image to analyse.")],
    patient: Annotated[
        str | None, typer.Option("--patient", help="Opaque patient/wound token (never a name).")
    ] = None,
    body_part: Annotated[
        str | None, typer.Option("--body-part", help="Opaque body-site token.")
    ] = None,
    mm_per_px: Annotated[
        float | None, typer.Option("--mm-per-px", help="Known scale in millimeters per pixel.")
    ] = None,
    root: Annotated[
        Path | None, typer.Option("--root", help="Feedback store root (else $SORBED_FEEDBACK_DIR).")
    ] = None,
) -> None:
    """Analyse an image and append an inference record with its mask."""
    if not image.exists():
        err_console.print(f"[red]No such file:[/red] {image}")
        raise typer.Exit(2)

    options = AnalyzeOptions(mm_per_px=mm_per_px)
    try:
        bundle = analyze_image(image, options=options)
    except Exception as exc:  # surface a clean message, not a traceback
        err_console.print(f"[red]Analysis failed:[/red] {exc}")
        raise typer.Exit(1) from exc

    store = FeedbackStore(root)
    record_id = emit_inference(
        bundle,
        store,
        patient_ref=patient,
        body_part=body_part,
        image_ref=str(image),
        image_bytes=image.read_bytes(),
    )
    decision = bundle.analysis.decision
    grade = "abstained" if decision.abstained else decision.stage.value
    console.print(f"[green]Recorded[/green] {record_id}  stage={grade}")


@feedback_app.command()
def submit(
    record_id: Annotated[str, typer.Option("--record-id", help="Inference record_id to review.")],
    role: Annotated[
        str, typer.Option("--role", help="Reviewer role: nurse | hq | physician.")
    ],
    reviewer: Annotated[
        str, typer.Option("--reviewer", help="Pseudonymous reviewer id (never a name).")
    ],
    agree: Annotated[
        bool, typer.Option("--agree/--disagree", help="Accept or reject the model's stage.")
    ],
    stage: Annotated[
        str | None, typer.Option("--stage", help="Corrected PressureInjuryStage value on disagree.")
    ] = None,
    mask: Annotated[
        Path | None, typer.Option("--mask", help="Optional human-fixed mask PNG.")
    ] = None,
    notes: Annotated[str, typer.Option("--notes", help="Free-text reviewer notes.")] = "",
    root: Annotated[
        Path | None, typer.Option("--root", help="Feedback store root (else $SORBED_FEEDBACK_DIR).")
    ] = None,
) -> None:
    """Attach a human feedback record to a prior inference."""
    corrected_stage = _validate_stage(stage)

    store = FeedbackStore(root)
    corrected_mask_path: str | None = None
    if mask is not None:
        if not mask.exists():
            err_console.print(f"[red]No such mask file:[/red] {mask}")
            raise typer.Exit(2)
        corrected_mask_path = store.save_correction_png(_load_mask(mask), record_id)

    store.append_feedback(
        FeedbackRecord(
            record_id=record_id,
            created_at=_now_iso(),
            reviewer_role=role,
            reviewer_id=reviewer,
            agree=agree,
            corrected_stage=corrected_stage,
            corrected_mask_path=corrected_mask_path,
            notes=notes,
            source="human",
        )
    )
    console.print(f"[green]Feedback stored[/green] for {record_id}  role={role}")


@feedback_app.command()
def export(
    out: Annotated[Path, typer.Option("--out", "-o", help="Manifest CSV path to write.")],
    root: Annotated[
        Path | None, typer.Option("--root", help="Feedback store root (else $SORBED_FEEDBACK_DIR).")
    ] = None,
) -> None:
    """Write a joined training manifest CSV of confirmed/corrected rows."""
    store = FeedbackStore(root)
    rows = list(_manifest_rows(store))
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=_MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    console.print(f"[green]Wrote {len(rows)} manifest row(s) to[/green] {out}")


def _manifest_rows(store: FeedbackStore) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for inf, fb in store.join_pairs():
        label = inf.predicted_stage
        mask = inf.mask_path
        source = "rule"
        weight = _RULE_WEIGHT
        if fb is not None:
            has_correction = bool(fb.corrected_stage or fb.corrected_mask_path)
            if fb.corrected_stage:
                label = fb.corrected_stage
            if fb.corrected_mask_path:
                mask = fb.corrected_mask_path
            if has_correction:
                # An independent human correction is the informative gradient —
                # it enters at the reviewer's role weight.
                source = "human_corrected"
                weight = _ROLE_WEIGHTS.get(fb.reviewer_role, _RULE_WEIGHT)
                if fb.corrected_stage and fb.corrected_stage != inf.predicted_stage:
                    weight *= _OVERRIDE_BOOST
            else:
                # A bare "agree" is a VOTE on the model's own output, not an
                # independent label. It must never enter at human role weight (that
                # laundered model predictions into human-weighted ground truth and
                # closed a feedback echo chamber) and is capped at the rule tier.
                source = "human_confirmed"
                weight = min(_ROLE_WEIGHTS.get(fb.reviewer_role, _RULE_WEIGHT), _RULE_WEIGHT)
        if not label:
            # Model abstained and no human correction — nothing to learn from.
            continue
        rows.append(
            {
                "image": inf.image_ref,
                "mask": mask,
                "stage": label,
                # An empty patient key makes the downstream leakage check trivially
                # true; give unkeyed rows a unique per-record group instead.
                "patient_id": inf.patient_ref or f"__record__:{inf.record_id}",
                "sample_weight": round(weight, 4),
                "source": source,
                # Provenance: which grader produced the predicted label this row is
                # built on — lets the continual trainer bound self-generated labels.
                "grader_sha": inf.model_versions.grader,
            }
        )
    return rows


def _validate_stage(stage: str | None) -> str | None:
    if stage is None:
        return None
    try:
        return PressureInjuryStage(stage).value
    except ValueError as exc:
        valid = ", ".join(s.value for s in PressureInjuryStage)
        err_console.print(f"[red]Invalid stage '{stage}'.[/red] Expected one of: {valid}")
        raise typer.Exit(2) from exc


def _load_mask(path: Path) -> np.ndarray:
    with Image.open(path) as img:
        return np.asarray(img.convert("L")) > 0


def _now_iso() -> str:
    from datetime import UTC, datetime

    return datetime.now(UTC).isoformat()

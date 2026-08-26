# Model and learning methods

This document describes how Sorbed's learned backends are trained, validated, and
updated. It applies only to the optional learned backends; the classical CPU
pipeline does not depend on any of it and runs offline. See
[`training/RUNBOOK.md`](RUNBOOK.md) for the operational procedure.

---

## 1. Deployed models

Two learned models are served, both exported to ONNX for CPU inference:

| Model | Role | Trainer | Config |
| --- | --- | --- | --- |
| SegFormer MiT-b3 / U-Net++ | wound-area segmenter | `training/train_seg.py` | `training/configs/seg_segformer.yaml`, `seg_unetpp_effnet.yaml` |
| ConvNeXt-V2 | pressure-injury stage grader | `training/train_grade.py` | `training/configs/grade_convnextv2.yaml` |

The segmenter is trained on the public AZH / MICCAI-2021 FUSeg foot-ulcer datasets.
The U-Net++/EfficientNetV2 recipe in `training/configs/seg_unetpp_effnet.yaml` is
the default bring-up configuration; it fits one ~40 GB MIG slice and produces a
verified ONNX export. Reported internal-validation Dice is 0.87. The stage grader
reaches 5-fold patient-independent quadratic-weighted κ = 0.919 ± 0.006 on PIID.

The remaining sections describe the grading pipeline, which is constrained by the
absence of a public staging dataset.

---

## 2. Teacher–student grading

No large, public, permissively-licensed pressure-injury staging dataset was found
(see [`docs/MODELS.md`](../docs/MODELS.md)). Staging ground truth is also noisy:
reported human inter-rater agreement is ~23–58%. Supervised training of a grader on
a non-existent labelled corpus is therefore not possible.

Grading is bootstrapped by teacher–student distillation
(`training/teacher_student.py`, config `training/configs/teacher_student.yaml`):

- The `DirectiveTeacher` encodes NPIAP/EPUAP staging criteria and the HD_T86
  directive pack (`scripts/build_directive_pack.py`) as a deterministic rule
  engine. Given the morphometric and tissue features the pipeline computes, it
  emits a soft label over stages and abstains when the evidence is ambiguous (for
  example a depth-dependent Stage 3-vs-4 call that a photo cannot resolve).
- The ConvNeXt-V2 student learns those soft labels (distillation loss over the
  non-abstained rows), generalising the rules into a visual model rather than
  memorising a hard-label set.

Where no graded corpus exists, an abstaining rule teacher provides the supervisory
signal, and distillation converts it into a fast image model. The supervised grader
described in this repository was trained directly on PIID stage labels; the
teacher–student path applies when human labels are unavailable.

---

## 3. MedSAM as a mask generator

Several stage-labelled but maskless collections exist (PIID stage labels,
DFUC-2021, Medetec aetiology images). They carry stage labels but no segmentation
masks, so they cannot be used for joint segmentation and grading directly.

`training/finetune_medsam.py` fine-tunes MedSAM and runs it as an offline mask
generator that produces wound masks for those sets. MedSAM is not deployed at
inference; it is a promptable, heavier model whose role is to generate masks once,
offline. The deployed segmenter remains the ONNX model of §1. This separation lets
the mask generator be large, since it never runs at inference, while the deployed
segmenter stays small and CPU-friendly.

---

## 4. Clinician corrections as a second label source

The rule teacher (§2) is the first label source. Clinician corrections are the
second, and take precedence over the rules.

At runtime the feedback loop is implemented in `src/sorbed/feedback/`
(`records.py`, `store.py`, `emit.py`) and the `sorbed feedback` CLI group
(`src/sorbed/cli/feedback_cmd.py`):

- Every analysis emits an inference record. A nurse can submit a grade, and a
  reviewer can submit a correction (mask fix and confirmed stage). These are stored
  as ground truth for the images the system presented to a clinician.
- `sorbed feedback export` joins the feedback log onto the inference log and writes
  a training manifest CSV. Reviewer/physician corrections take precedence over the
  rule-teacher label for the same row (role precedence is defined in
  `feedback_cmd.py`).

Because corrections are weighted above rule-teacher labels, the student
incorporates clinical judgment on deployed images.

---

## 5. Continual learning

`training/continual.py` (config `training/configs/continual.yaml`) consumes the
exported manifest and performs a replay-buffered continual fine-tune of the
currently-promoted grader:

1. **Harvest.** Read the new joined rows from `sorbed feedback export`.
2. **Replay.** Mix them with a stratified, per-class-capped sample of
   previously-confirmed rows from the replay manifest, with anchoring to the base
   weights and a small learning rate, to limit overfitting to the latest round.
3. **Fine-tune** the ConvNeXt student on the combined set, weighting human
   corrections above rule-teacher labels.
4. **Gate.** `promotion_decision` promotes only when the candidate shows no
   regression on a frozen hold-out in balanced accuracy, quadratic-weighted kappa
   (QWK), or calibration (ECE). Each check is one-sided; ties go to the base.
5. **Promote.** On a pass it writes `promoted.pt`, exports ONNX
   (`training/models.export_onnx`) with a recorded sha256, and appends the round's
   confirmed rows to the replay manifest.

If the gate fails, the previously-promoted ONNX remains live. This property allows
an unattended periodic loop to run without regression.

---

## 6. Federated-learning client seam

`training/fl_client.py` packages each promoted continual round as a client delta
(`ClientUpdate`, `compute_delta` = `fine_tuned − base`) so a FedAvg/FedProx server
can average updates across sites without patient data leaving the site; only the
parameter delta and non-PHI aggregate statistics are transmitted. `apply_delta` and
`fedavg_aggregate` implement the inverse and the server step. FedProx suits
non-IID hospital data (differing stage mix, camera, and skin-tone distribution); the
packaging is identical for either aggregator. This seam is implemented and covered
by tests but is not wired to a live server; secure aggregation and differential
privacy are noted as wrappers and are not enforced.

---

## 7. End-to-end flow

```
                         ┌──────────────────────────────────────────────────┐
                         │  OFFLINE BOOTSTRAP (one-time / periodic)          │
                         │                                                  │
  NPIAP/EPUAP doctrine ──▶ DirectiveTeacher (teacher_student.py)            │
  + HD_T86 pack           │   soft labels, ABSTAINS when ambiguous          │
                         │            │ distill                             │
  staged-but-maskless ──▶ MedSAM mask factory (finetune_medsam.py)         │
  sets (PIID/DFUC/…)      │   auto-masks ─────────┐                         │
                         │            ▼           ▼                         │
                         │      ConvNeXt-V2 student  ◀── train_grade.py     │
                         │      SegFormer segmenter  ◀── train_seg.py       │
                         └───────────────┬──────────────────────────────────┘
                                         │ export ONNX
                                         ▼
   ┌─────────────────────── RUNTIME (CPU, offline) ───────────────────────┐
   │  capture ─▶ analyze ─▶ emit record + mask + stats                    │
   │                              (src/sorbed/feedback: records/store/emit)│
   │                                   │                                   │
   │        nurse grade / HQ correction ▼  (sorbed feedback record|submit) │
   │                          feedback store (ground truth)                │
   └───────────────────────────────────┬───────────────────────────────────┘
                                        │ sorbed feedback export (JOIN)
                                        ▼
                             training manifest CSV
                                        │  + replay manifest (past rounds)
                                        ▼
        replay-buffered continual fine-tune  (training/continual.py)
                                        │
                             promotion gate (no regression:
                             balanced-acc / QWK / ECE on frozen holdout)
                              ┌─────────┴─────────┐
                         FAIL │                   │ PASS
                       keep live ONNX      promote ▶ export ONNX (+sha256)
                                                   │  append rows to replay
                                                   ▼
                                    (future) FL client delta
                                    training/fl_client.py ─▶ FedAvg/FedProx server
```

---

## 8. Limitations

- Grading accuracy is bounded by the rule teacher and the clinicians who correct
  it. Distillation cannot exceed the quality of its supervision, and staging ground
  truth is inherently noisy (§2). Grading is provided as decision support with
  explicit uncertainty, not as a diagnosis.
- Foot ulcers and pressure injuries differ in site, tissue, and depth. The public
  segmentation benchmarks are foot ulcers; site-specific deployment requires
  training on a pressure-injury set. The FUSeg path is the reproducible bring-up,
  not the clinical endpoint.
- MedSAM masks are machine-generated and carry the model's biases; they should be
  reviewed before anchoring a grading round.
- The continual gate protects against regression on the hold-out, not against
  distribution shift that the hold-out does not represent. The hold-out must be
  curated and refreshed.
- Federated learning is implemented and tested but not deployed; there is no live
  cross-hospital server, and secure aggregation and differential privacy are not
  enforced.

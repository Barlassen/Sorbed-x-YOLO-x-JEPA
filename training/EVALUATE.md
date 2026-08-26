# Evaluation: metrics, harness, and the auto-mask MedSAM fine-tune

This directory contains two evaluation surfaces:

- `training/evaluate.py` — the **metric harness** (segmentation
  and grading), dependency-light (numpy + scipy only).
- `training/finetune_medsam.py` — fine-tunes SAM/MedSAM for **automatic**
  (prompt-free) wound masks and has a built-in `eval` that compares the fine-tuned
  model with the supervised sorbed segmenter.

Both surfaces are patient-aware and report confidence intervals. See
`../DISCLAIMER.md` and the scope box in `DATA_README.md`:
a photo-only model is decision-support, not a standalone diagnostic, and true depth
/ Stage 1-vs-DTI in pigmented skin are not recoverable from a flat RGB frame.

---

## 1. `evaluate.py` — metric harness

Install the runtime core (numpy, scipy, opencv are already Sorbed core deps):

```bash
pip install -e .
```

### Segmentation — Dice / IoU / HD95 / ASSD

Point at a directory of predicted masks and a directory of ground-truth masks,
paired by filename **stem** (`0007.png` ↔ `0007.png`). Masks are binary
(wound = nonzero); mismatched sizes are nearest-resized to the GT.

```bash
python training/evaluate.py seg \
  --pred-dir /data/briefer/preds/segmenter \
  --gt-dir   /data/briefer/datasets/azh/test/labels \
  --out      /data/briefer/reports/seg_metrics.json
```

Reports, each with a bootstrap 95% CI over images:

- **Dice**, **IoU** — region overlap (empty-vs-empty scores as 1.0).
- **HD95** (pixels) — 95th-percentile symmetric boundary distance; robust to a few
  outlier boundary pixels. Reported in **pixels**; multiply by a real per-image
  `mm_per_px` (from a fiducial) for millimetres — the harness does not guess scale.
- **ASSD** (pixels) — average symmetric surface distance.

Single-sided-empty boundary cases are `NaN` and excluded from the boundary means,
so a model that predicts nothing is not rewarded with a zero distance.

### Grading — balanced-acc / quadratic-weighted κ / per-stage sensitivity / calibration

Feed a predictions CSV. Required columns `true` and `pred` (each a class string
from the ordinal order, or a class index). Optional `patient_id` groups rows so the
bootstrap is **clustered at the patient level** (multiple wounds per patient are
correlated, so the CI is clustered). Optional `prob_<class>` columns for
every class turn on **calibration** (ECE, Brier, reliability table).

```
true,pred,patient_id,prob_stage_1,prob_stage_2,prob_stage_3,prob_stage_4,prob_unstageable,prob_deep_tissue_injury
stage_2,stage_2,P001,0.02,0.86,0.07,0.02,0.02,0.01
stage_3,stage_4,P001,0.01,0.04,0.35,0.55,0.03,0.02
stage_1,stage_1,P002,0.71,0.20,0.03,0.02,0.02,0.02
```

```bash
python training/evaluate.py grade \
  --csv /data/briefer/preds/grade_test.csv \
  --out /data/briefer/reports/grade_metrics.json
```

Reports:

- **Quadratic-weighted Cohen's κ** (primary) — penalises distant mis-grades
  (Stage 4 → Stage 1) far more than adjacent ones, matching the clinical cost of
  under-staging. Patient-clustered bootstrap CI.
- **Balanced accuracy** — mean per-class recall (stage prevalence is skewed; plain
  accuracy would hide rare, critical Stage 4 / Unstageable).
- **Per-stage sensitivity** — recall for each stage, so under-staging is
  visible per class.
- **Confusion matrix** — rows = truth, cols = prediction.
- **Calibration** (when probabilities are given) — **ECE**, **Brier**, and a
  **reliability table** (per-bin confidence vs accuracy) emitted as data for
  plotting. Report ECE **before and after** temperature scaling, and stratified by
  skin tone, to surface miscalibration on darker skin.

The default class order matches `sorbed.domain.enums.PressureInjuryStage`:
`stage_1, stage_2, stage_3, stage_4, unstageable, deep_tissue_injury`. Override with
`--classes` (comma-separated, ordinal order first — it drives the κ weights).

### Scope left to the caller

The methods plan calls for more than point metrics: **patient-level grouped CV**
(use `training.data.group_kfold_indices` / `training.datasets.split_by_patient`),
**per-skin-tone / per-site subgroup** slices (run `grade` on each stratum's CSV),
DeLong AUROC CIs, temperature scaling, and an MRMC reader study. Produce the
prediction CSVs per fold / per subgroup and call this harness on each — it is the
per-run metric core, not the experiment orchestrator.

---

## 2. `finetune_medsam.py` — automatic (prompt-free) SAM/MedSAM

### Overview

SAM, MedSAM, and SAM-2 are **promptable**: they need a box or point, so they cannot
run unattended. This module makes them automatic by freezing the heavy image encoder and
adding one of two cheap prompt sources, then fine-tuning the mask decoder:

- `--auto-prompt learned` — a small CNN on the frozen image embedding regresses a
  wound bounding box; that box drives the prompt encoder at inference. Trained
  jointly with the decoder.
- `--auto-prompt heuristic` — no learned head; at inference the box is the bounding
  box of the weight-free `ClassicalSegmenter` mask (box-from-heuristic). Only the
  decoder is trained.

The decoder is supervised with **GT-jittered box prompts** (the MedSAM recipe), so
it tolerates the imperfect boxes the auto-prompt yields at test time. The image
encoder and prompt encoder stay **frozen** — only the mask decoder (and the head)
learn, which fits on one ~40 GB H200 MIG slice and keeps the export
surface small.

### Weights (HuggingFace) and offline fallback

Uses the `transformers` `SamModel` / `SamProcessor` API, so any SAM-format
checkpoint works via `--model-id`:

- `facebook/sam-vit-base` (default), `facebook/sam-vit-large`, `facebook/sam-vit-huge`
  — original Meta SAM.
- A MedSAM checkpoint exported to the `transformers` SAM format (medical fine-tune
  of SAM-ViT-B). Pass its Hub id with `--model-id`; **verify the exact repo name on
  the Hub** before relying on it — this script does not hard-code a MedSAM URL.

**SAM-2 / MedSAM-2** (`wanglab/MedSAM2`) run on the separate `facebookresearch/sam2`
package (not `transformers`) and are tuned for 3D/video; for single 2D wound photos
this module uses the `transformers` SAM/MedSAM path, which is ONNX-friendlier.

Offline / air-gapped — pre-download once, then no further network access is required:

```bash
huggingface-cli download facebook/sam-vit-base --local-dir /data/briefer/sam-vit-base
python training/finetune_medsam.py train --weights-dir /data/briefer/sam-vit-base ...
```

`--weights-dir` (or `HF_HUB_OFFLINE=1`) forces `local_files_only`.

### Install

```bash
pip install "sorbed[hf]"          # torch + transformers + huggingface-hub
# or: pip install torch transformers huggingface-hub opencv-python-headless
```

### Data

A manifest from `training/data_prep.py` whose rows carry an `image_path` and a
binary `mask_path`. AZH/FUSeg foot-ulcer masks are the recommended open start
(`DATA_README.md`). Prepare it once:

```bash
python training/data_prep.py \
  --data-root /data/briefer/datasets/wound-segmentation/data/Foot_Ulcer_Segmentation_Challenge \
  --adapter azh_fuseg --source azh_fuseg --license research-only \
  --out-dir /data/briefer/manifests/azh
```

### Train (on the H200, under tmux)

```bash
python training/finetune_medsam.py train \
  --train-manifest /data/briefer/manifests/azh/train.jsonl \
  --val-manifest   /data/briefer/manifests/azh/val.jsonl \
  --paths-relative-to /data/briefer/datasets \
  --auto-prompt learned \
  --out-dir /data/briefer/artifacts/medsam_auto \
  --epochs 30 --batch-size 4 --lr 1e-4
```

Writes `best.pt` / `last.pt` (mask-decoder + auto-prompt-head weights, selected by
val Dice). Best-checkpoint selection uses the same auto-mask path used at inference,
so the reported val Dice reflects the prompt-free pipeline, not a box-fed oracle.

### Evaluate — auto-SAM vs the supervised segmenter

```bash
python training/finetune_medsam.py eval \
  --test-manifest /data/briefer/manifests/azh/test.jsonl \
  --paths-relative-to /data/briefer/datasets \
  --auto-prompt learned \
  --checkpoint /data/briefer/artifacts/medsam_auto/best.pt \
  --out /data/briefer/reports/medsam_eval.json
```

Prints Dice/IoU for the auto-SAM model **and** for the supervised sorbed segmenter
on the same test images, plus the delta. The supervised baseline is the weight-free
`ClassicalSegmenter`, or the ONNX U-Net from `scripts/train_segmenter.py` when
`SORBED_ONNX_MODEL` points at an exported `model.onnx`:

```bash
SORBED_ONNX_MODEL=/data/briefer/artifacts/segmenter/model.onnx \
python training/finetune_medsam.py eval --test-manifest ... --out ...
```

For HD95/ASSD on these predictions, dump the auto-SAM masks to a directory and run
`evaluate.py seg` against the GT labels — the two tools compose.

---

## Reporting checklists

Map results to **CLAIM**, **TRIPOD-AI**, **STARD-AI**, **DECIDE-AI**, and
**PROBAST-AI** as described in the methods plan. Report internal **and**
external-validation numbers separately, with subgroup (skin-tone / site / device)
slices for every primary metric. The external drop and the skin-tone sensitivity
gap are reported alongside the internal best case.

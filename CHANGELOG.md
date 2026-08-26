# Changelog

All notable changes to Sorbed are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- Project scaffolding, governance, and community health files (Apache-2.0
  license, code of conduct, contributing guide, security policy, clinical
  disclaimer).
- Core analysis pipeline: multi-format image ingestion, preprocessing,
  segmentation, tissue analysis, morphometrics, explainable staging engine,
  visualization, and reporting.
- Command-line interface and FastAPI service.
- `scripts/watchdog.py` integrity guard that fails the build on placeholder,
  stub, or fabricated-output code paths.
- Clinical **directive grounding**: build a citable "directive pack" from a
  hospital's pressure-injury guideline PDF (`scripts/build_directive_pack.py`)
  and ground every report in its numbered sections and figures
  (`sorbed.guidelines`).
- **PDF reports** (`sorbed.report`, optional `[pdf]` extra): a concise one-page
  summary report, a detailed directive-grounded grading report, and a
  longitudinal follow-up report (healing verdict, inline SVG trend charts,
  per-visit analysis, directive-cited alerts). Bilingual (Turkish · English),
  with the vendored Manrope (OFL) typeface and grade/confidence colouring.
  Revised against a strict two-reviewer clinical audit: safety caveats sit under
  the grade, estimates are labelled "not measurements", a slough-free bed raises
  an under-detection warning rather than a false "clean wound", and the guideline
  is shown as a schematic (not a photo) for the detected grade only.
- Healing **follow-up alerts** (`sorbed.trend.alerts`): a better/worse verdict
  plus ranked, directive-cited signals (area/PAR, PUSH, granulation, necrosis,
  stage progression).
- Segmentation trainer gains `--decoder-attention scse` (spatial-and-channel
  Squeeze-and-Excitation), the FUSegNet-style attention, plus documented
  EfficientNet-encoder training toward the AZH/FUSeg benchmark.
- Segmentation trainer gains `--arch` (routed through segmentation-models-pytorch):
  the CNN FUSegNet line (`unet`/`deeplabv3plus`/`manet`) **and** the modern
  transformer recipe (`segformer` with a MiT encoder, e.g. `--encoder mit_b2`),
  all ONNX-exportable. Docs record the 2024–2026 landscape (nnU-Net, Mamba,
  SAM-2 / MedSAM-2 / BiomedParse) as not-yet-integrated.
- **`training/` GPU fine-tuning package** (kept out of the CPU-only `sorbed`
  wheel; heavy deps lazy-imported). Sized for a single ~40 GB H200 MIG slice:
  - `train_seg.py` (SegFormer/MiT + U-Net++/EfficientNetV2 segmenter, AMP,
    cosine-warmup, patient-level grouped K-fold, TensorBoard, opset-17 ONNX
    export) and `train_grade.py` (ConvNeXt-V2/ViT stage classifier, CE/focal/
    ordinal-CORN, quadratic-weighted-κ eval).
  - `teacher_student.py` — a deterministic, auditable **directive teacher** that
    pseudo-labels per the HD_T86 / NPIAP schema with abstention and section
    citations, plus KD + mean-teacher/FixMatch student distillation.
  - `finetune_medsam.py` — prompt-free SAM/MedSAM mask-decoder fine-tune
    (learned or heuristic auto-box) with an auto-SAM-vs-supervised eval.
  - `evaluate.py` — publication-grade metrics (Dice/IoU/HD95/ASSD; balanced
    accuracy, quadratic-weighted κ, per-stage sensitivity, ECE/Brier) with
    patient-clustered bootstrap CIs, for NEJM/CLAIM/TRIPOD-AI reporting.
  - `data_prep.py` + `datasets.py` — multi-source, leakage-free patient-level
    manifests with honest dataset/license auditing (`DATA_README.md`); no
    dataset download URLs invented, gated sets consumed from local dirs.
  - `server/` scripts + `RUNBOOK.md` — reproducible transfer→setup→train→
    monitor→export flow under tmux + venv, MIG-UUID-pinned.
- **Feedback + continual-learning loop** (human-in-the-loop as the second
  teacher, federated-learning-ready):
  - `sorbed.feedback` (CPU runtime, no torch): every analysis emits an
    `InferenceRecord` + saved mask PNG + tissue/area/PUSH stats to an append-only
    JSONL store; clinician corrections attach as `FeedbackRecord`s. New
    `sorbed feedback record|submit|export` CLI; the export joins the two logs
    into a weighted training manifest (human corrections outrank rule labels).
  - `training/continual.py` — replay-buffered continual fine-tune of the grader
    that **refuses to promote** a new checkpoint unless it holds balanced-
    accuracy / quadratic-weighted-κ and does not worsen calibration (ECE) on a
    frozen hold-out, then re-exports ONNX.
  - `training/fl_client.py` — the federated seam: `ClientUpdate` (weight deltas,
    sample count, metrics) with FedAvg aggregation, usable locally now so a
    future cross-hospital FedAvg/FedProx server drops in without a rewrite; no
    data leaves the client, only deltas.
  - `training/STRATEGY.md` documents the model and learning methods (SegFormer +
    ConvNeXt-V2 deployed; MedSAM as an offline mask generator; teacher–student
    grading; clinician-in-the-loop continual learning; federated client seam).
- **Extended dataset corpus**: `training/fetch_corpus.py` (+ `configs/datasets.yaml`)
  aggregates the open, non-interactively fetchable wound sources — AZH·FUSeg,
  Mendeley Lower-Limb-&-Feet (CC-BY), CO2Wounds-V2, DFUTissue, WSNet,
  ComplexWoundDB, PIID, Roboflow/Kaggle pressure-injury staging sets — and
  documents the gated ones; no download URLs invented.
- **One-shot server bootstrap** `training/server/bootstrap.sh`: `curl … | bash`
  clones the repo, builds the CUDA venv, fetches open data, and launches training
  in tmux — pinned to a MIG UUID, fully env-configurable.
- **Out-of-memory safety** (`training/memory.py`): the trainers no longer crash on
  a shared MIG slice. On a CUDA OOM the step empties the cache and **retries the
  batch split into more micro-batches** (gradient accumulation — effective batch
  unchanged), remembering the working split; validation forward passes shrink the
  same way. `--vram-fraction` caps the process to a share of the slice (catchable
  early OOM), `run_tmux.sh` exports `PYTORCH_CUDA_ALLOC_CONF=expandable_segments`,
  and `--num-workers` is auto-capped by available RAM/CPU. Wired into `train_seg`,
  `train_grade`, and `continual`; covered by `tests/test_training_memory.py`.
- **No-API model weights** (`training/fetch_models.py`): downloads SAM / MedSAM /
  SAM-2 weights with **no HuggingFace token and no API key** (public URLs). The
  transformers-format `medsam-vit-base` (public `flaviagiammarino` mirror) loads
  directly into `finetune_medsam.py` via `--weights-dir`; original Meta SAM /
  SAM-2.1 CDN checkpoints are also offered. `SAM 3` (gated) is intentionally
  excluded. Corrected the PIID note (stages are EPUAP I–IV, not NPIAP).
- **Grading benchmark** (`training/benchmark.py`): compares the learned grader
  against the directive rule-engine baseline on a validation fold (accuracy,
  quadratic-weighted κ, rule-engine abstention rate) and renders a montage of
  graded images with wound-mask overlays.
- **Trained-model results** (single-institution internal validation): segmenter
  Dice 0.87 (AZH/FUSeg); ConvNeXt-V2 stage grader 5-fold patient-independent
  quadratic-weighted κ = 0.919 ± 0.006 on PIID (EPUAP I–IV), balanced accuracy
  ≈ 0.80; temperature scaling (T=3.05) reduces ECE from 0.149 to 0.039. Model
  weights are tracked with Git LFS (see `.gitattributes`).

<!--
Template for future releases:

## [x.y.z] - YYYY-MM-DD
### Added
### Changed
### Deprecated
### Removed
### Fixed
### Security
-->

[Unreleased]: https://github.com/ArioMoniri/Sorbed/commits/main

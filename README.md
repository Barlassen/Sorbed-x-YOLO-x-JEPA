# Sorbed

[![License: Apache-2.0](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python 3.11+](https://img.shields.io/badge/python-3.11%2B-blue.svg)](https://www.python.org/)

Sorbed analyses a photograph of a pressure injury (bedsore) and returns an
explainable, provisional grade: a stage, the tissue composition of the wound bed,
size measurements when the image carries a scale reference, and a structured
account of the evidence behind each decision. The default pipeline is classical
computer vision and runs offline on a laptop CPU with no model downloads. Learned
ONNX backends are optional and can be enabled; they are not required.

> **This is decision-support software, not a medical device.** It does not
> diagnose. Every value it reports is an estimate computed from a 2D image. A
> pressure-injury stage is a clinical determination made by a qualified
> professional; Sorbed's output is one input to that judgment. See
> [`DISCLAIMER.md`](DISCLAIMER.md) before applying it to a real wound.

## What it does

Given an image and an optional scale, the CLI writes a folder of artifacts:

```bash
sorbed analyze wound.jpg --mm-per-px 0.15 --out reports
```

```
Provisional grade: Stage 3  (62%)
Area: 8.40 cm²   Size: 41 × 29 mm
Tissue: granulation 58%, slough 27%, eschar 9%
! Depth-dependent grade — confirm Stage 3 vs 4 at the bedside.
```

Alongside the console summary it writes a binary mask, a tissue overlay, an
annotated clinician guide, a schematic, the canonical JSON, and a self-contained
HTML report into `reports/`. Phone photos are supported; the wound may be any size
or body location. When the photo has no scale reference, physical measurements are
reported as `null` and pixel-space metrics are reported instead.

## Pipeline overview

Each stage is a typed transform that only adds to an immutable context, so any
point can be snapshotted and inspected.

```
 image bytes ─▶ Ingest ─▶ Decode & Normalize ─▶ Preprocess (calibrate) ─▶
   Wound Segmentation ─▶ Tissue Segmentation ─▶ Morphometrics ─▶
     Staging Engine (rule ⊕ ML ⊕ arbiter) ─▶ Explainability ─▶ Outputs
```

The staging engine uses a deterministic rule layer encoding NPIAP/EPUAP staging
criteria; a learned head, when enabled, proposes candidates that an arbiter
reconciles against the rules. See [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md).

## Install

Requires Python 3.11+.

```bash
pip install -e .                 # core: classical pipeline, CLI, no downloads
pip install -e .[formats]        # HEIC/HEIF, DICOM, camera RAW, BigTIFF decoders
pip install -e .[ml]             # optional ONNX learned backends
pip install -e .[api]            # FastAPI service
pip install -e .[pdf]            # PDF grading & follow-up reports (headless Chromium)
pip install -e .[dev]            # ruff, mypy, pytest, hypothesis
pip install -e .[all]            # formats + ml + pdf + api + hf
```

After installing the `[pdf]` extra, fetch the Chromium build once:

```bash
playwright install chromium
```

## Quickstart

CLI:

```bash
sorbed analyze wound.heic --marker-mm 20 --out reports   # calibrate from an ArUco marker
sorbed inspect wound.dcm                                  # decode & show metadata + scale, no grading
sorbed formats                                            # which decoders are installed
sorbed schema --out analysis.schema.json                 # the JSON contract
```

Python:

```python
from sorbed.pipeline import analyze_image, AnalyzeOptions

bundle = analyze_image("wound.jpg", options=AnalyzeOptions(mm_per_px=0.15))
d = bundle.analysis.decision
print(d.stage, round(d.confidence, 2), d.abstained)
print(bundle.analysis.metrics.tissue.fractions)
```

## Outputs

Every artifact is a rendering of one canonical JSON record; nothing downstream
computes a value the JSON does not already contain.

| Artifact | File | Description |
|---|---|---|
| Mask | `*_mask.png` | Binary wound-boundary mask. |
| Overlay | `*_overlay.png` | Per-tissue colouring over the original photo. |
| Guide | `*_guide.png` | Annotated photo: contour, length × width, scale bar, legend, grade banner. |
| Schematic | `*_schematic.png` | Synthetic drawing of the wound, free of photographic noise. |
| Schematic guide | `*_schematic_guide.png` | The schematic with measurements and labels. |
| JSON | `*.json` | Canonical `WoundAnalysis` record. |
| HTML | `*.html` | Self-contained clinician report. |
| PDF (summary) | `*_summary.pdf` | One-page single-image report (needs `[pdf]`). |
| PDF (grade) | `*_grading.pdf` | Directive-grounded grading report (needs `[pdf]`). |
| PDF (follow-up) | `*_followup.pdf` | Longitudinal healing report with trend charts and alerts. |

Select a subset with `--format mask,json` or take all with `--format all` (default).

## Trained models and results

Two learned models are provided, both exported to ONNX for CPU inference. They are
trained out-of-tree in [`training/`](training/) and are not required by the
classical pipeline.

| Model | Architecture | Data | Metric |
|---|---|---|---|
| Wound segmenter | U-Net++ / EfficientNetV2-S (scSE decoder attention) | AZH Chronic Wound + MICCAI-2021 FUSeg, 768 px | Internal-validation Dice **0.87** |
| Stage grader | ConvNeXt-V2, CORN ordinal head | PIID (EPUAP stages 1–4; 1,091 images, 1,087 after removing 4 exact duplicates), 288 px | 5-fold patient-independent CV quadratic-weighted **κ = 0.919 ± 0.006**; balanced accuracy ≈ 0.80 |

Grader calibration: temperature scaling with T = 3.05 reduces the expected
calibration error from 0.149 to 0.039 on the validation fold.

A SegFormer / MiT-b3 recipe is provided as an alternative segmenter, and a MedSAM
mask-decoder fine-tune ([`training/finetune_medsam.py`](training/finetune_medsam.py))
is provided as an offline mask generator for stage-labelled sets that lack masks.
[`training/benchmark.py`](training/benchmark.py) compares the learned grader against
the directive rule-engine baseline on a validation fold and renders a montage of
graded images with their wound-mask overlays.

Reported numbers are single-institution internal validation. PIID is
EPUAP-staged and has no external test set; the segmenter is trained on
diabetic-foot ulcers, which differ from pressure injuries in anatomical site,
tissue, and depth. See [`docs/MODELS.md`](docs/MODELS.md) and the reproduction
steps in [`training/RUNBOOK.md`](training/RUNBOOK.md).

### Model weights

Trained ONNX models and checkpoints are tracked with [Git LFS](https://git-lfs.com)
(see [`.gitattributes`](.gitattributes)); install it once with `git lfs install`
before cloning or pulling the weights. Enable a learned backend at runtime:

```bash
export SORBED_SEGMENTATION_BACKEND=onnx
export SORBED_ONNX_MODEL=artifacts/seg_unetpp/model.onnx
```

## Clinical directive grounding

Sorbed can ground a report in an institution's own pressure-injury directive. A
**directive pack** is that guideline — for example an institutional NPIAP/EPUAP-aligned
protocol — distilled from its PDF into citable numbered sections plus the
guideline's figures:

```bash
python scripts/build_directive_pack.py --pdf HD_T86_REV10.pdf --slug my_hospital
export SORBED_DIRECTIVE_PACK=var/directive_packs/my_hospital
```

The analysis is mapped onto the directive's criteria: the staging definition and
the guideline's illustration for the detected stage, its tissue-colour
(red-yellow-black) model, sizing and PUSH guidance, and the reassessment cadence,
each with a section/page citation. Institutional directive content is loaded at
runtime and is not vendored into this repository.

## PDF reports and healing follow-up

Three print-ready, self-contained, bilingual (Turkish · English) PDFs (Manrope
typeface, grade- and confidence-based colouring), rendered by headless Chromium:

- **Summary report** — a one-page per-upload report: the grade and confidence, a
  clinician-confirmation banner, the uploaded photo beside the tissue overlay, key
  measurements, the viable/non-viable bar, and the directive's stage criterion
  (verbatim, with an NPIAP English gloss and citation).
- **Grading report** — safety caveats under the grade; a clinical-statistics card
  (bed-normalised tissue viability as a viable/non-viable split, tissue areas in
  cm², a qualitative wound-bed descriptor, standard L×W×area, and an
  under-detection warning when the classifier reports a slough-free bed on a deep
  wound); generated-analysis visuals (binary mask, tissue overlay, detection,
  relative-depth, schematic); a guideline-comparison panel placing the directive's
  schematic and section-cited text beside Sorbed's schematic and findings; and the
  tissue-colour model and care/reassessment cadence.
- **Follow-up report** — a better/worse verdict banner with a healing gauge and
  healing-velocity band, vector trend charts (wound area with a projected-closure
  line, tissue mix over visits, PUSH total), directive-cited clinical alerts, and a
  per-visit analysis section (each visit's photo, overlay, and
  stage/area/granulation/necrosis/PUSH/confidence).

Estimates are labelled as estimates rather than measurements; absence of flags is
not reported as "safe"; undermining and depth are stated as non-assessable from a
photo. Derived clinical statistics are defined in `sorbed.report.stats`
(named constants, bed-normalised, each proxy flagged).

Longitudinal analytics track granulation and full tissue composition, surface area
(cm² when calibrated), PUSH, percent area reduction, the 4-week PAR predictor, and
the Gilman perimeter-normalized healing rate — see [`docs/TREND.md`](docs/TREND.md).

## Clinical workflow: autograde on upload and QA

The target workflow follows how wound imaging flows through an EHR, where a nurse
photographs the wound, uploads it, and enters a stage that a central quality office
re-checks. Sorbed slots into that loop as decision support:

1. **On upload**, autograde the image and pre-fill a provisional stage, size,
   tissue composition, and a directive-cited rationale, so review starts from a
   structured draft rather than a blank field.
2. **Human-in-the-loop:** the reviewer confirms or edits; low-confidence or
   out-of-distribution images (poor lighting, obscured bed) abstain and route to
   review rather than forcing a stage.
3. **Follow-up** runs across a patient's visits, surfacing the healing trajectory
   and better/worse alerts.

A clinician owns the final determination. **Privacy and governance:** the
weight-free classical core runs on-premises with no upload, DICOM PHI is stripped,
learned weights are sha256-verified, and the federated-learning path keeps patient
data on-site. These properties are consistent with **KVKK** (Türkiye) and the
**GDPR**, and position the software as clinician decision support under
EU-MDR-harmonised Turkish medical-device regulation (**TİTCK**).

## How the grade is decided

The stage is produced by an explicit, auditable rule engine. Staging is driven by
which tissues are present: visible fat forces at least Stage 3, structural tissue
forces Stage 4, a bed obscured by slough or eschar forces Unstageable, and
maroon-to-purple discoloration on intact skin indicates Deep Tissue Injury. Each
grade carries a ranked evidence trace referencing metric fields (for example
`metrics.tissue.fractions.eschar`), a confidence, caveats, and a deterministic
narrative.

Depth-dependent grades are damped and flagged for clinician review, because depth
cannot be asserted from a photograph. When the evidence does not support a
confident grade, the engine returns **Indeterminate** and abstains. The clinical
specification is in [`docs/CLINICAL.md`](docs/CLINICAL.md).

## Skin-tone equity

Stage 1 and Deep Tissue Injury are defined partly by colour changes that are harder
to observe in darkly pigmented skin, and naive colour analysis can amplify that
bias. Sorbed estimates an ITA-based skin-tone band, lowers confidence and raises an
explicit warning on darker skin, and prompts assessment of temperature, firmness,
and edema. A low or negative result on dark skin does not rule out injury. See
[`docs/EQUITY.md`](docs/EQUITY.md).

Skin tone is one of several deployment considerations. For a given site the
material factors also include the **foot-ulcer ↔ pressure-injury domain gap**,
**imaging variability** (ward phone photos, lighting, no scale — uncalibrated
images report pixel metrics and low-quality inputs route to review), and **data
privacy**. For a predominantly Fitzpatrick II–IV population (for example Türkiye)
the erythema-visibility gap is smaller than in more diverse settings but is not
eliminated; the ITA safeguard is retained.

## Formats

| Family | Formats | Requires |
|---|---|---|
| Common | PNG, JPEG, WEBP, BMP, GIF, TIFF | core |
| Phone | HEIC / HEIF | `[formats]` (pillow-heif) |
| Clinical | DICOM (pixel-spacing calibration + PHI stripping) | `[formats]` (pydicom) |
| Camera | RAW | `[formats]` (rawpy) |

Run `sorbed formats` to see what is installed.

## Segmentation background and training

For wound-area segmentation, public benchmarks exist. Sorbed's learned segmenter
is trained on the **AZH Chronic Wound / MICCAI-2021 FUSeg** foot-ulcer datasets.
The trainer (segmentation-models-pytorch) supports the CNN FUSegNet line (U-Net /
DeepLabV3+ / MAnet with EfficientNet or ResNet encoders, plus **scSE** decoder
attention) and transformer encoders (**SegFormer / MiT-b***), all ONNX-exportable.

Published segmentation baselines on FUSeg (for reference):

| Approach | Reported DSC | Source |
|---|---|---|
| FUSegNet (EfficientNet-b7 + P-scSE) | 92.70% | Dhar et al., *Biomed. Signal Process. Control* 2024 |
| x-FUSegNet (5-fold ensemble) | 89.23% | FUSeg-2021 challenge leaderboard |
| LinkNet-EffB1 + UNet-EffB2 (ensemble) | 92.07% | Mahbod et al. |
| DeepLabV3+ / PSPNet / MANet | 91–92% | baselines |
| U-Net + scSE | 91.85% | scSE over plain U-Net (90.88%) |

```bash
# CNN FUSegNet-line recipe
python scripts/train_segmenter.py --images imgs/ --masks masks/ \
    --arch unet --encoder efficientnet-b4 --decoder-attention scse

# transformer recipe (SegFormer + MiT encoder)
python scripts/train_segmenter.py --images imgs/ --masks masks/ \
    --arch segformer --encoder mit_b2
```

Foot ulcers and pressure injuries differ in anatomical site, tissue, and depth. The
segmenter shipped here is trained on public foot-ulcer data; site-specific
deployment requires training or fine-tuning on a pressure-injury dataset. The
[`training/`](training/) package supports per-site retraining, continual learning
from clinician QA corrections ([`training/continual.py`](training/continual.py)),
and a federated-learning client seam ([`training/fl_client.py`](training/fl_client.py))
so patient data remains on-site.

For a cited survey of 2024–2026 model and system designs — promptable foundation
models (SAM / MedSAM / MedSAM-2), state-space (Mamba) segmenters, on-device staging
(YOLOv8), skin-tone equity, and the EHR/regulatory picture (FDA SaMD, EU MDR + AI
Act, TİTCK) — see [`docs/MODELS.md`](docs/MODELS.md).

### Training and continual learning

The learned backends are trained in [`training/`](training/) and are not required
by the CPU pipeline. Stage grading uses a rule teacher and a learned student: the
rule-based [`DirectiveTeacher`](training/teacher_student.py) (NPIAP/EPUAP criteria,
with abstention) supplies soft stage labels that a ConvNeXt-V2 grader distils; a
U-Net++/SegFormer segmenter ([`train_seg.py`](training/train_seg.py)) covers wound
area; and MedSAM ([`finetune_medsam.py`](training/finetune_medsam.py)) generates
masks for stage-labelled sets that lack them. The runtime feedback store
([`src/sorbed/feedback/`](src/sorbed/feedback/), via
`sorbed feedback record|submit|export`) records clinician corrections, which a
replay-buffered [`continual.py`](training/continual.py) fine-tune incorporates
behind a no-regression promotion gate before re-exporting ONNX; each promoted round
is packaged as a federated client delta ([`fl_client.py`](training/fl_client.py)).

- **Methods:** [`training/STRATEGY.md`](training/STRATEGY.md)
- **Server runbook (H200 MIG, curl bootstrap):** [`training/RUNBOOK.md`](training/RUNBOOK.md)
- **Datasets and placement:** [`training/DATA_README.md`](training/DATA_README.md)
- **Evaluation metrics:** [`training/EVALUATE.md`](training/EVALUATE.md)
- **Trainers:** `train_seg.py`, `train_grade.py`, `teacher_student.py`, `finetune_medsam.py`, `continual.py`, `benchmark.py`

## API

An optional FastAPI service exposes the pipeline over HTTP (install `[api]`):

```bash
uvicorn sorbed.api.app:app --reload
# POST /v1/analyze   GET /v1/health   GET /v1/formats
```

## Project layout

```
Sorbed/
├── src/sorbed/
│   ├── domain/          pydantic v2 data contracts
│   ├── io/              format sniffing, decoders, normalization
│   ├── preprocess/      calibration, fiducial detection, colour normalization
│   ├── segmentation/    wound boundary — classical + pluggable backends
│   ├── tissue/          tissue-type classification
│   ├── morphometrics/   geometry, tissue %, healing sub-scores
│   ├── staging/         features · rules · ml_head · arbiter · engine
│   ├── explain/         evidence trace
│   ├── visualize/       masks, overlays, guides, schematics
│   ├── report/          HTML / PDF / JSON assembly
│   ├── feedback/        inference + clinician-correction store
│   ├── pipeline/        orchestrator
│   ├── api/             FastAPI service
│   └── cli/             Typer command line
├── training/            out-of-tree GPU training, evaluation, benchmark
├── docs/                CLINICAL · MODELS · ARCHITECTURE · USAGE · EQUITY · TRAINING · TREND · API
├── scripts/             watchdog, training helpers
└── tests/
```

## Limitations

- **Depth is inferred, not measured.** Stage 3 vs 4 and Unstageable depend on depth
  and structures a single photo cannot reliably convey. The shading depth proxy is
  a weak, flagged cue, not a measurement.
- **Undermining and tunneling are not visible** on the surface and must be entered
  by a clinician.
- **Scale depends on calibration.** Without DICOM spacing, a fiducial, or a manual
  `--mm-per-px`, measurements are pixel counts, not centimetres.
- **Skin-tone bias** affects early-stage and deep-tissue detection, as above.
- **Image quality** — lighting, white balance, focus, angle, and occlusion move the
  result.
- **Partial scores.** PUSH, BWAT, and DESIGN-R sub-scores that need palpation or
  probing are left `null` and reported as partial totals.
- **Validation scope.** Reported metrics are single-institution internal validation
  without an external test set.

## Contributing and policies

[`CONTRIBUTING.md`](CONTRIBUTING.md) · [`CODE_OF_CONDUCT.md`](CODE_OF_CONDUCT.md) ·
[`SECURITY.md`](SECURITY.md) · [`CHANGELOG.md`](CHANGELOG.md) ·
[`DISCLAIMER.md`](DISCLAIMER.md)

Development tasks are wrapped in the [`Makefile`](Makefile): `make check` runs the
CI gate (lint, the integrity watchdog, and tests). A pre-commit config is provided —
`pre-commit install`.

## License

Apache-2.0. © 2026 Ariorad Moniri. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

*Not affiliated with, endorsed by, or a product of NPIAP, EPUAP, PPPIA, or JSPU.
Sorbed encodes publicly documented staging criteria from those bodies' guidelines
but is an independent open-source project.*

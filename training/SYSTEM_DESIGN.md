# Single-model wound analysis and pressure-injury grading — system design

This document specifies the structure, model, and training plan for Sorbed's
learned pathway: **one** guideline-aligned tissue-segmentation model whose output
is sufficient to generate the full Acıbadem HD_T86 pressure-injury report, paired
with a deterministic staging engine and a calibrated cross-check. It is the
engineering reference for the learned backend; the classical CPU pipeline runs
without any of it. Operational commands are in [`RUNBOOK.md`](RUNBOOK.md); metrics
definitions are in [`EVALUATE.md`](EVALUATE.md); data provenance is in
[`DATA_README.md`](DATA_README.md).

The design targets four non-negotiable constraints: offline / on-premises
inference (no cloud, no API keys — patient photos are sensitive health data under
KVKK/GDPR); ONNX-CPU deployability; guideline-faithful, auditable staging; and no
fabricated data or metrics. Every architectural choice below is made under those
constraints, not against an unconstrained leaderboard.

---

## 1. Overview and data flow

```
                 ┌──────────────────────────────────────────────┐
 wound photo ──▶ │  tissue segmenter  (SegFormer/MiT-b4, 7-class)│──▶ logits[1,7,H,W]
                 └──────────────────────────────────────────────┘
                        │ argmax + softmax
                        ▼
     ┌───────────────────────────────────────────────────────────┐
     │ derived fields (deterministic, no learning):                │
     │  • wound mask        = union of non-background classes       │
     │  • RYB composition   = per-class pixel fractions (§4.5.9)     │
     │  • size / morphometry= mask geometry (§4.5.11)               │
     │  • PUSH sub-scores   = area × exudate/tissue proxies (§4.5.8) │
     └───────────────────────────────────────────────────────────┘
                        │ tissue composition + morphometry
                        ▼
     ┌───────────────────────────────────────────────────────────┐
     │ staging engine (src/sorbed/staging):                        │
     │  rules.py  → HD_T86 tissue-driven stage + abstention        │
     │  ml_head.py→ transparent GBM over 15 clinical features       │
     │  arbiter.py→ hard-rule override, agreement/disagreement      │
     │  conformal.py (optional) → coverage-set depth abstention     │
     └───────────────────────────────────────────────────────────┘
                        ▼
             HD_T86 bilingual PDF report
```

The learned component is **one** network. Everything the report needs beyond the
tissue map is either a deterministic function of that map (mask, composition,
size, PUSH) or a rule/arbiter decision over it (stage). The stage is **derived,
not learned end-to-end** — the single most important design decision, justified
in §4.

---

## 2. The model

| Property | Value |
| --- | --- |
| Architecture | SegFormer decoder + MiT-b4 encoder (`segmentation-models-pytorch` 0.5.0) |
| Task | 7-class semantic segmentation |
| Input | RGB, 512×512, ImageNet-normalized, gray-world white-balanced |
| Output | `logits[1, 7, 512, 512]`, ONNX opset-17, CPU |
| Params | ≈64 M (encoder-dominated); exports with no custom ops |
| Config | [`configs/seg_tissue_segformer.yaml`](configs/seg_tissue_segformer.yaml) |
| Trainer | [`train_seg.py`](train_seg.py) `--num-classes 7 --arch segformer --encoder mit_b4` |

**Class order (ONNX channel index → tissue):**

| idx | class | RYB / HD_T86 role |
| --- | --- | --- |
| 0 | background | intact skin / off-wound |
| 1 | epithelial | red — re-epithelialising margin |
| 2 | granulation | red — healthy healing bed |
| 3 | slough | yellow — fibrin / non-viable soft tissue |
| 4 | eschar | black — necrotic |
| 5 | adipose | visible fat → forces ≥ Evre 3 |
| 6 | deep_structure | muscle / tendon / bone → forces Evre 4 |

**Backbone rationale.** A July-2026 cross-dataset wound-segmentation benchmark
reports that SegFormer/MiT leads cross-hospital generalization (SegFormer-B2 Dice
0.786 vs U-Net 0.737) with cross-initialization variance an order of magnitude
smaller than CNNs — cross-site *stability* is the property this task needs, and
the transformer family has it (arXiv:2607.02555 [reference verified at abstract
level]). MiT encoders export cleanly to ONNX; Mamba- and SAM-family segmenters do
not, and promptable models need a per-image operator prompt, so neither fits the
ONNX-CPU, fully-automated constraint. The backbone is at the practical frontier
for this regime; the one generation it is behind — self-supervised in-domain
pretraining — is addressed as a distillation-only roadmap item in §7, not on the
deployed path.

---

## 3. How each HD_T86 report field is produced

| Report field (HD_T86 §) | Source | Computation |
| --- | --- | --- |
| Wound mask | model | pixels where `argmax ≠ 0` |
| Tissue composition, RYB (§4.5.9) | model | per-class pixel fraction within the mask |
| Wound size (§4.5.11) | model + scale | mask bounding geometry × per-image fiducial scale |
| PUSH sub-scores (§4.5.8) | model | area band + tissue-type band from composition |
| Stage Evre 1–4 (§3.3–3.6) | rule engine | tissue-driven thresholds over composition + morphometry |
| Unstageable (§3.7) | rule engine | bed obscured by slough/eschar beyond threshold |
| Deep Tissue Injury (§3.8) | rule engine | maroon/purple on intact skin (colour + intactness) |
| Reassessment cadence (§4.5.1) | rule engine | stage → cadence table |

Size and PUSH require a real per-image scale from a fiducial; the pipeline never
guesses physical scale (see `EVALUATE.md`). Depth-dependent distinctions the
photo cannot support are abstained on — see §4 and §6.

---

## 4. Staging: derived, not learned

Human inter-rater agreement on pressure-injury staging is only κ≈0.57 (Fulbrook
2023; the widely cited 23–58% range corroborates this). Training an end-to-end
stager on labels that unreliable learns a corrupted feature extractor; the field's
high headline numbers are typically obtained by *deleting* the two categories a
photo cannot resolve (Deep Tissue Injury and Unstageable). Sorbed keeps them and
abstains. Staging is therefore a deterministic, auditable function:

- **`rules.py`** implements the HD_T86 tissue-driven logic: visible adipose →
  ≥ Evre 3; visible muscle/tendon/bone → Evre 4; bed obscured → Unstageable;
  maroon on intact skin → DTI. Each call is traceable to a guideline clause.
- **`ml_head.py`** is a *transparent* gradient-boosted classifier over a 15-dim
  clinical feature vector (not a neural net) — the heavy network's output stays
  inspectable pixels upstream. It ships untrained until a permissively-licensed
  staging dataset exists, and is a cross-check, never the source of truth.
- **`arbiter.py`** combines them: hard-rule outcomes (Unstageable, exposed-
  structure Stage 4, DTI, Indeterminate) always override the ML head; agreement
  raises confidence; disagreement lowers it and adds a caveat.
- **`conformal.py`** (optional, §7 item 1) adds a distribution-free coverage set:
  when the guaranteed set spans the depth-ambiguous stages, the system defers.

This is the most defensible part of the system and must not be replaced by a
learned stager.

---

## 5. Training plan

### 5.1 Data strategy — partial-label, multi-source

No public dataset carries wound + tissue + stage together, and tissue-mask images
total only ≈400. The design splits supervision by what each corpus can actually
teach:

- **Tissue classes (1–6)** are supervised by the tissue-mask sets — DFUTissue
  (110), WoundTissue (147), Wounds-307, ComplexWoundDB (27) — after each is
  re-labelled to the unified 7-class order.
- **Localization (class 0 vs foreground)** is supervised by the large binary
  wound corpora — AZH/FUSeg, Mendeley lower-limb (2686), CO2Wounds (~607), WSNet —
  through a **partial-label superset loss**: a binary mask's foreground is written
  as a sentinel (`SUPERSET_SENTINEL = -2`), excluded from cross-entropy/Dice, and
  contributes `-log P(foreground)` so it drives the wound-vs-background boundary
  without inventing a tissue class ([`losses.py`](losses.py) `MulticlassDiceCELoss`,
  [`data.py`](data.py) `SegmentationDataset`).

Each source declares `mask_kind: tissue | binary` in the data-prep config so the
manifest carries it and the trainer routes it automatically.

### 5.2 Data preparation pipeline

```
fetch_corpus.py                    # open sources -> /data/sorbed/corpus/<name>
arrange_tissue.py  (per tissue set)# native mask classes -> unified 7-class order
data_prep.py       (datasets.yaml) # leakage-free patient-level manifests
dedup.py                           # perceptual-hash near-duplicate removal
train_seg.py       (--manifest)    # partial-label 7-class training -> ONNX
external_validation.py             # score vs human ceiling + subgroups + conformal
```

`arrange_tissue.py` applies an operator-confirmed mapping
(`configs/tissue_maps/*.yaml`) and emits a `mapping_report.json` of native-value
pixel coverage; an unmapped value is an error by default, so an unknown tissue is
never silently folded into background. `dedup.py` removes photos shared across
sources (several public sets overlap) before the split, so no image leaks across
train/val and inflates the score.

### 5.3 Optimization

| Setting | Value | Note |
| --- | --- | --- |
| Loss | Dice + CE + superset term | class 0 = background, sentinel = −2 |
| Optimizer | AdamW, lr 6e-5, wd 0.01 | |
| Schedule | cosine with 5-epoch warmup | |
| Epochs | 200 | early-stop on val tissue-Dice |
| Batch / size | 12 @ 512² | fits one 40–71 GB MIG slice with AMP |
| Precision | AMP (bf16/fp16) | adaptive micro-batch on CUDA OOM |
| Split | grouped patient-level K-fold | `patient_id` from the manifest |
| Augmentation | flips, rotation, scale, mild photometric | large hue shifts would corrupt tissue colour cues |
| Memory safety | [`memory.py`](memory.py) | VRAM-fraction cap, `num_workers=0` on small `/dev/shm`, expandable segments |

Validation excludes sentinel pixels from per-class Dice (binary-source foreground
carries no tissue label), so tissue classes are scored only where they are truly
labelled.

### 5.4 Honest bottleneck

The wound-area number is not the limiting metric — 6-class tissue Dice on ≈400
masks is. Best-in-class tissue models reach ≈0.77 Dice on *three* tissues
(Nature s41598-025-06703-5 [abstract-level]); six classes on fewer masks will be
lower. The roadmap (§7) attacks this at its root (annotation, labeling format,
verified pseudo-labels), not with a bigger backbone.

---

## 6. Safety envelope — what the system must not claim

- **Depth it cannot see.** Evre 3-vs-4 and Unstageable hinge on depth/undermining
  and under-eschar tissue a 2D photo cannot confirm. The rule engine and the
  conformal gate abstain here rather than commit.
- **Domain gap.** Reported segmentation Dice (0.87) is foot-ulcer domain; pressure
  injuries are predominantly sacral/ischial/heel. That number must not be quoted
  as a pressure-injury capability — it is an internal foot-ulcer figure pending
  external validation (§7 item 0).
- **No external validation yet.** All learned metrics are internal, single-source.
- **No autonomous diagnosis.** Output is decision support requiring clinician
  sign-off; the report caveats every abstention and every depth-inferred stage.

---

## 7. Roadmap — evidence-ranked, constraint-preserving

Ranked by value-per-effort adjusted for overfitting risk on ≈400 masks. All items
preserve offline / ONNX-CPU / no-API / no-hallucination. Items marked ✅ are
implemented in this repository.

**0 — External PI validation set (~100–200 sacral/ischial/heel images,
multi-rater consensus).** Not a model change; it gates everything else — it makes
conformal calibration computable and every experiment falsifiable. The harness
that consumes it is implemented (✅ [`external_validation.py`](external_validation.py):
scores against the human κ ceiling and per-site subgroups); the data requires a
clinical partnership.

**1 — Ordinal conformal prediction sets. ✅** Post-hoc, distribution-free coverage
over the stage; defers when the coverage set spans depth-ambiguous stages
([`conformal.py`](conformal.py), `src/sorbed/staging/conformal.py`). Fitting needs
the item-0 calibration set; the code is ready.

**2 — Detector→segmenter cascade.** A small (permissively-licensed) box detector
localizes the wound before the frozen segmenter; reported to recover large
off-domain mIoU using cheap boxes rather than starved masks (arXiv:2505.23392
[abstract-level]). Insertion point: `src/sorbed/pipeline/analyzer.py`, mirroring
`onnx_backend.py`. License caveat: the method is sound; the specific detector
weights must be permissively licensed for on-prem clinical use.

**3 — Patch/superpixel-consistency labeling for rare tissue classes.** A training-
data/loss change (no deployment impact) attacking the tissue-Dice bottleneck at
its root; labeling *format* was found to matter as much as architecture at N=147
(arXiv:2502.10652 [abstract-level]).

**4 — SAM-family as an offline, human-verified pseudo-labeler** to grow the
400-mask corpus. Never on the deployed path (KVKK-safe). Human verification is
mandatory — zero-shot SAM on medical images is inconsistent.

**5 — Foundation-backbone gain via distillation only** (DINOv3/PanDerm teacher →
MiT-b4-sized student). The only constraint-preserving form: the teacher runs
offline at training time; the deployed student stays ONNX-CPU. Ship only if the
distilled student beats MiT-b4 on the item-0 external set — the reported ≈1–2%
gain is within the seed variance of a 400-image evaluation.

**Rejected** (SOTA-looking but vanity, unsafe, or non-deployable here): end-to-end
learned staging (destroys auditability, overfits label noise); multimodal-LLM
stagers (break offline/ONNX/CPU, add a hallucination surface to a clinical stage);
any promptable/SAM model on the deployed path (needs a per-image prompt);
RAD-DINO/BiomedParse as backbone (modality mismatch); a direct DINOv3/PanDerm swap
on the shipped path (CPU-latency regression for an in-domain-unproven gain);
standing TTA/ensembling (SegFormer's cross-init variance is already low).

**Do first:** acquire the item-0 validation set, then fit the conformal calibrator
(item 1), then the detector cascade (item 2). Do **not** spend the first
engineering effort swapping the backbone.

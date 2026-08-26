# Models, Datasets & Data Availability

This document describes the machine-learning landscape Sorbed operates in. Wound imaging is a **data-poor** field, and pressure-injury staging is the most data-scarce area within it. During the survey behind Sorbed's model choices, **no large, public, permissively-licensed pressure-injury *staging* dataset could be verified to exist.** The staging studies in the literature are trained on private, single-center data. This constraint shapes the recommendations below and is the reason Sorbed treats staging as decision support with explicit uncertainty rather than as a settled classification problem.

A disclaimer applies to every number in this document: the benchmark figures cited here are **author-reported and, in most cases, single-center.** They have not been independently reproduced by the Sorbed project. They are indicative of relative capability, not guarantees of performance on other data.

## Two very different tasks

Two problems that are often conflated are separated here:

1. **Wound boundary segmentation** — drawing the outline of the wound. This task is comparatively **mature**. Public data is reasonably good, and models routinely reach Dice scores in the **0.85–0.93** range.
2. **Tissue-type semantic segmentation** — labeling the bed as granulation, slough, eschar, epithelial tissue, and so on. This task is **hard**. Public data is small, class boundaries are ambiguous, and the common failure mode is **slough being confused with granulation** (yellow-tinged granulation and pink-tinged slough sit close together in color space).

Sorbed's architecture reflects this split: it is confident about *where* the wound is and applies explicit uncertainty to *what every pixel of it is made of*.

## Segmentation models

| Model | License | Notes |
|---|---|---|
| [SAM 2.1 Hiera](https://github.com/facebookresearch/sam2) | Apache-2.0 | Promptable and class-agnostic; suited to interactive use and for bootstrapping labels. Does not identify a wound on its own. |
| [MedSAM / MedSAM2](https://github.com/bowang-lab/MedSAM) | Apache-2.0 | Oriented toward CT/MR and 3D volumes; **not suited for 2D RGB clinical photos**. |
| [nnU-Net v2](https://github.com/MIC-DKFZ/nnUNet) | Apache-2.0 | Self-configuring; trains from scratch. Highest automatic ceiling of the models here, but requires a GPU and a labeled dataset. |
| DeepLabV3+ (torchvision) | BSD | Well-supported semantic segmentation backbone. |
| [segmentation_models.pytorch (SMP)](https://github.com/qubvel-org/segmentation_models.pytorch) | MIT | U-Net / DeepLab families with ImageNet encoders. A MobileNet encoder runs in **real time on CPU**. |
| WSNet | (wound-specific) | Reports Dice 0.847. |
| FUSegNet | (wound-specific) | Reports 92.7% Dice on foot ulcers. |
| DFUTissueSegNet | (wound-specific) | Tissue segmentation, trained on very little data. |

The **WoundAmbit** benchmark found that **SegNeXt** and **SegFormer** outperformed a plain U-Net, and that **all evaluated models ran at ≥1 image/second on CPU.** This indicates that CPU-only serving is viable for this workload.

## Datasets

The table below records size and license, because those two facts constrain what can be built and deployed.

| Dataset | Size | Content | License / availability |
|---|---|---|---|
| AZH Chronic Wound | 1,109 images | Binary wound masks | Public |
| FUSeg 2021 | 1,210 images | Wound masks | Public |
| [DFUC 2022](https://dfuc2022.grand-challenge.org/) | 4,000 images | Diabetic foot ulcer | **Non-commercial** license agreement |
| DFUTissue | 110 images | Tissue labels | Small |
| 6-tissue set (arXiv:2502.10652) | 147 images | Six tissue classes | Small |
| WoundSeg / WSNet | 2,686 images | 8 wound types, incl. pressure ulcer | — |
| Medetec | — | Clinical stock photos | **No formal license** ("free stock") |

**There is no verifiable public, permissively-licensed pressure-injury *staging* dataset.** The public data that exists is either boundary segmentation, non-pressure wounds (foot ulcers), or small tissue sets; the published staging work relies on private single-center collections that cannot be redistributed.

## Staging models

The reported accuracy figures for staging classifiers require context:

- **DenseNet121 ~93.7%** and **ResNet18 ~92.4%** on an **853-image single-center** set (JMIR, 2025). These numbers are **influenced by augmentation** and reflect one institution's imaging.
- **YOLOv8m** for mobile deployment: **84.6%**.
- **Thermal imaging: 95.45%** — this requires a thermal camera, which most deployments will not have.

**Human raters agree with one another only 23–58% of the time** on staging. A model that agrees with the labels is agreeing with a noisy ground truth. **Staging from a single 2D image is intrinsically hard.** Depth is ambiguous from RGB, and the **Unstageable and DTPI categories are, in principle, unresolvable from surface RGB alone** — Unstageable means the depth is hidden by definition, and DTPI's severity is beneath intact skin.

Sorbed's approach is to ship staging as **decision support with uncertainty**. Low-confidence results, and specifically **Unstageable and DTPI, are routed to a clinician** rather than reported as confident predictions.

## Tissue classification approaches

There are two broad families, with a trade-off between them:

- **Color-space methods** (HSV or Lab thresholds, the Red-Yellow-Black proxy) are **cheap, interpretable, and brittle.** They run anywhere, need no training data, and can be explained pixel by pixel, but they degrade under variable lighting, skin tone, and camera characteristics.
- **Learned semantic segmentation** is **higher-performing but data-limited.** Where it has enough data it performs clearly better — Swift's SmartTissue reports **94%**, but it is **proprietary** and not available to build on.

Given the limited data, Sorbed reports tissue as **percentage-of-area estimates with confidence, not hard per-pixel claims.** Reporting "roughly 60% granulation, 30% slough, 10% eschar, moderate confidence" reflects the data available rather than asserting a crisp boundary the data cannot support.

## Recommended stack

The recommended Sorbed model stack is:

- **Automatic boundary segmentation:** [SMP](https://github.com/qubvel-org/segmentation_models.pytorch) DeepLabV3+ or U-Net with an EfficientNet or MobileNet encoder, fine-tuned on **FUSeg + AZH**.
- **Interactive segmentation:** [SAM 2.1](https://github.com/facebookresearch/sam2) for clinician-in-the-loop prompting and label bootstrapping.
- **Serving:** PyTorch → ONNX → ONNX Runtime behind FastAPI, **CPU-first with oneDNN**, scaling out to Triton/GPU when available.
- **Medical metrics and transforms:** [MONAI](https://github.com/Project-MONAI/MONAI) (Apache-2.0).
- **Explainability:** [pytorch-grad-cam](https://github.com/jacobgil/pytorch-grad-cam) (MIT) and Captum (BSD) for the vision models, plus SHAP for the feature-based staging head.

Underneath this sits a **weight-free classical backend** that guarantees end-to-end operation with no model downloads — the classical path (from the [uwm-bigdata wound-segmentation](https://github.com/uwm-bigdata/wound-segmentation) lineage of approaches and simple color-space tissue analysis) always produces a result, including fully offline.

## Licensing summary

The **code dependencies are uniformly permissive** — Apache-2.0, MIT, and BSD throughout. The constraints are on **data**, not software: **DFUC is non-commercial**, and **Medetec carries no formal license.** Before any commercial deployment, **verify each dataset's license individually.** A permissive code stack does not make a non-commercial dataset commercially usable.

## Provenance & model cards

Sorbed **does not vendor model weights.** Each model is:

- downloaded only on **explicit request**,
- **sha256-verified** on download,
- and **recorded, with its license, in a model registry** that functions as a lightweight model card for every artifact the system can load.

Because a **weight-free classical backend is always available**, the system remains fully functional when no weights have been fetched.

---

## Running with real HuggingFace models (SAM / MedSAM)

Sorbed ships a learned segmentation backend that refines the wound mask with
a promptable Segment-Anything model from the HuggingFace Hub. The weight-free
classical proposal supplies a bounding-box prompt; the model returns a precise
mask and its own predicted IoU, which Sorbed uses as the segmentation confidence
(a model-derived value).

```bash
pip install -e '.[hf]'                        # torch + transformers + huggingface-hub
export SORBED_SEGMENTATION_BACKEND=hf_sam
export SORBED_HF_MODEL_ID=facebook/sam-vit-base          # or a medical variant:
# export SORBED_HF_MODEL_ID=flaviagiammarino/medsam-vit-base
sorbed analyze wound.jpg --mm-per-px 0.15 --out reports
```

Verified permissively-licensed Hub models usable as drop-in prompts:

| Model id | What it is | License |
|---|---|---|
| `facebook/sam-vit-base` | Segment Anything (ViT-B), promptable, class-agnostic | Apache-2.0 |
| `flaviagiammarino/medsam-vit-base` | MedSAM, box-prompted medical SAM | Apache-2.0 |

Weights download on first use and are cached under the HuggingFace cache; nothing
is vendored, and the resolved Hub commit is recorded in the report's provenance.

> **Network note.** The backend needs outbound access to `huggingface.co`. Some
> managed/sandboxed environments firewall it; in that case pre-download the model
> in a networked environment (`huggingface-cli download <id>`) and point
> `HF_HOME` at the shared cache, or run Sorbed where the Hub is reachable. Sorbed
> does not fabricate a result when a model cannot be loaded — it raises a clear
> error and the classical backend remains available offline.

## SOTA landscape (2024–2026)

A cited survey of the model and system designs Sorbed positions against. Figures
are from published literature; where a single source or an unverified value is
involved it is flagged.

### Wound-area segmentation

Public benchmarks exist here (unlike staging). The reference datasets are
the **AZH Chronic Wound** set (1,010 foot-ulcer images, UW-Milwaukee + AZH Wound
& Vascular Center) and the **MICCAI-2021 FUSeg Challenge** set (1,210 images from
889 patients; Wang et al., *MDPI Information* 15(3):140, 2024).

- **FUSegNet / x-FUSegNet** (Dhar et al., *Biomed. Signal Process. Control*, 2024;
  arXiv:2305.02961; `github.com/mrinal054/FUSegNet`). An encoder–decoder with a
  pretrained **EfficientNet-b7** encoder and a **P-scSE** ("parallel scSE")
  attention block in each decoder stage, combining *additive* and *max-out* scSE
  fusions. Reports **data-based DSC 92.70%** on AZH; the 5-fold ensemble
  **x-FUSegNet tops the FUSeg-2021 leaderboard at 89.23%**. (Image-based DSC is
  reported in the mid-80s; exact value unverified here.)
- **scSE** (Roy, Navab & Wachinger, MICCAI 2018; arXiv:1803.02579). Channel SE
  (cSE) gates *which channels* matter via global pooling; spatial SE (sSE) gates
  *where* via a 1×1 convolution; concurrent scSE fuses both and outperforms either
  alone on segmentation. Sorbed's trainer exposes scSE via `--decoder-attention`.
- Baselines in this literature: LinkNet/U-Net with EfficientNet or DenseNet
  backbones, DeepLabV3+, MANet, PSPNet, TransUNet.
- **Transformer encoders (2024→2026).** For 2D RGB wound photos the field has
  moved toward transformer backbones — **SegFormer / MiT**, Swin-UNet, and hybrids.
  The 2025 real-world wound benchmark **WoundAmbit** (ECML-PKDD 2025;
  arXiv:2504.06185) finds transformer backbones (TransNeXt, ConvNeXt, VWFormer,
  SegFormer) competitive-to-best; a HarDNet–transformer hybrid (arXiv:2410.03359)
  claims to surpass FUSegNet. Sorbed's trainer exposes this via `--arch segformer
  --encoder mit_b2/mit_b3` (ONNX-exportable), alongside the CNN FUSegNet recipe.
- **CNN baselines.** *nnU-Net Revisited* (MICCAI 2024;
  arXiv:2404.09556) shows that under matched validation, well-configured CNN U-Nets
  (ResEnc, MedNeXt, STU-Net) outperform transformer **and** Mamba variants — though that
  study is 3D-from-scratch, so for 2D pretrained wound photos transformer encoders
  remain a modern option. Both are presented; neither is universally
  superior.
- **State-space / Mamba (U-Mamba, VM-UNet arXiv:2402.02491, Swin-UMamba MICCAI
  2024, Mamba-UNet): not yet adopted.** The nnU-Net Revisited ablation found the Mamba
  layers themselves contributed no gain (the residual U-Net wrapper did); no mature
  ONNX/CPU path exists. Not used in Sorbed.

### Promptable foundation models

**SAM** (Kirillov et al., 2023), **SAM 2** (2024), **MedSAM** (Ma, Wang et al.,
*Nature Communications*, 2024), and **MedSAM-2** (2025) are
fundamentally **promptable** — a human supplies a box/point and quality is
prompt-sensitive; they do not assign semantic class labels. For an *unattended*
EHR upload, a dedicated supervised segmenter (FUSegNet-class) gives the automatic
mask that prompt-free operation needs; auto-prompting SAM variants
(Self-Prompt-SAM, MedSAM-U) are an active but not-yet-canonical direction, and no
widely-benchmarked wound-specific SAM fine-tune was identified as of early 2026.
**BiomedParse** (Microsoft; *Nature Methods* 2025, arXiv:2405.12971) is a
text-promptable joint segment/detect/recognise model across nine modalities, but
GPU/text-driven, not a drop-in CPU-ONNX single-class wound masker.
These are not part of the shipped CPU pipeline.

### Photo-based staging

- **MDPI *Applied Sciences* 2024, 14(16):7124** (Chang et al.): an on-device
  **YOLOv8** pipeline that localizes and classifies six severities (Stage 1–4,
  DTPI, Unstageable) on 2,800 images — **YOLOv8m: 84.6% accuracy, mAP@50 90.8%**.
- **JMIR Med Inform 2025, e62774**: CNNs on 853→7,677 augmented images;
  **DenseNet121 93.71%** best (single-source figure — verify).

**Central validity caveat.** Staging is defined by *depth* and *what tissue is
visible*. A flat RGB photo has no true depth and cannot see under obscuring
slough/eschar, so Stage 3/4, DTPI, and Unstageable are intrinsically error-prone
from photography; high reported accuracies are usually on curated, class-balanced
sets and do not transfer to ambiguous real-world cases. For this reason Sorbed damps
depth-dependent grades and abstains rather than forcing them.

### Skin-tone equity

Stage 1 and DTPI are defined by **erythema**, which is harder to see in Fitzpatrick
V–VI skin (injury presents as hyperpigmentation instead), so photo models trained
on light-skinned data systematically under-detect them (NPIAP, *Perspectives on
pressure injuries in dark skin tones*; AJN 2023). Mitigations: tone-robust sensing
(long-wave thermography detects abnormalities at comparable rates across tones —
85% of Fitzpatrick I–III vs 82% of IV–VI; PMC12689484), SEM devices, deliberate
Fitzpatrick/ITA sampling, and **per-tone stratified metric reporting**. For a
**Türkiye** deployment the population is predominantly Fitzpatrick II–IV, so the
erythema-visibility gap is less extreme than in more diverse settings, but it is
not eliminated (the darker, Fitzpatrick-IV end and darker-skinned residents still
warrant the ITA safeguard), and local re-validation with per-tone metrics remains
the appropriate practice.

### System & regulatory framing

Interoperability via **HL7 FHIR** (Observation/DiagnosticReport for mobile wound
photos) and **DICOM** (now with an AI-Results object type + IHE AI-Workflow
profiles). AI output is kept as a clinician-confirmed **draft** with an audit trail;
calibrated confidence is reported and the system **abstains/routes-to-clinician** on low-confidence
or out-of-distribution inputs; a **model card** is published (Mitchell et al., 2019).
Regulatory: software that outputs a specific stage/recommendation is generally
**Software as a Medical Device** — US **FDA** 510(k) + Predetermined Change
Control Plans (Jan 2025 draft AI guidance); **EU** MDR 2017/745 + the AI Act
(medical AI = high-risk). A **Türkiye** deployment falls under the Turkish
medical-device regulation (Tıbbi Cihaz Yönetmeliği), which is **harmonised with
EU MDR 2017/745** and administered by **TİTCK** (Türkiye İlaç ve Tıbbi Cihaz
Kurumu); positioning the tool as clinician decision support with human-in-the-loop
confirmation and abstention keeps it out of autonomous-diagnosis territory.

### Longitudinal metrics

Percent area reduction and the validated **4-week PAR predictor** (~50% for
diabetic-foot ulcers, Sheehan et al., *Diabetes Care* 2003; ~40% for venous leg
ulcers, Kantor & Margolis); **Gilman** perimeter-normalized healing rate
(size/shape-independent; Gilman, *Wounds* 1990); and the structured scores
**PUSH** (0–17), **BWAT**, and **DESIGN-R**. Sorbed computes PAR, the 4-week
predictor, the Gilman rate, and a partial PUSH — see [`TREND.md`](TREND.md).

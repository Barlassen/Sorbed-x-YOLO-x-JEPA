# Released model weights

Trained ONNX models for Sorbed's optional learned backends. The files here are
tracked with [Git LFS](https://git-lfs.com); run `git lfs install` once before
cloning or pulling. The classical CPU pipeline does not require these files.

## Contents

| File | Model | Input | Output |
| --- | --- | --- | --- |
| `seg_unetpp.onnx` | U-Net++ / EfficientNetV2-S wound segmenter | `(1, 3, 768, 768)` RGB, ImageNet-normalized | `(1, 1, 768, 768)` foreground logits |
| `grade_convnextv2.onnx` | ConvNeXt-V2 stage grader, CORN ordinal head | `(1, 3, 288, 288)` RGB, ImageNet-normalized | `(1, 3)` CORN threshold logits (stages 1–4) |

## Provenance and metrics

| Model | Training data | Validation metric | sha256 |
| --- | --- | --- | --- |
| `seg_unetpp.onnx` | AZH Chronic Wound + MICCAI-2021 FUSeg (foot ulcers) | Internal-validation Dice 0.87 | `4e9cae8af67f90df3a6a90509693006779d8c16ad0cd04bb0ed83d46e1c435cb` |
| `grade_convnextv2.onnx` | PIID (EPUAP stages 1–4; 1,087 images after de-duplication) | Fold-0 quadratic-weighted κ 0.911; 5-fold κ 0.919 ± 0.006; balanced accuracy ≈ 0.80 | `8918e1147c77a6a1e7003bdd4861f1bab1fb63267aae587250b02c4de95a169f` |

Grader calibration: temperature scaling with T = 3.05 reduces expected calibration
error from 0.149 to 0.039 on the validation fold. The CORN logits should be divided
by 3.05 before the sigmoid at inference to obtain calibrated probabilities.

## Scope

Metrics are single-institution internal validation. PIID is EPUAP-staged and has no
external test set. The segmenter is trained on diabetic-foot ulcers, which differ
from pressure injuries in anatomical site, tissue, and depth. Reproduction steps are
in [`training/RUNBOOK.md`](../../training/RUNBOOK.md); evaluation is described in
[`training/EVALUATE.md`](../../training/EVALUATE.md).

## Use

```bash
export SORBED_SEGMENTATION_BACKEND=onnx
export SORBED_ONNX_MODEL=models/release/seg_unetpp.onnx
sorbed analyze wound.jpg --out reports
```

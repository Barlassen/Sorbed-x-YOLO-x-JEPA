"""Sorbed training stack — segmentation, stage grading, and teacher–student.

This top-level ``training`` package holds the *heavy* GPU training pipelines that
are intentionally kept out of the shipped ``sorbed`` runtime package (which stays
CPU-only and dependency-light). Everything here targets a single ~40 GB VRAM
slice (H200 MIG) at 768–1024 px with AMP, cosine schedules, patient-level
cross-validation, checkpointing, TensorBoard, and ONNX export.

Install the training extras first::

    pip install torch torchvision segmentation-models-pytorch timm tensorboard \
        pyyaml opencv-python-headless

Modules
-------
``training.data``
    Datasets (paired segmentation, stage-grading manifest, unlabeled pool),
    weak/strong augmentation, and patient-level group K-fold splitting.
``training.datasets`` / ``training.data_prep``
    Multi-source manifest schema (image/mask/body-part/stage/patient/visit/
    source/license), dataset adapters (AZH·FUSeg, generic), leakage-free
    patient-level splits, and the ``data_prep`` CLI that writes train/val/test
    manifests + ``summary.json``.
``training.losses``
    Dice+BCE, multiclass Dice, focal, ordinal CORN, distillation KL, and
    consistency losses.
``training.models``
    Model factories: SegFormer/MiT + U-Net++/MAnet segmenters (via
    ``segmentation-models-pytorch``) and ConvNeXt-V2/ViT graders (via ``timm``),
    plus ONNX export helpers.
``training.utils``
    Seeding, device/AMP helpers, cosine-with-warmup schedule, EMA (mean teacher),
    checkpoint management, TensorBoard, and YAML config loading.
``training.train_seg``
    Segmentation trainer CLI.
``training.train_grade``
    Stage-grading classifier trainer CLI.
``training.teacher_student``
    Directive-driven rule teacher (pseudo-labeller with abstention) plus student
    distillation and FixMatch/mean-teacher consistency semi-supervision.
``training.finetune_medsam``
    SAM/MedSAM mask-decoder fine-tune made prompt-free (learned or heuristic
    auto-box), with an eval that compares auto-SAM against the supervised
    ``sorbed`` segmenter. Needs the ``hf`` extra (``transformers``).
``training.evaluate``
    Publication-grade metrics: segmentation (Dice/IoU/HD95/ASSD) and grading
    (balanced accuracy, quadratic-weighted κ, per-stage sensitivity, ECE/Brier)
    with patient-clustered bootstrap CIs.
``training.benchmark``
    Compares the learned grader against the directive rule-engine baseline on a
    validation fold and renders a montage of graded images with their wound-mask
    overlays.

None of these modules is imported by the ``sorbed`` runtime; the CPU inference
path never depends on torch.
"""

from __future__ import annotations

__all__ = ["__version__"]

__version__ = "0.1.0"

"""Model factories and ONNX export for segmentation and stage grading.

Segmentation uses ``segmentation-models-pytorch`` (SMP); grading uses ``timm``.
Both export to a clean static-graph ONNX (standard conv/attention ops, opset 17)
that ``onnxruntime`` runs on CPU — the deciding deployment constraint.

Recommended recipes (H200 MIG, ~40 GB, 768–1024 px)
---------------------------------------------------
* **Segmentation:** ``arch="segformer"``, ``encoder="mit_b3"`` (or ``mit_b4``).
  SMP's SegFormer/MiT is the pragmatic winner: trains comfortably on one slice,
  exports frictionlessly to ONNX-CPU. U-Net++/MAnet are available but require a
  *CNN* encoder (e.g. ``tu-tf_efficientnetv2_s``) — MiT is SegFormer-only in SMP.
* **Grading:** ``convnextv2_base.fcmae_ft_in22k_in1k_384`` at 384 px. Pure conv,
  quantizes well for CPU, and beats same-budget ViTs on small medical sets. Use
  ``eva02_base_patch14_448.mim_in22k_ft_in1k`` only with a large labeled set.

nnU-Net note
------------
nnU-Net v2 (``nnUNetResEncUNetLPlans``) sets the accuracy *ceiling* but has no
first-class ONNX path — its value is the self-configuring preprocess/patch/TTA
pipeline that a naked graph export loses. Use it as an offline reference model,
not the shipping CPU model. See ``training/configs/seg_nnunet_reference.yaml``.
"""

from __future__ import annotations

from pathlib import Path

import segmentation_models_pytorch as smp
import timm
import torch
from torch import nn

# SMP architectures that accept a MiT (SegFormer) encoder.
_MIT_ONLY_ARCHS = frozenset({"segformer"})


def build_segmenter(
    *,
    arch: str = "segformer",
    encoder_name: str = "mit_b3",
    encoder_weights: str | None = "imagenet",
    in_channels: int = 3,
    classes: int = 1,
    decoder_attention: str | None = None,
    gradient_checkpointing: bool = False,
) -> nn.Module:
    """Construct an SMP segmentation model.

    ``encoder_weights=None`` builds untrained weights (offline / CI). ``classes``
    is 1 for binary wound masks or ``C`` for multiclass tissue segmentation.
    ``decoder_attention="scse"`` is honored only by the U-Net family.
    """
    if encoder_name.startswith("mit_") and arch not in _MIT_ONLY_ARCHS:
        raise ValueError(
            f"encoder {encoder_name!r} (MiT) is only valid with arch='segformer' in SMP; "
            f"use a CNN encoder (e.g. 'tu-tf_efficientnetv2_s') for arch={arch!r}"
        )
    kwargs: dict[str, object] = {
        "encoder_name": encoder_name,
        "encoder_weights": encoder_weights,
        "in_channels": in_channels,
        "classes": classes,
    }
    if arch in {"unet", "unetplusplus"} and decoder_attention:
        kwargs["decoder_attention_type"] = decoder_attention
    model = smp.create_model(arch, **kwargs)
    if gradient_checkpointing:
        enable_gradient_checkpointing(model)
    return model


def build_grader(
    *,
    model_name: str = "convnextv2_base.fcmae_ft_in22k_in1k_384",
    num_classes: int = 6,
    pretrained: bool = True,
    drop_rate: float = 0.0,
    drop_path_rate: float = 0.1,
    gradient_checkpointing: bool = False,
) -> nn.Module:
    """Construct a timm classification backbone with a fresh ``num_classes`` head.

    For CORN ordinal grading over Stages 1–4, pass ``num_classes = K - 1`` (the
    number of ordinal thresholds); for plain CE/focal pass the full class count.
    """
    model = timm.create_model(
        model_name,
        pretrained=pretrained,
        num_classes=num_classes,
        drop_rate=drop_rate,
        drop_path_rate=drop_path_rate,
    )
    if gradient_checkpointing and hasattr(model, "set_grad_checkpointing"):
        model.set_grad_checkpointing(True)
    return model


def grader_data_config(model: nn.Module) -> dict[str, object]:
    """Resolve timm's mean/std/input-size for a grader (for parity at inference)."""
    return dict(timm.data.resolve_model_data_config(model))


def enable_gradient_checkpointing(model: nn.Module) -> bool:
    """Best-effort activation of gradient checkpointing on an SMP model.

    timm-backed SMP encoders (``tu-*``) expose ``set_grad_checkpointing`` on their
    inner module; native SMP encoders (including MiT) do not, in which case this
    returns ``False`` and memory must be controlled via batch size + AMP instead.
    """
    encoder = getattr(model, "encoder", None)
    if encoder is None:
        return False
    if hasattr(encoder, "set_grad_checkpointing"):
        encoder.set_grad_checkpointing(True)
        return True
    inner = getattr(encoder, "model", None)
    if inner is not None and hasattr(inner, "set_grad_checkpointing"):
        inner.set_grad_checkpointing(True)
        return True
    return False


def export_onnx(
    model: nn.Module,
    out_path: Path,
    *,
    input_size: int,
    input_names: tuple[str, ...] = ("input",),
    output_names: tuple[str, ...] = ("output",),
    dynamic_batch: bool = True,
    opset: int = 17,
    device: torch.device | None = None,
) -> Path:
    """Export ``model`` to ONNX with a ``(1, 3, H, W)`` example input.

    ``dynamic_batch`` marks axis 0 dynamic so the graph serves any batch size on
    CPU; height/width stay fixed (SMP/timm graphs are simplest at static spatial
    dims). The model is switched to ``eval`` and exported on CPU by default.
    """
    device = device or torch.device("cpu")
    model = model.to(device).eval()
    example = torch.randn(1, 3, input_size, input_size, device=device)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    dynamic_axes = None
    if dynamic_batch:
        dynamic_axes = {name: {0: "batch"} for name in (*input_names, *output_names)}
    # Use the stable TorchScript exporter (dynamo=False) so opset_version is
    # honored exactly and the graph stays static — the layout onnxruntime-CPU and
    # sorbed's OnnxSegmenter expect. The dynamo exporter silently upgrades opsets.
    torch.onnx.export(
        model,
        example,
        str(out_path),
        input_names=list(input_names),
        output_names=list(output_names),
        opset_version=opset,
        dynamic_axes=dynamic_axes,
        dynamo=False,
    )
    return out_path

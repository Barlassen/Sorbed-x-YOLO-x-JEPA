"""Loss functions for wound segmentation, stage grading, and distillation.

All losses operate on raw logits (no sigmoid/softmax applied upstream) and return
a scalar tensor. They are AMP-safe: reductions run in the autocast dtype but the
soft-Dice numerator/denominator use a small epsilon to stay finite.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional


def soft_dice_coefficient(
    probs: torch.Tensor,
    target: torch.Tensor,
    *,
    dims: tuple[int, ...],
    smooth: float = 1.0,
) -> torch.Tensor:
    """Soft Dice over ``dims`` (elementwise probs and target in ``[0, 1]``)."""
    intersection = (probs * target).sum(dim=dims)
    cardinality = probs.sum(dim=dims) + target.sum(dim=dims)
    return (2.0 * intersection + smooth) / (cardinality + smooth)


class DiceBCELoss(nn.Module):
    """Binary Dice + BCE for single-foreground-channel segmentation.

    Expects ``logits`` of shape ``(N, 1, H, W)`` and float ``target`` in
    ``{0, 1}`` of the same shape.
    """

    def __init__(self, *, bce_weight: float = 1.0, dice_weight: float = 1.0,
                 smooth: float = 1.0) -> None:
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.smooth = smooth

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = functional.binary_cross_entropy_with_logits(logits, target)
        probs = torch.sigmoid(logits)
        dice = soft_dice_coefficient(probs, target, dims=(1, 2, 3), smooth=self.smooth)
        dice_loss = 1.0 - dice.mean()
        return self.bce_weight * bce + self.dice_weight * dice_loss


class MulticlassDiceCELoss(nn.Module):
    """Multiclass Dice + cross-entropy for tissue-class segmentation.

    Expects ``logits`` of shape ``(N, C, H, W)`` and integer ``target`` of shape
    ``(N, H, W)`` with values in ``[0, C)``. ``ignore_index`` pixels are excluded
    from both terms.

    **Partial-label (superset) supervision.** For multi-dataset training where a
    source provides only a *wound-vs-background* mask (no tissue class), set
    ``superset_index``: pixels with that sentinel value are known to be foreground
    (any class other than ``background_index``) but not which one. Those pixels are
    excluded from CE/Dice and instead add ``-log(P(foreground))`` — driving the
    background probability down without inventing a tissue label. This lets the
    large binary wound corpora supervise localization while the small tissue-mask
    sets supervise the classes. ``superset_index=None`` reproduces the plain loss.
    """

    def __init__(self, *, num_classes: int, ce_weight: float = 1.0,
                 dice_weight: float = 1.0, ignore_index: int = -100,
                 smooth: float = 1.0, superset_index: int | None = None,
                 background_index: int = 0, superset_weight: float = 1.0) -> None:
        super().__init__()
        self.num_classes = num_classes
        self.ce_weight = ce_weight
        self.dice_weight = dice_weight
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.superset_index = superset_index
        self.background_index = background_index
        self.superset_weight = superset_weight

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        # Superset pixels carry no class label; exclude them from CE/Dice (treat
        # them like ignore) and add the foreground-superset term separately below.
        ce_target = target
        if self.superset_index is not None:
            ce_target = torch.where(
                target == self.superset_index,
                torch.full_like(target, self.ignore_index), target)
        probs = logits.softmax(dim=1)
        valid = ce_target != self.ignore_index
        # cross_entropy is NaN when every pixel is ignored (a fully binary-mask
        # image, all pixels superset); contribute 0 CE in that case.
        if bool(valid.any()):
            ce = functional.cross_entropy(logits, ce_target, ignore_index=self.ignore_index)
        else:
            ce = logits.new_zeros(())
        safe_target = torch.where(valid, ce_target, torch.zeros_like(ce_target))
        one_hot = functional.one_hot(safe_target, num_classes=self.num_classes)
        one_hot = one_hot.permute(0, 3, 1, 2).to(probs.dtype)
        mask = valid.unsqueeze(1).to(probs.dtype)
        dice = soft_dice_coefficient(probs * mask, one_hot * mask,
                                     dims=(0, 2, 3), smooth=self.smooth)
        dice_loss = 1.0 - dice.mean()
        loss = self.ce_weight * ce + self.dice_weight * dice_loss
        if self.superset_index is not None:
            fg_pixels = target == self.superset_index
            if bool(fg_pixels.any()):
                p_fg = (1.0 - probs[:, self.background_index]).clamp_min(1e-6)
                loss = loss + self.superset_weight * (-p_fg.log())[fg_pixels].mean()
        return loss


class BinaryFocalLoss(nn.Module):
    """Focal loss (Lin et al., 2017) for binary segmentation logits."""

    def __init__(self, *, alpha: float = 0.25, gamma: float = 2.0) -> None:
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = functional.binary_cross_entropy_with_logits(logits, target, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * target + (1.0 - probs) * (1.0 - target)
        alpha_t = self.alpha * target + (1.0 - self.alpha) * (1.0 - target)
        loss = alpha_t * (1.0 - p_t).pow(self.gamma) * bce
        return loss.mean()


class FocalCrossEntropy(nn.Module):
    """Multiclass focal loss for classification logits ``(N, C)``.

    ``class_weights`` (length ``C``) re-weights rare stages such as Stage 4 /
    Unstageable, which are clinically critical and prevalence-scarce.
    """

    def __init__(self, *, gamma: float = 2.0,
                 class_weights: torch.Tensor | None = None) -> None:
        super().__init__()
        self.gamma = gamma
        self.register_buffer("class_weights", class_weights)

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        log_probs = functional.log_softmax(logits, dim=1)
        ce = functional.nll_loss(log_probs, target, weight=self.class_weights, reduction="none")
        p_t = log_probs.gather(1, target.unsqueeze(1)).squeeze(1).exp()
        loss = (1.0 - p_t).pow(self.gamma) * ce
        return loss.mean()


class CornOrdinalLoss(nn.Module):
    """CORN conditional-ordinal loss (Shi, Cao & Raschka, 2023).

    The head emits ``num_classes - 1`` binary logits; rank ``k`` predicts
    ``P(y > k | y >= k)``. Appropriate for a *totally ordered* label set such as
    NPIAP Stage 1 < 2 < 3 < 4. Side classes (Unstageable / DTI) are not ordinal
    and should be handled by a separate head, not folded onto this axis.
    """

    def __init__(self, num_classes: int) -> None:
        super().__init__()
        if num_classes < 2:
            raise ValueError("CORN needs at least 2 ordinal classes")
        self.num_classes = num_classes

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if logits.shape[1] != self.num_classes - 1:
            raise ValueError(
                f"CORN expects {self.num_classes - 1} logits, got {logits.shape[1]}"
            )
        losses: list[torch.Tensor] = []
        for k in range(self.num_classes - 1):
            subset = target >= k
            if not bool(subset.any()):
                continue
            binary_target = (target[subset] > k).float()
            log_prob = functional.logsigmoid(logits[subset, k])
            log_one_minus = functional.logsigmoid(-logits[subset, k])
            term = binary_target * log_prob + (1.0 - binary_target) * log_one_minus
            losses.append(-term.mean())
        if not losses:
            return logits.sum() * 0.0
        return torch.stack(losses).mean()


def corn_logits_to_probs(logits: torch.Tensor) -> torch.Tensor:
    """Convert CORN conditional logits ``(N, K-1)`` to class probs ``(N, K)``."""
    conditional = torch.sigmoid(logits)
    cumulative = torch.cumprod(conditional, dim=1)
    n = cumulative.shape[0]
    ones = torch.ones(n, 1, dtype=cumulative.dtype, device=cumulative.device)
    padded = torch.cat([ones, cumulative], dim=1)
    survive = padded
    probs = survive[:, :-1] - survive[:, 1:]
    last = survive[:, -1:]
    return torch.cat([probs, last], dim=1).clamp_min(0.0)


class DistillationKLLoss(nn.Module):
    """Temperature-scaled KL divergence from a soft teacher to the student.

    ``student_logits`` and ``teacher_probs`` are ``(N, C)``. ``teacher_probs`` is
    an already-normalized soft target (the rule teacher's soft stage
    distribution). ``sample_weight`` (``(N,)``) optionally down-weights
    low-confidence teacher outputs; abstained rows should be masked out by the
    caller (given zero weight or dropped) since they carry no pseudo-label.
    """

    def __init__(self, temperature: float = 2.0) -> None:
        super().__init__()
        if temperature <= 0:
            raise ValueError("temperature must be positive")
        self.temperature = temperature

    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_probs: torch.Tensor,
        *,
        sample_weight: torch.Tensor | None = None,
    ) -> torch.Tensor:
        t = self.temperature
        student_log = functional.log_softmax(student_logits / t, dim=1)
        teacher = teacher_probs.clamp_min(1e-8)
        teacher = teacher / teacher.sum(dim=1, keepdim=True)
        per_sample = functional.kl_div(student_log, teacher, reduction="none").sum(dim=1)
        # T^2 keeps gradient magnitude comparable across temperatures.
        per_sample = per_sample * (t * t)
        if sample_weight is not None:
            denom = sample_weight.sum().clamp_min(1e-8)
            return (per_sample * sample_weight).sum() / denom
        return per_sample.mean()


class SoftmaxConsistencyLoss(nn.Module):
    """Symmetric MSE consistency between two softmax predictions.

    Used for mean-teacher / colour-geometry consistency: the student's prediction
    under a strong augmentation should match the (EMA-teacher's) prediction under
    a weak augmentation. Operates on probabilities, so it is bounded and stable.
    """

    def forward(self, logits_a: torch.Tensor, logits_b: torch.Tensor) -> torch.Tensor:
        probs_a = logits_a.softmax(dim=1)
        probs_b = logits_b.softmax(dim=1)
        return functional.mse_loss(probs_a, probs_b)


def fixmatch_pseudo_label(
    logits_weak: torch.Tensor,
    *,
    threshold: float = 0.95,
) -> tuple[torch.Tensor, torch.Tensor]:
    """FixMatch pseudo-labels from weak-view logits.

    Returns ``(pseudo_labels, mask)`` where ``mask`` is 1.0 for samples whose max
    softmax confidence clears ``threshold`` and 0.0 otherwise. Detached — this is
    a target, not a differentiable path.
    """
    with torch.no_grad():
        probs = logits_weak.softmax(dim=1)
        confidence, pseudo = probs.max(dim=1)
        mask = (confidence >= threshold).float()
    return pseudo, mask


def masked_cross_entropy(
    logits: torch.Tensor,
    pseudo: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Cross-entropy against pseudo-labels, averaged over confident samples."""
    per_sample = functional.cross_entropy(logits, pseudo, reduction="none")
    denom = mask.sum().clamp_min(1e-8)
    return (per_sample * mask).sum() / denom

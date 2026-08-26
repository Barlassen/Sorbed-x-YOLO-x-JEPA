#!/usr/bin/env python3
"""Mask-guided JEPA pre-training for wound representations (skeleton).

This is the self-supervised stage of the "YOLO26 ⊕ JEPA" plan. Standard I-JEPA
learns by predicting the latent representation of *randomly* masked image blocks
from a visible context block. Here the masking is **guided by the wound mask**
that the fine-tuned YOLO26 segmenter produces: the prediction *targets* are patches
that fall inside the wound, and the *context* is the surrounding tissue. The model
is therefore pushed to encode "given the healthy skin around it, what is the wound
like" — a wound-specific representation, learned from unlabeled images.

    image ──▶ YOLO26-seg ──▶ wound mask ──┐
                                          ▼
    image ──▶ patch grid ──▶ [ context = outside mask ] ──▶ encoder ─┐
                             [ targets = inside mask   ] ──▶ EMA enc ─┤ (stop-grad)
                                          ▼                          ▼
                              predictor(context, target-pos) ≈ EMA targets   → L2 loss

Pipeline role: stage 2 of 3. Stage 1 fine-tunes YOLO26-seg on FUSeg
(``prepare_yolo_seg.py``); stage 3 attaches an ordinal grading head onto the
encoder trained here (adapt ``train_grade.py``).

Status: a runnable **scaffold**, not a tuned trainer. The architecture wires up
and a smoke step runs on CPU with random data (``python -m training.train_jepa
--smoke``). The real run — a proper ViT backbone, the unlabeled wound corpus, an
AdamW/cosine schedule, and many epochs — belongs on the GPU pod. TODOs mark the
seams that must be filled there.
"""

from __future__ import annotations

import argparse
import copy
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
@dataclass
class JepaConfig:
    img_size: int = 224
    patch_size: int = 16
    enc_dim: int = 192          # tiny by default; use vit-base dims on the pod
    enc_depth: int = 6
    enc_heads: int = 3
    pred_dim: int = 96
    pred_depth: int = 4
    pred_heads: int = 3
    ema_momentum: float = 0.996  # target-encoder EMA decay

    @property
    def grid(self) -> int:
        return self.img_size // self.patch_size

    @property
    def num_patches(self) -> int:
        return self.grid * self.grid


class PatchEmbed(nn.Module):
    """Non-overlapping conv patchifier: (B,3,H,W) -> (B, N, D)."""

    def __init__(self, cfg: JepaConfig) -> None:
        super().__init__()
        self.proj = nn.Conv2d(3, cfg.enc_dim, cfg.patch_size, stride=cfg.patch_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.proj(x)                    # (B, D, Gh, Gw)
        return x.flatten(2).transpose(1, 2)  # (B, N, D)


def _sincos_pos_embed(num_patches: int, dim: int) -> torch.Tensor:
    """Fixed 1-D sin-cos positional embedding, (1, N, D). Good enough for a scaffold."""
    pos = torch.arange(num_patches).unsqueeze(1)
    i = torch.arange(dim // 2).unsqueeze(0)
    freq = torch.exp(-math.log(10000.0) * (2 * i) / dim)
    table = torch.zeros(num_patches, dim)
    table[:, 0::2] = torch.sin(pos * freq)
    table[:, 1::2] = torch.cos(pos * freq)
    return table.unsqueeze(0)


class TransformerTrunk(nn.Module):
    """A stack of pre-norm transformer encoder blocks."""

    def __init__(self, dim: int, depth: int, heads: int) -> None:
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=dim, nhead=heads, dim_feedforward=dim * 4,
            dropout=0.0,  # MPS's scaled_dot_product_attention rejects attention dropout
            activation="gelu", batch_first=True, norm_first=True,
        )
        self.blocks = nn.TransformerEncoder(layer, num_layers=depth)
        self.norm = nn.LayerNorm(dim)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.norm(self.blocks(tokens))


class Encoder(nn.Module):
    """Patch-embed + positional embed + transformer trunk over a token subset."""

    def __init__(self, cfg: JepaConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.patch_embed = PatchEmbed(cfg)
        self.register_buffer("pos_embed", _sincos_pos_embed(cfg.num_patches, cfg.enc_dim))
        self.trunk = TransformerTrunk(cfg.enc_dim, cfg.enc_depth, cfg.enc_heads)

    def tokens_with_pos(self, images: torch.Tensor) -> torch.Tensor:
        return self.patch_embed(images) + self.pos_embed

    def forward(self, images: torch.Tensor, keep_idx: torch.Tensor | None = None) -> torch.Tensor:
        """Encode all patches, or only the ``keep_idx`` subset (B, K) when given."""
        tokens = self.tokens_with_pos(images)
        if keep_idx is not None:
            tokens = _gather(tokens, keep_idx)
        return self.trunk(tokens)


class Predictor(nn.Module):
    """Predict target-patch latents from encoded context + target positions."""

    def __init__(self, cfg: JepaConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.in_proj = nn.Linear(cfg.enc_dim, cfg.pred_dim)
        self.out_proj = nn.Linear(cfg.pred_dim, cfg.enc_dim)
        self.mask_token = nn.Parameter(torch.zeros(1, 1, cfg.pred_dim))
        nn.init.normal_(self.mask_token, std=0.02)
        self.register_buffer("pos_embed", _sincos_pos_embed(cfg.num_patches, cfg.pred_dim))
        self.trunk = TransformerTrunk(cfg.pred_dim, cfg.pred_depth, cfg.pred_heads)

    def forward(self, context: torch.Tensor, ctx_idx: torch.Tensor, tgt_idx: torch.Tensor) -> torch.Tensor:
        # Context tokens carry their own encoded content; target slots are mask
        # tokens + target positional embeddings. The trunk lets targets attend to
        # context, then we read the target slots back out to encoder dimensions.
        ctx = self.in_proj(context) + _gather(self.pos_embed.expand(context.size(0), -1, -1), ctx_idx)
        tgt = self.mask_token + _gather(self.pos_embed.expand(context.size(0), -1, -1), tgt_idx)
        n_ctx = ctx.size(1)
        fused = self.trunk(torch.cat([ctx, tgt], dim=1))
        return self.out_proj(fused[:, n_ctx:])  # (B, n_target, enc_dim)


class MaskGuidedJEPA(nn.Module):
    """Online encoder + EMA target encoder + predictor."""

    def __init__(self, cfg: JepaConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder = Encoder(cfg)
        self.target_encoder = copy.deepcopy(self.encoder)
        for p in self.target_encoder.parameters():
            p.requires_grad_(False)
        self.predictor = Predictor(cfg)

    @torch.no_grad()
    def update_target(self) -> None:
        m = self.cfg.ema_momentum
        for online, target in zip(self.encoder.parameters(), self.target_encoder.parameters()):
            target.mul_(m).add_(online.detach(), alpha=1 - m)

    def forward(self, images: torch.Tensor, ctx_idx: torch.Tensor, tgt_idx: torch.Tensor) -> torch.Tensor:
        context = self.encoder(images, keep_idx=ctx_idx)              # online, with grad
        with torch.no_grad():
            full_target = self.target_encoder(images)                 # EMA, all patches
            targets = _gather(full_target, tgt_idx)                   # stop-grad targets
        preds = self.predictor(context, ctx_idx, tgt_idx)
        return F.smooth_l1_loss(preds, targets)


def _gather(tokens: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Gather tokens (B, N, D) at per-sample indices idx (B, K) -> (B, K, D)."""
    b, _, d = tokens.shape
    return torch.gather(tokens, 1, idx.unsqueeze(-1).expand(b, idx.size(1), d))


# --------------------------------------------------------------------------- #
# Mask-guided block sampling — the heart of the "YOLO mask -> JEPA" idea
# --------------------------------------------------------------------------- #
def sample_indices(
    wound_masks: torch.Tensor, cfg: JepaConfig, n_target: int, n_context: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Choose target patches inside the wound and context patches outside it.

    ``wound_masks`` is (B, H, W) in {0,1} (from YOLO26-seg). It is pooled onto the
    patch grid; a patch is "wound" when its coverage exceeds 0.5. Targets are
    sampled from wound patches (falling back to random patches when an image has
    too few — e.g. YOLO found nothing), context from the remaining patches. This
    is what makes the pretext task wound-centric rather than uniform.
    """
    b = wound_masks.size(0)
    g = cfg.grid
    patch_cov = F.adaptive_avg_pool2d(wound_masks.unsqueeze(1).float(), (g, g)).flatten(1)  # (B, N)

    tgt_list, ctx_list = [], []
    for i in range(b):
        cov = patch_cov[i]
        wound_idx = torch.nonzero(cov > 0, as_tuple=False).flatten()  # any overlap counts
        if wound_idx.numel() >= n_target:
            # Prefer the *most*-wound patches, so even a small wound (few covered
            # patches) still steers the targets — the point of the mask guidance.
            order = torch.argsort(cov[wound_idx], descending=True)
            tgt = wound_idx[order[:n_target]]
        else:
            # Too little wound (or YOLO found nothing): keep what we have, fill random.
            tgt = _choose(wound_idx, n_target, cfg.num_patches, avoid=None)
        other_idx = torch.nonzero(cov == 0, as_tuple=False).flatten()
        ctx = _choose(other_idx, n_context, cfg.num_patches, avoid=tgt)
        tgt_list.append(tgt)
        ctx_list.append(ctx)
    return torch.stack(ctx_list), torch.stack(tgt_list)


def _choose(pool: torch.Tensor, k: int, num_patches: int, avoid: torch.Tensor | None) -> torch.Tensor:
    """Pick k indices from ``pool``; fall back to any patch when the pool is short."""
    if pool.numel() < k:
        everything = torch.arange(num_patches)
        if avoid is not None:
            mask = torch.ones(num_patches, dtype=torch.bool)
            mask[avoid] = False
            everything = everything[mask]
        extra = everything[torch.randperm(everything.numel())][: k - pool.numel()]
        pool = torch.cat([pool, extra])
    return pool[torch.randperm(pool.numel())][:k]


# --------------------------------------------------------------------------- #
# Data — unlabeled wound images paired with a wound mask
# --------------------------------------------------------------------------- #
_IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
_IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
_IMG_EXTS = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}


class WoundJepaDataset(Dataset):
    """Unlabeled wound images, each paired with a wound mask for guided sampling.

    JEPA is self-supervised — no stage/label is needed. The mask (the "where is
    the wound" signal that steers the pretext task) comes from one of two sources:

    * ``masks_dir`` — precomputed mask PNGs matched by file stem. Use this for the
      fine-tuned YOLO26-seg masks dumped to disk, or ground-truth masks (e.g.
      FUSeg's) while developing.
    * ``mask_fn`` — a callable ``(image_bgr_uint8) -> (H, W) array`` computed on the
      fly, e.g. a live YOLO segmenter (see :func:`yolo_mask_fn`). Used only when
      ``masks_dir`` is None.

    An image whose mask is missing/empty still trains — ``sample_indices`` falls
    back to random targets, so pretraining degrades gracefully rather than crashing.
    """

    def __init__(self, images_dir, masks_dir=None, mask_fn=None, img_size: int = 224) -> None:
        self.img_size = img_size
        self.paths = sorted(p for p in Path(images_dir).iterdir() if p.suffix.lower() in _IMG_EXTS)
        if not self.paths:
            raise FileNotFoundError(f"no images under {images_dir}")
        self.masks_dir = Path(masks_dir) if masks_dir else None
        self.mask_fn = mask_fn
        if self.masks_dir is None and self.mask_fn is None:
            raise ValueError("provide either masks_dir (precomputed) or mask_fn (live)")

    def __len__(self) -> int:
        return len(self.paths)

    def _mask_for(self, path: Path, bgr: np.ndarray) -> np.ndarray:
        if self.masks_dir is not None:
            m = cv2.imread(str(self.masks_dir / f"{path.stem}.png"), cv2.IMREAD_GRAYSCALE)
            if m is None:
                m = np.zeros(bgr.shape[:2], np.uint8)
        else:
            m = self.mask_fn(bgr)
        return (m > 0).astype(np.float32)

    def __getitem__(self, i: int):
        path = self.paths[i]
        bgr = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if bgr is None:
            raise RuntimeError(f"could not read image {path}")
        mask = self._mask_for(path, bgr)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (self.img_size, self.img_size), interpolation=cv2.INTER_LINEAR)
        mask = cv2.resize(mask, (self.img_size, self.img_size), interpolation=cv2.INTER_NEAREST)
        img = torch.from_numpy(rgb).permute(2, 0, 1).float() / 255.0
        img = (img - _IMAGENET_MEAN) / _IMAGENET_STD
        return img, torch.from_numpy(mask)


def yolo_mask_fn(weights: str, conf: float = 0.25):
    """Wrap a fine-tuned YOLO26-seg model as a ``(image_bgr) -> binary mask`` fn."""
    from ultralytics import YOLO

    model = YOLO(weights)

    def _fn(bgr: np.ndarray) -> np.ndarray:
        h, w = bgr.shape[:2]
        result = model.predict(bgr, conf=conf, verbose=False)[0]
        if result.masks is None or len(result.masks.data) == 0:
            return np.zeros((h, w), np.uint8)
        union = result.masks.data.cpu().numpy().max(axis=0)
        union = cv2.resize(union.astype(np.float32), (w, h), interpolation=cv2.INTER_LINEAR)
        return (union >= 0.5).astype(np.uint8)

    return _fn


def build_dataloader(
    images_dir, *, masks_dir=None, yolo_weights=None, img_size=224,
    batch_size=16, workers=2, shuffle=True,
) -> DataLoader:
    """Build a DataLoader of (image, wound_mask). Live YOLO masks force workers=0
    (the model is not fork-safe across worker processes)."""
    mask_fn = None
    if masks_dir is None:
        if not yolo_weights:
            raise ValueError("give --masks (a mask dir) or --yolo-weights (live masks)")
        mask_fn = yolo_mask_fn(yolo_weights)
        workers = 0
    dataset = WoundJepaDataset(images_dir, masks_dir=masks_dir, mask_fn=mask_fn, img_size=img_size)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                      num_workers=workers, drop_last=True)


# --------------------------------------------------------------------------- #
# Training loop (skeleton)
# --------------------------------------------------------------------------- #
def train(cfg: JepaConfig, loader, *, epochs: int, lr: float, device: str) -> MaskGuidedJEPA:
    """Minimal loop. TODO on the pod: AdamW + cosine warmup, AMP, ckpt, logging."""
    model = MaskGuidedJEPA(cfg).to(device)
    opt = torch.optim.AdamW(
        list(model.encoder.parameters()) + list(model.predictor.parameters()), lr=lr
    )
    n_target = max(1, cfg.num_patches // 8)
    n_context = cfg.num_patches // 2
    for epoch in range(epochs):
        running, n = 0.0, 0
        for images, wound_masks in loader:
            # Sample indices on CPU (nonzero/argsort/ per-sample loop are cheap and
            # reliable there; running them on MPS is slow and flaky), then move only
            # the images and the small index tensors to the device.
            ctx_idx, tgt_idx = sample_indices(wound_masks, cfg, n_target, n_context)
            images = images.to(device)
            ctx_idx, tgt_idx = ctx_idx.to(device), tgt_idx.to(device)
            loss = model(images, ctx_idx, tgt_idx)
            opt.zero_grad()
            loss.backward()
            opt.step()
            model.update_target()
            running += loss.item()
            n += 1
        print(f"epoch {epoch + 1}/{epochs}  loss={running / max(n, 1):.4f}")
    return model


# --------------------------------------------------------------------------- #
# Smoke test — proves the architecture wires up on CPU with no data
# --------------------------------------------------------------------------- #
def _smoke() -> None:
    torch.manual_seed(0)
    cfg = JepaConfig()
    model = MaskGuidedJEPA(cfg)
    b = 2
    images = torch.randn(b, 3, cfg.img_size, cfg.img_size)
    # fake "YOLO" masks: a filled square stands in for a wound region.
    masks = torch.zeros(b, cfg.img_size, cfg.img_size)
    masks[:, 60:150, 60:150] = 1.0
    ctx_idx, tgt_idx = sample_indices(masks, cfg, n_target=cfg.num_patches // 8, n_context=cfg.num_patches // 2)
    loss = model(images, ctx_idx, tgt_idx)
    loss.backward()
    model.update_target()
    params = sum(p.numel() for p in model.encoder.parameters())
    print(f"[smoke] ok — encoder params={params/1e6:.2f}M  "
          f"context={ctx_idx.size(1)} targets={tgt_idx.size(1)}  loss={loss.item():.4f}")


def _default_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def main() -> None:
    ap = argparse.ArgumentParser(description="Mask-guided JEPA wound pre-training.")
    ap.add_argument("--smoke", action="store_true", help="Forward/backward on random data, then exit.")
    ap.add_argument("--images", type=Path, help="Directory of (unlabeled) wound images.")
    ap.add_argument("--masks", type=Path, help="Directory of precomputed mask PNGs (stem-matched).")
    ap.add_argument("--yolo-weights", help="Fine-tuned YOLO26-seg weights to make masks live (if --masks absent).")
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--out", type=Path, default=Path("runs_jepa/jepa_encoder.pt"),
                    help="Where to save the pre-trained encoder.")
    ap.add_argument("--device", default=_default_device())
    args = ap.parse_args()

    if args.smoke:
        _smoke()
        return

    if not args.images:
        raise SystemExit("give --images DIR (+ --masks DIR or --yolo-weights FILE), or --smoke.")

    cfg = JepaConfig(img_size=args.img_size)
    loader = build_dataloader(
        args.images, masks_dir=args.masks, yolo_weights=args.yolo_weights,
        img_size=args.img_size, batch_size=args.batch, workers=args.workers,
    )
    print(f"device={args.device}  images={len(loader.dataset)}  batches/epoch={len(loader)}")
    model = train(cfg, loader, epochs=args.epochs, lr=args.lr, device=args.device)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.encoder.state_dict(), args.out)
    print(f"saved encoder -> {args.out}  (attach a grading head in stage 3)")


if __name__ == "__main__":
    main()

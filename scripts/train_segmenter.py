#!/usr/bin/env python3
"""Train a wound-boundary segmentation U-Net and export it to ONNX.

This is a real, runnable training pipeline. It is intentionally *not* executed as
part of the Sorbed test/CI run because it needs the heavy optional stack (PyTorch
+ segmentation-models-pytorch). Install those first::

    pip install "sorbed[ml]"           # if the extra is defined, or:
    pip install torch torchvision segmentation-models-pytorch

All torch / smp imports are lazy (inside ``main``), so importing this module — or
linting it — never requires those packages.

Data layout
-----------
Point ``--images`` at a directory of RGB wound photos and ``--masks`` at a
directory of single-channel binary masks (wound = nonzero). Files are paired by
stem, so ``images/0007.png`` pairs with ``masks/0007.png`` (the mask extension may
differ). Example::

    dataset/
      images/  0001.png 0002.png ...
      masks/   0001.png 0002.png ...   # 0/255 or 0/1, wound = nonzero

Getting data
------------
Two public, permissively-usable binary wound-segmentation sets work directly:

* **AZH Chronic Wound** (1,109 images) and **FUSeg 2021** (1,210 images), both
  distributed from https://github.com/uwm-bigdata/wound-segmentation — clone that
  repo and use its ``data/`` image/label folders. Foot-ulcer boundary masks
  transfer reasonably to general wound boundaries; see docs/MODELS.md for the
  honest caveats about the data-poverty reality of this field.

Output
------
The best-by-validation-Dice checkpoint is written to ``--out-dir`` as
``best.pt``, and the final model is exported to ``model.onnx`` with a fixed
``(1, 3, H, W)`` input and a ``(1, 1, H, W)`` foreground-logit output — exactly
the layout :class:`sorbed.segmentation.backends.onnx_backend.OnnxSegmenter` reads.
Register the exported file's SHA-256 in ``models/registry.json`` (see
docs/TRAINING.md) or point ``SORBED_ONNX_MODEL`` straight at it.
"""

from __future__ import annotations

import argparse
from pathlib import Path

_IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train a U-Net wound segmenter and export it to ONNX.",
    )
    parser.add_argument("--images", type=Path, required=True, help="RGB image directory")
    parser.add_argument("--masks", type=Path, required=True, help="binary mask directory")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/segmenter"))
    parser.add_argument("--epochs", type=int, default=40)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--input-size", type=int, default=512, help="square H=W input")
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument(
        "--arch",
        type=str,
        default="unet",
        choices=["unet", "unetplusplus", "deeplabv3plus", "manet", "segformer"],
        help="Segmentation architecture (segmentation-models-pytorch). 'unet' + an "
             "EfficientNet encoder + scSE is the FUSegNet-line CNN recipe; 'segformer' "
             "with a MiT encoder (--encoder mit_b2/mit_b3) is the modern transformer "
             "recipe. All export to ONNX for CPU inference.",
    )
    parser.add_argument("--encoder", type=str, default="efficientnet-b0")
    parser.add_argument(
        "--encoder-weights",
        type=str,
        default="imagenet",
        help="Pretrained encoder weights ('imagenet') or 'none' to train from scratch "
        "(use 'none' in environments without access to the weight host).",
    )
    parser.add_argument(
        "--decoder-attention",
        type=str,
        default="scse",
        choices=["none", "scse"],
        help="Decoder attention. 'scse' adds spatial-and-channel Squeeze-and-Excitation "
             "blocks (Roy et al., MICCAI 2018) to each decoder stage — the mechanism the "
             "FUSegNet line uses to reach SOTA on the AZH/FUSeg chronic-wound benchmark.",
    )
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="'auto', 'cpu', or 'cuda'",
    )
    return parser.parse_args(argv)


def _pair_paths(images_dir: Path, masks_dir: Path) -> list[tuple[Path, Path]]:
    """Pair each image with the mask that shares its stem."""
    masks_by_stem: dict[str, Path] = {}
    for mask in masks_dir.iterdir():
        if mask.suffix.lower() in _IMAGE_SUFFIXES:
            masks_by_stem[mask.stem] = mask
    pairs: list[tuple[Path, Path]] = []
    for image in sorted(images_dir.iterdir()):
        if image.suffix.lower() not in _IMAGE_SUFFIXES:
            continue
        mask = masks_by_stem.get(image.stem)
        if mask is not None:
            pairs.append((image, mask))
    if not pairs:
        raise SystemExit(
            f"no image/mask pairs found between {images_dir} and {masks_dir}"
        )
    return pairs


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    # Lazy, heavy imports — kept out of module import so linting needs no torch.
    import cv2
    import numpy as np
    import segmentation_models_pytorch as smp
    import torch
    from torch import nn
    from torch.utils.data import DataLoader, Dataset

    size = int(args.input_size)
    mean = np.asarray(_IMAGENET_MEAN, dtype=np.float32).reshape(3, 1, 1)
    std = np.asarray(_IMAGENET_STD, dtype=np.float32).reshape(3, 1, 1)

    class WoundDataset(Dataset):
        """Pairs of resized image tensors and binary mask tensors."""

        def __init__(self, pairs: list[tuple[Path, Path]], *, augment: bool) -> None:
            self._pairs = pairs
            self._augment = augment

        def __len__(self) -> int:
            return len(self._pairs)

        def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
            image_path, mask_path = self._pairs[index]
            bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if bgr is None:
                raise RuntimeError(f"could not read image {image_path}")
            rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = cv2.resize(rgb, (size, size), interpolation=cv2.INTER_LINEAR)

            mask_raw = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
            if mask_raw is None:
                raise RuntimeError(f"could not read mask {mask_path}")
            mask_raw = cv2.resize(mask_raw, (size, size), interpolation=cv2.INTER_NEAREST)
            mask = (mask_raw > 0).astype(np.float32)

            if self._augment and bool(torch.rand(1).item() < 0.5):
                rgb = np.ascontiguousarray(rgb[:, ::-1, :])
                mask = np.ascontiguousarray(mask[:, ::-1])

            img = rgb.astype(np.float32) / 255.0
            img = np.transpose(img, (2, 0, 1))
            img = (img - mean) / std
            image_tensor = torch.from_numpy(np.ascontiguousarray(img))
            mask_tensor = torch.from_numpy(mask[np.newaxis, ...].copy())
            return image_tensor, mask_tensor

    def dice_coefficient(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        probs = torch.sigmoid(logits)
        preds = (probs >= 0.5).float()
        intersection = (preds * target).sum(dim=(1, 2, 3))
        union = preds.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        dice = (2.0 * intersection + 1.0) / (union + 1.0)
        return dice.mean()

    def dice_bce_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        bce = nn.functional.binary_cross_entropy_with_logits(logits, target)
        probs = torch.sigmoid(logits)
        intersection = (probs * target).sum(dim=(1, 2, 3))
        union = probs.sum(dim=(1, 2, 3)) + target.sum(dim=(1, 2, 3))
        soft_dice = (2.0 * intersection + 1.0) / (union + 1.0)
        return bce + (1.0 - soft_dice.mean())

    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    torch.manual_seed(args.seed)
    generator = torch.Generator().manual_seed(args.seed)

    pairs = _pair_paths(args.images, args.masks)
    permutation = torch.randperm(len(pairs), generator=generator).tolist()
    pairs = [pairs[i] for i in permutation]
    n_val = max(1, round(len(pairs) * args.val_fraction))
    val_pairs = pairs[:n_val]
    train_pairs = pairs[n_val:]
    if not train_pairs:
        raise SystemExit("not enough data for a training split; add more images")

    train_loader = DataLoader(
        WoundDataset(train_pairs, augment=True),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        WoundDataset(val_pairs, augment=False),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    encoder_weights = None if args.encoder_weights.lower() == "none" else args.encoder_weights
    decoder_attention = None if args.decoder_attention == "none" else args.decoder_attention
    # scSE decoder attention is only defined for the U-Net family; other
    # architectures (SegFormer, DeepLabV3+, MAnet) ignore it.
    model_kwargs: dict = {
        "encoder_name": args.encoder,
        "encoder_weights": encoder_weights,
        "in_channels": 3,
        "classes": 1,
    }
    if args.arch in ("unet", "unetplusplus") and decoder_attention is not None:
        model_kwargs["decoder_attention_type"] = decoder_attention
    model = smp.create_model(args.arch, **model_kwargs).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    best_ckpt = args.out_dir / "best.pt"
    best_dice = -1.0

    for epoch in range(1, args.epochs + 1):
        model.train()
        running = 0.0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = dice_bce_loss(logits, masks)
            loss.backward()
            optimizer.step()
            running += float(loss.item()) * images.size(0)
        train_loss = running / len(train_loader.dataset)

        model.eval()
        val_dice = 0.0
        with torch.no_grad():
            for images, masks in val_loader:
                images = images.to(device)
                masks = masks.to(device)
                logits = model(images)
                val_dice += float(dice_coefficient(logits, masks).item()) * images.size(0)
        val_dice /= len(val_loader.dataset)

        print(
            f"epoch {epoch:03d}/{args.epochs}  "
            f"train_loss={train_loss:.4f}  val_dice={val_dice:.4f}"
        )

        if val_dice > best_dice:
            best_dice = val_dice
            torch.save(
                {
                    "model_state": model.state_dict(),
                    "arch": args.arch,
                    "encoder": args.encoder,
                    "decoder_attention": decoder_attention,
                    "input_size": size,
                    "val_dice": val_dice,
                    "epoch": epoch,
                },
                best_ckpt,
            )

    print(f"best validation Dice: {best_dice:.4f} (checkpoint: {best_ckpt})")

    # Restore the best weights and export to ONNX for the ONNX Runtime backend.
    state = torch.load(best_ckpt, map_location=device)
    model.load_state_dict(state["model_state"])
    model.eval()

    onnx_path = args.out_dir / "model.onnx"
    example = torch.randn(1, 3, size, size, device=device)
    torch.onnx.export(
        model,
        example,
        str(onnx_path),
        input_names=["input"],
        output_names=["logits"],
        opset_version=17,
        dynamic_axes=None,
    )
    print(f"exported ONNX model to {onnx_path}")

    digest = _sha256_file(onnx_path)
    print(f"model.onnx sha256: {digest}")
    print(
        "Register this digest in models/registry.json, or run with "
        f"SORBED_ONNX_MODEL={onnx_path}"
    )
    return 0


def _sha256_file(path: Path) -> str:
    import hashlib

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


if __name__ == "__main__":
    raise SystemExit(main())

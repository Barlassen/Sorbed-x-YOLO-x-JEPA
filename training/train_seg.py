#!/usr/bin/env python3
"""Train a wound segmenter (SegFormer/MiT or U-Net++/MAnet) and export to ONNX.

A real, runnable pipeline with AMP, cosine-with-warmup scheduling, optional
patient-level cross-validation, checkpointing, TensorBoard, gradient
checkpointing, and ONNX export. Defaults target one ~40 GB H200 MIG slice at
768 px; drop the batch size (or raise ``--input-size`` to 1024) as needed.

Data
----
``--images`` / ``--masks`` are directories paired by file stem (wound = nonzero
for binary; pixel value = class index for multiclass). The best public binary
sets are AZH + FUSeg from ``github.com/uwm-bigdata/wound-segmentation`` (foot
ulcers, open); fetch them with ``scripts/fetch_fuseg.py``.

Patient-level CV
----------------
Pass ``--patient-manifest`` (CSV: ``stem,patient_id``) and ``--folds N`` to run a
grouped K-fold split so no patient leaks across train/val; ``--fold`` selects the
fold. Without a manifest the split is a seeded random hold-out (``--val-fraction``).

Example
-------
    python -m training.train_seg --config training/configs/seg_segformer.yaml \\
        --images data/fuseg/train/images --masks data/fuseg/train/labels \\
        --out-dir artifacts/seg_segformer
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import DataLoader

from training import data as data_mod
from training import memory, utils
from training.data import SUPERSET_SENTINEL
from training.losses import DiceBCELoss, MulticlassDiceCELoss
from training.models import build_segmenter, export_onnx


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=None, help="YAML config; CLI overrides it")
    parser.add_argument("--images", type=Path, default=None)
    parser.add_argument("--masks", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None,
                        help="data_prep manifest (CSV/JSONL) to train on a combined, "
                        "multi-source corpus instead of a single --images/--masks pair; "
                        "uses the manifest's patient_id for the grouped split")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--arch", type=str, default=None,
                        choices=["segformer", "unet", "unetplusplus", "manet", "deeplabv3plus"])
    parser.add_argument("--encoder", type=str, default=None)
    parser.add_argument("--encoder-weights", type=str, default=None,
                        help="'imagenet' or 'none' (offline / from scratch)")
    parser.add_argument("--decoder-attention", type=str, default=None, choices=["none", "scse"])
    parser.add_argument("--num-classes", type=int, default=None, help="1 = binary wound mask")
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--warmup-epochs", type=int, default=None)
    parser.add_argument("--val-fraction", type=float, default=None)
    parser.add_argument("--patient-manifest", type=Path, default=None,
                        help="CSV 'stem,patient_id' enabling grouped K-fold CV")
    parser.add_argument("--folds", type=int, default=None, help="K for grouped K-fold")
    parser.add_argument("--fold", type=int, default=None, help="fold index to train")
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--vram-fraction", type=float, default=None,
                        help="cap the process to this fraction (0-1] of MIG VRAM")
    parser.add_argument("--amp", dest="amp", action="store_true", default=None)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--grad-checkpointing", dest="grad_checkpointing",
                        action="store_true", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None, help="auto|cpu|cuda|cuda:N")
    parser.add_argument("--no-onnx", dest="export_onnx", action="store_false", default=None)
    return parser.parse_args(argv)


_DEFAULTS: dict[str, Any] = {
    "arch": "segformer",
    "encoder": "mit_b3",
    "encoder_weights": "imagenet",
    "decoder_attention": "none",
    "num_classes": 1,
    "input_size": 768,
    "epochs": 60,
    "batch_size": 8,
    "lr": 6e-5,
    "weight_decay": 0.01,
    "warmup_epochs": 3,
    "val_fraction": 0.2,
    "folds": 5,
    "fold": 0,
    "num_workers": 4,
    "vram_fraction": None,
    "amp": True,
    "grad_checkpointing": False,
    "seed": 1234,
    "device": "auto",
    "export_onnx": True,
    "out_dir": "artifacts/segmenter",
}


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    file_config = utils.load_yaml_config(args.config) if args.config else {}
    cli = {k: v for k, v in vars(args).items() if k != "config"}
    merged = utils.merge_cli_over_config({**_DEFAULTS, **file_config}, cli)
    if merged.get("manifest"):
        merged["manifest"] = Path(merged["manifest"])
    else:
        for required in ("images", "masks"):
            if merged.get(required) is None:
                raise SystemExit(f"--{required} (or --manifest) is required (via CLI or config)")
        merged["images"] = Path(merged["images"])
        merged["masks"] = Path(merged["masks"])
    merged["out_dir"] = Path(merged["out_dir"])
    return merged


def _load_patient_groups(manifest: Path, stems: list[str]) -> list[str]:
    mapping: dict[str, str] = {}
    with Path(manifest).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or {"stem", "patient_id"} - set(reader.fieldnames):
            raise SystemExit(f"patient manifest {manifest} needs columns stem,patient_id")
        for row in reader:
            mapping[row["stem"].strip()] = row["patient_id"].strip()
    missing = [s for s in stems if s not in mapping]
    if missing:
        raise SystemExit(
            f"{len(missing)} images missing from patient manifest (e.g. {missing[:3]})"
        )
    return [mapping[s] for s in stems]


def splits_from_manifest(
    cfg: dict[str, Any],
) -> tuple[
    list[tuple[Path, Path]], list[tuple[Path, Path]], list[bool], list[bool]
]:
    """Build train/val pairs and per-pair binary-mask flags from a manifest.

    Only rows with a ``mask_path`` are used (segmentation needs masks); the split
    is grouped by the manifest's ``patient_id`` so no source's patient leaks. Each
    record's ``mask_kind`` decides its flag: ``"tissue"`` masks carry real class
    indices (flag ``False``), everything else is treated as a wound-vs-background
    mask (flag ``True``) that supplies partial-label supervision via the loss's
    superset term. Returns ``(train_pairs, val_pairs, train_binary, val_binary)``.
    """
    from training.datasets import read_manifest

    records = [r for r in read_manifest(Path(cfg["manifest"])) if r.mask_path]
    if not records:
        raise SystemExit(f"no masked rows in manifest {cfg['manifest']}")
    pairs = [(Path(r.image_path), Path(r.mask_path)) for r in records]
    binary = [(r.mask_kind or "binary").lower() != "tissue" for r in records]
    groups = [r.patient_id for r in records]
    folds = data_mod.group_kfold_indices(groups, n_splits=int(cfg["folds"]), seed=int(cfg["seed"]))
    train_idx, val_idx = folds[int(cfg["fold"]) % len(folds)]
    n_tissue = sum(1 for b in binary if not b)
    print(f"manifest: {len(records)} masked rows from {len({r.source for r in records})} "
          f"source(s); {n_tissue} tissue-labeled, {len(records) - n_tissue} binary")
    return (
        [pairs[i] for i in train_idx], [pairs[i] for i in val_idx],
        [binary[i] for i in train_idx], [binary[i] for i in val_idx],
    )


def build_splits(
    pairs: list[tuple[Path, Path]],
    cfg: dict[str, Any],
) -> tuple[list[tuple[Path, Path]], list[tuple[Path, Path]]]:
    """Return ``(train_pairs, val_pairs)`` via grouped K-fold or random hold-out."""
    if cfg.get("patient_manifest"):
        stems = [img.stem for img, _ in pairs]
        groups = _load_patient_groups(Path(cfg["patient_manifest"]), stems)
        folds = data_mod.group_kfold_indices(groups, n_splits=int(cfg["folds"]),
                                              seed=int(cfg["seed"]))
        train_idx, val_idx = folds[int(cfg["fold"]) % len(folds)]
        return [pairs[i] for i in train_idx], [pairs[i] for i in val_idx]
    generator = torch.Generator().manual_seed(int(cfg["seed"]))
    order = torch.randperm(len(pairs), generator=generator).tolist()
    shuffled = [pairs[i] for i in order]
    n_val = max(1, round(len(shuffled) * float(cfg["val_fraction"])))
    return shuffled[n_val:], shuffled[:n_val]


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device,
             num_classes: int) -> float:
    """Mean foreground Dice over the validation loader."""
    model.eval()
    meter = utils.AverageMeter()
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        logits = memory.safe_infer(model, images)
        if num_classes == 1:
            preds = (torch.sigmoid(logits) >= 0.5).float()
            inter = (preds * targets).sum(dim=(1, 2, 3))
            union = preds.sum(dim=(1, 2, 3)) + targets.sum(dim=(1, 2, 3))
            dice = (2.0 * inter + 1.0) / (union + 1.0)
            meter.update(float(dice.mean().item()), images.size(0))
        else:
            preds = logits.argmax(dim=1)
            # Exclude superset (unknown-class) pixels from partial-label masks:
            # they carry no tissue label, so they score neither class.
            valid = (targets != SUPERSET_SENTINEL).float()
            per_class = []
            for cls in range(1, num_classes):
                p = (preds == cls).float() * valid
                t = (targets == cls).float() * valid
                inter = (p * t).sum(dim=(1, 2))
                union = p.sum(dim=(1, 2)) + t.sum(dim=(1, 2))
                per_class.append(((2.0 * inter + 1.0) / (union + 1.0)).mean())
            meter.update(float(torch.stack(per_class).mean().item()), images.size(0))
    return meter.average


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = build_config(args)
    utils.set_seed(int(cfg["seed"]))
    device = utils.resolve_device(str(cfg["device"]))
    memory.configure(device, vram_fraction=cfg.get("vram_fraction"))
    num_workers = memory.safe_num_workers(int(cfg["num_workers"]))
    num_classes = int(cfg["num_classes"])
    size = int(cfg["input_size"])

    train_binary: list[bool] | None = None
    val_binary: list[bool] | None = None
    if cfg.get("manifest"):
        train_pairs, val_pairs, train_binary, val_binary = splits_from_manifest(cfg)
    else:
        pairs = data_mod.pair_by_stem(cfg["images"], cfg["masks"])
        train_pairs, val_pairs = build_splits(pairs, cfg)
    if not train_pairs or not val_pairs:
        raise SystemExit("empty train or val split; add more data or adjust the split")

    # Partial-label superset supervision only applies to a multiclass run that
    # actually mixes in binary (wound-vs-background) sources.
    superset_index = (
        SUPERSET_SENTINEL
        if num_classes > 1 and train_binary is not None and any(train_binary)
        else None
    )
    if superset_index is not None:
        print(f"partial-label superset supervision on (sentinel={superset_index})")

    train_loader = DataLoader(
        data_mod.SegmentationDataset(train_pairs, input_size=size,
                                     num_classes=num_classes, augment=True,
                                     superset_index=superset_index,
                                     binary_flags=train_binary),
        batch_size=int(cfg["batch_size"]), shuffle=True,
        num_workers=num_workers, drop_last=True, pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        data_mod.SegmentationDataset(val_pairs, input_size=size,
                                     num_classes=num_classes, augment=False,
                                     superset_index=superset_index,
                                     binary_flags=val_binary),
        batch_size=int(cfg["batch_size"]), shuffle=False,
        num_workers=num_workers, pin_memory=device.type == "cuda",
    )

    encoder_weights = None if str(cfg["encoder_weights"]).lower() == "none" \
        else str(cfg["encoder_weights"])
    decoder_attention = None if str(cfg["decoder_attention"]) == "none" \
        else str(cfg["decoder_attention"])
    model = build_segmenter(
        arch=str(cfg["arch"]), encoder_name=str(cfg["encoder"]),
        encoder_weights=encoder_weights, classes=num_classes,
        decoder_attention=decoder_attention,
        gradient_checkpointing=bool(cfg["grad_checkpointing"]),
    ).to(device)
    print(f"model: {cfg['arch']}/{cfg['encoder']} | params={utils.count_parameters(model):,} "
          f"| device={device} | classes={num_classes}")

    criterion: torch.nn.Module = DiceBCELoss() if num_classes == 1 \
        else MulticlassDiceCELoss(num_classes=num_classes,
                                  superset_index=superset_index, background_index=0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]),
                                  weight_decay=float(cfg["weight_decay"]))
    steps_per_epoch = max(1, len(train_loader))
    total_steps = steps_per_epoch * int(cfg["epochs"])
    scheduler = utils.cosine_warmup_scheduler(
        optimizer, total_steps=total_steps,
        warmup_steps=steps_per_epoch * int(cfg["warmup_epochs"]),
    )
    scaler = utils.make_grad_scaler(device, enabled=bool(cfg["amp"]))
    amp_on = bool(cfg["amp"])
    trainer_step = memory.AdaptiveTrainStep(
        optimizer, scaler, max_microbatches=int(cfg["batch_size"]))

    def forward_fn(img: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        with utils.autocast_context(device, enabled=amp_on):
            return criterion(model(img), tgt)

    ckpt = utils.CheckpointManager(cfg["out_dir"], mode="max")
    writer = utils.make_summary_writer(cfg["out_dir"] / "tb")

    global_step = 0
    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        loss_meter = utils.AverageMeter()
        for images, targets in train_loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            # On CUDA OOM this shrinks the micro-batch and retries (grad
            # accumulation), so a memory spike degrades throughput, not crashes.
            batch_loss = trainer_step.run((images, targets), forward_fn)
            scheduler.step()
            loss_meter.update(batch_loss, images.size(0))
            writer.add_scalar("train/loss", batch_loss, global_step)
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)
            global_step += 1

        val_dice = evaluate(model, val_loader, device, num_classes)
        writer.add_scalar("val/dice", val_dice, epoch)
        is_best = ckpt.save(
            {
                "model_state": model.state_dict(), "arch": cfg["arch"],
                "encoder": cfg["encoder"], "num_classes": num_classes,
                "input_size": size, "epoch": epoch, "config": _jsonable(cfg),
            },
            metric=val_dice,
        )
        flag = " *best" if is_best else ""
        print(f"epoch {epoch:03d}/{cfg['epochs']}  train_loss={loss_meter.average:.4f}  "
              f"val_dice={val_dice:.4f}{flag}")

    writer.flush()
    writer.close()
    print(f"best val dice: {ckpt.best_metric:.4f} ({ckpt.best_path})")

    if bool(cfg["export_onnx"]):
        state = torch.load(ckpt.best_path, map_location="cpu")
        model.load_state_dict(state["model_state"])
        onnx_path = export_onnx(
            model, cfg["out_dir"] / "model.onnx", input_size=size,
            output_names=("logits",), device=torch.device("cpu"),
        )
        print(f"exported ONNX: {onnx_path}  sha256={utils.sha256_file(onnx_path)}")
    return 0


def _jsonable(cfg: dict[str, Any]) -> dict[str, Any]:
    return {k: (str(v) if isinstance(v, Path) else v) for k, v in cfg.items()}


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Train a pressure-injury stage classifier (ConvNeXt-V2 / ViT via timm) → ONNX.

Real, runnable trainer with AMP, cosine-with-warmup, patient-level grouped K-fold
CV, class-balanced losses, checkpointing, TensorBoard, and ONNX export. Defaults
target one ~40 GB H200 MIG slice at 384 px.

Data & the manual step
----------------------
Grading needs a **prepared manifest** you build locally (see ``training/data``):
``image,stage,patient_id[,skin_tone,site]``. There is no open, permissively
licensed, NPIAP-staged photo set that can be auto-downloaded — PIID
(``github.com/FU-MedicalAI/PIID``, Stage 1–4, no masks, no stated license) and
the DFUC challenge sets (gated behind a data-use agreement) must be obtained by
hand and staged into the manifest. This trainer never fabricates or fetches
staged data; it consumes what you prepared.

Loss modes
----------
* ``ce`` / ``focal`` — full 6-way head over :data:`training.data.GRADE_CLASSES`.
* ``corn`` — ordinal CORN over Stages 1–4 only (manifest must contain no
  Unstageable/DTI rows); the head emits ``K-1`` thresholds.

Example
-------
    python -m training.train_grade --config training/configs/grade_convnextv2.yaml \\
        --manifest data/piid/manifest.csv --out-dir artifacts/grade_convnextv2
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader

from training import data as data_mod
from training import memory, utils
from training.losses import (
    CornOrdinalLoss,
    FocalCrossEntropy,
    corn_logits_to_probs,
)
from training.models import build_grader, export_onnx

_ORDINAL_STAGES = 4  # Stages 1–4 form the CORN axis.


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--manifest", type=Path, default=None,
                        help="CSV: image,stage,patient_id[,skin_tone,site]")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--model-name", type=str, default=None, help="timm model id")
    parser.add_argument("--pretrained", dest="pretrained", action="store_true", default=None)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--loss", type=str, default=None, choices=["ce", "focal", "corn"])
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--warmup-epochs", type=int, default=None)
    parser.add_argument("--drop-path", type=float, default=None)
    parser.add_argument("--folds", type=int, default=None)
    parser.add_argument("--fold", type=int, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
    parser.add_argument("--vram-fraction", type=float, default=None,
                        help="cap the process to this fraction (0-1] of MIG VRAM")
    parser.add_argument("--amp", dest="amp", action="store_true", default=None)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    parser.add_argument("--grad-checkpointing", dest="grad_checkpointing",
                        action="store_true", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--no-onnx", dest="export_onnx", action="store_false", default=None)
    return parser.parse_args(argv)


_DEFAULTS: dict[str, Any] = {
    "model_name": "convnextv2_base.fcmae_ft_in22k_in1k_384",
    "pretrained": True,
    "loss": "focal",
    "input_size": 384,
    "epochs": 50,
    "batch_size": 24,
    "lr": 2e-4,
    "weight_decay": 0.05,
    "warmup_epochs": 3,
    "drop_path": 0.1,
    "folds": 5,
    "fold": 0,
    "num_workers": 4,
    "vram_fraction": None,
    "amp": True,
    "grad_checkpointing": False,
    "seed": 1234,
    "device": "auto",
    "export_onnx": True,
    "out_dir": "artifacts/grader",
}


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    file_config = utils.load_yaml_config(args.config) if args.config else {}
    cli = {k: v for k, v in vars(args).items() if k != "config"}
    merged = utils.merge_cli_over_config({**_DEFAULTS, **file_config}, cli)
    if merged.get("manifest") is None:
        raise SystemExit("--manifest is required (via CLI or config)")
    merged["manifest"] = Path(merged["manifest"])
    merged["out_dir"] = Path(merged["out_dir"])
    return merged


def quadratic_weighted_kappa(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    """Quadratic-weighted Cohen's kappa — the primary ordinal grading metric."""
    conf = np.zeros((num_classes, num_classes), dtype=np.float64)
    for t, p in zip(y_true, y_pred, strict=True):
        conf[int(t), int(p)] += 1.0
    if conf.sum() == 0:
        return 0.0
    weights = np.zeros((num_classes, num_classes), dtype=np.float64)
    for i in range(num_classes):
        for j in range(num_classes):
            weights[i, j] = ((i - j) ** 2) / ((num_classes - 1) ** 2)
    actual_hist = conf.sum(axis=1)
    pred_hist = conf.sum(axis=0)
    expected = np.outer(actual_hist, pred_hist) / conf.sum()
    numerator = float((weights * conf).sum())
    denominator = float((weights * expected).sum())
    if denominator == 0:
        return 1.0
    return 1.0 - numerator / denominator


def balanced_accuracy(y_true: np.ndarray, y_pred: np.ndarray, num_classes: int) -> float:
    """Macro-averaged recall across classes present in ``y_true``."""
    recalls: list[float] = []
    for cls in range(num_classes):
        mask = y_true == cls
        support = int(mask.sum())
        if support == 0:
            continue
        recalls.append(float((y_pred[mask] == cls).mean()))
    return float(np.mean(recalls)) if recalls else 0.0


def _head_and_eval_classes(loss_mode: str) -> tuple[int, int]:
    """Return ``(head_outputs, eval_num_classes)`` for the loss mode."""
    if loss_mode == "corn":
        return _ORDINAL_STAGES - 1, _ORDINAL_STAGES
    return len(data_mod.GRADE_CLASSES), len(data_mod.GRADE_CLASSES)


def _class_weights(labels: list[int], num_classes: int) -> torch.Tensor:
    counts = np.bincount(labels, minlength=num_classes).astype(np.float64)
    counts = np.clip(counts, 1.0, None)
    weights = counts.sum() / (num_classes * counts)
    return torch.tensor(weights, dtype=torch.float32)


@torch.no_grad()
def evaluate(model: torch.nn.Module, loader: DataLoader, device: torch.device,
             loss_mode: str, eval_classes: int) -> tuple[float, float]:
    model.eval()
    trues: list[int] = []
    preds: list[int] = []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        logits = memory.safe_infer(model, images)
        if loss_mode == "corn":
            probs = corn_logits_to_probs(logits)
            pred = probs.argmax(dim=1)
        else:
            pred = logits.argmax(dim=1)
        trues.extend(int(t) for t in targets.tolist())
        preds.extend(int(p) for p in pred.cpu().tolist())
    y_true = np.asarray(trues)
    y_pred = np.asarray(preds)
    qwk = quadratic_weighted_kappa(y_true, y_pred, eval_classes)
    bacc = balanced_accuracy(y_true, y_pred, eval_classes)
    return qwk, bacc


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = build_config(args)
    utils.set_seed(int(cfg["seed"]))
    device = utils.resolve_device(str(cfg["device"]))
    memory.configure(device, vram_fraction=cfg.get("vram_fraction"))
    num_workers = memory.safe_num_workers(int(cfg["num_workers"]))
    loss_mode = str(cfg["loss"])
    size = int(cfg["input_size"])
    head_outputs, eval_classes = _head_and_eval_classes(loss_mode)

    records = data_mod.read_grade_manifest(cfg["manifest"])
    if loss_mode == "corn":
        bad = [r for r in records if r.stage_index >= _ORDINAL_STAGES]
        if bad:
            raise SystemExit(
                "loss=corn requires only Stages 1–4 in the manifest; found "
                f"{len(bad)} Unstageable/DTI rows. Use loss=ce/focal for the full set."
            )

    groups = [r.patient_id for r in records]
    folds = data_mod.group_kfold_indices(groups, n_splits=int(cfg["folds"]),
                                          seed=int(cfg["seed"]))
    train_idx, val_idx = folds[int(cfg["fold"]) % len(folds)]
    train_records = [records[i] for i in train_idx]
    val_records = [records[i] for i in val_idx]
    print(f"records={len(records)} | train={len(train_records)} val={len(val_records)} "
          f"| fold={cfg['fold']}/{cfg['folds']} | loss={loss_mode}")

    train_loader = DataLoader(
        data_mod.GradingDataset(train_records, input_size=size, augment=True),
        batch_size=int(cfg["batch_size"]), shuffle=True,
        num_workers=num_workers, drop_last=True, pin_memory=device.type == "cuda",
    )
    val_loader = DataLoader(
        data_mod.GradingDataset(val_records, input_size=size, augment=False),
        batch_size=int(cfg["batch_size"]), shuffle=False,
        num_workers=num_workers, pin_memory=device.type == "cuda",
    )

    model = build_grader(
        model_name=str(cfg["model_name"]), num_classes=head_outputs,
        pretrained=bool(cfg["pretrained"]), drop_path_rate=float(cfg["drop_path"]),
        gradient_checkpointing=bool(cfg["grad_checkpointing"]),
    ).to(device)
    print(f"grader: {cfg['model_name']} | head={head_outputs} "
          f"| params={utils.count_parameters(model):,} | device={device}")

    train_labels = [r.stage_index for r in train_records]
    if loss_mode == "corn":
        criterion: torch.nn.Module = CornOrdinalLoss(_ORDINAL_STAGES)
    elif loss_mode == "focal":
        criterion = FocalCrossEntropy(class_weights=_class_weights(train_labels, head_outputs))
    else:
        criterion = torch.nn.CrossEntropyLoss(
            weight=_class_weights(train_labels, head_outputs)
        )
    criterion = criterion.to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]),
                                  weight_decay=float(cfg["weight_decay"]))
    steps_per_epoch = max(1, len(train_loader))
    scheduler = utils.cosine_warmup_scheduler(
        optimizer, total_steps=steps_per_epoch * int(cfg["epochs"]),
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
            # OOM-safe: shrinks the micro-batch and retries instead of crashing.
            batch_loss = trainer_step.run((images, targets), forward_fn)
            scheduler.step()
            loss_meter.update(batch_loss, images.size(0))
            writer.add_scalar("train/loss", batch_loss, global_step)
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)
            global_step += 1

        qwk, bacc = evaluate(model, val_loader, device, loss_mode, eval_classes)
        writer.add_scalar("val/qwk", qwk, epoch)
        writer.add_scalar("val/balanced_accuracy", bacc, epoch)
        is_best = ckpt.save(
            {
                "model_state": model.state_dict(), "model_name": cfg["model_name"],
                "loss": loss_mode, "head_outputs": head_outputs, "input_size": size,
                "classes": list(data_mod.GRADE_CLASSES), "epoch": epoch,
            },
            metric=qwk,
        )
        flag = " *best" if is_best else ""
        print(f"epoch {epoch:03d}/{cfg['epochs']}  loss={loss_meter.average:.4f}  "
              f"val_qwk={qwk:.4f}  bal_acc={bacc:.4f}{flag}")

    writer.flush()
    writer.close()
    print(f"best val QWK: {ckpt.best_metric:.4f} ({ckpt.best_path})")

    if bool(cfg["export_onnx"]):
        state = torch.load(ckpt.best_path, map_location="cpu")
        model.load_state_dict(state["model_state"])
        onnx_path = export_onnx(
            model, cfg["out_dir"] / "model.onnx", input_size=size,
            output_names=("logits",), device=torch.device("cpu"),
        )
        print(f"exported ONNX: {onnx_path}  sha256={utils.sha256_file(onnx_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

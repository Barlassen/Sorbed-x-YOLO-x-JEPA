#!/usr/bin/env python3
"""Directive-driven rule teacher + student distillation with consistency SSL.

This module implements the teacher–student staging pipeline from the methods
plan for the Acıbadem HD_T86 (NPIAP/EPUAP 2019) directive:

* :class:`DirectiveTeacher` — a **deterministic, auditable** pseudo-labeller. It
  is *not learned*. It consumes classical colour/tissue features and emits a
  **soft** stage distribution with ordinal neighbour mass, a confidence, an
  **abstention** flag, the gate that fired, and directive citations. Abstained
  images produce no pseudo-label (used only for unsupervised consistency).
* :func:`train_student` — trains a modern grade student (ConvNeXt-V2 via timm)
  by combining (a) real expert labels (dominant), (b) KL distillation from the
  rule teacher's soft targets on non-abstained images, and (c) consistency
  semi-supervision: an EMA mean-teacher plus optional FixMatch.

Honest boundary (encoded, not hidden)
------------------------------------
The thresholds below encode the NPIAP/EPUAP staging *schema*, not the literal
HD_T86 constants — the directive text is Turkish prose, not numeric cut-offs. A
wound-care physician must reconcile every threshold (``obscure_tau``,
``structure_tau``, ``margin_delta`` …) against the real HD_T86 sections and sign
off before any pseudo-label is generated. The classical colour bootstrap **cannot
see** exposed adipose/muscle/bone, so Stage 3/4 gates fire only when a learned
tissue head supplies ``structure_fraction``; otherwise the teacher abstains
rather than fabricate a depth-dependent grade — the safe, intended behaviour.

Example
-------
    python -m training.teacher_student \\
        --config training/configs/teacher_student.yaml \\
        --unlabeled-images data/pool/images --unlabeled-masks data/pool/masks \\
        --labeled-manifest data/piid/train.csv --val-manifest data/piid/val.csv \\
        --out-dir artifacts/student
"""

from __future__ import annotations

import argparse
import itertools
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset

from sorbed.domain.enums import PressureInjuryStage, SkinToneBand
from sorbed.domain.enums import TissueClass as _Tissue
from sorbed.guidelines.pack import load_default_pack
from sorbed.tissue.color_model import ColorTissueClassifier
from training import data as data_mod
from training import utils
from training.losses import (
    DistillationKLLoss,
    SoftmaxConsistencyLoss,
    fixmatch_pseudo_label,
    masked_cross_entropy,
)
from training.models import build_grader, export_onnx

# Stage → index into training.data.GRADE_CLASSES.
_STAGE_INDEX = {name: i for i, name in enumerate(data_mod.GRADE_CLASSES)}
_N_STAGES = len(data_mod.GRADE_CLASSES)

# Gate → directive topic key (for citations), matching var/directive_packs/hd_t86.
_GATE_TOPIC = {
    "unstageable": "stage.unstageable",
    "stage_4": "stage.stage_4",
    "stage_3": "stage.stage_3",
    "deep_tissue_injury": "stage.deep_tissue_injury",
    "stage_1": "stage.stage_1",
    "stage_2": "stage.stage_2",
}


@dataclass(frozen=True)
class TeacherThresholds:
    """NPIAP-aligned gate thresholds — the *schema*, pending clinical sign-off."""

    obscure_tau: float = 0.5        # slough+eschar fraction that hides the base
    structure_tau: float = 0.05     # exposed muscle/bone/tendon fraction → Stage 4
    adipose_tau: float = 0.05       # exposed fat fraction → Stage 3
    erythema_min: float = 0.15      # non-blanchable-erythema cue → Stage 1
    maroon_min: float = 0.15        # maroon/purple cue → DTI
    devitalized_max_stage2: float = 0.10  # slough tolerated before leaving Stage 2
    margin_delta: float = 0.10      # min feature margin; below it → abstain
    min_confidence: float = 0.55    # soft-target confidence floor to not abstain
    logistic_gain: float = 12.0     # steepness of the margin→confidence logistic


@dataclass
class TeacherConfig:
    """Behavioural switches for :class:`DirectiveTeacher`."""

    thresholds: TeacherThresholds = field(default_factory=TeacherThresholds)
    abstain_dark_skin_low_confidence: bool = True
    require_scale_reference: bool = False  # abstain when no fiducial (methods §2.3a)


@dataclass(frozen=True)
class TeacherFeatures:
    """The intermediate features the teacher consumes for one image."""

    granulation: float
    slough: float
    eschar: float
    epithelial: float
    obscured_fraction: float
    open_bed_fraction: float
    maroon_purple: float
    erythema: float
    structure_fraction: float
    adipose_fraction: float
    skin_intact: bool
    depth_index: float
    skin_tone: SkinToneBand = SkinToneBand.UNKNOWN
    has_scale_reference: bool = False


@dataclass(frozen=True)
class TeacherOutput:
    """Soft pseudo-label with provenance and an explicit abstention."""

    soft_probs: np.ndarray  # shape (_N_STAGES,), sums to 1
    stage_index: int
    confidence: float
    abstain: bool
    gate: str
    reason: str
    citations: list[tuple[str, int]]


def _sigmoid(x: float) -> float:
    return float(1.0 / (1.0 + np.exp(-x)))


class DirectiveTeacher:
    """Deterministic rule pseudo-labeller with abstention and soft targets."""

    def __init__(self, config: TeacherConfig | None = None, *, attach_citations: bool = True):
        self.config = config or TeacherConfig()
        self._classifier = ColorTissueClassifier()
        self._pack = load_default_pack() if attach_citations else None

    # -- feature extraction ------------------------------------------------
    def features_from_image(
        self,
        rgb: np.ndarray,
        wound_mask: np.ndarray,
        *,
        structure_fraction: float = 0.0,
        adipose_fraction: float = 0.0,
        skin_tone: SkinToneBand = SkinToneBand.UNKNOWN,
        has_scale_reference: bool = False,
    ) -> TeacherFeatures:
        """Classical colour/tissue features for one RGB image + wound mask.

        ``structure_fraction`` / ``adipose_fraction`` come from a learned tissue
        head when available; the colour bootstrap cannot detect them and leaves
        them at 0 (so Stage 3/4 gates stay closed and the teacher abstains).
        """
        from sorbed.morphometrics.depth_proxy import shading_depth_proxy

        analysis = self._classifier.classify(rgb, wound_mask.astype(bool))
        comp = analysis.composition
        depth = shading_depth_proxy(rgb, wound_mask.astype(bool))
        depth_index = depth.relative_depth_index if depth is not None else 0.0
        return TeacherFeatures(
            granulation=comp.fraction_of(_Tissue.GRANULATION),
            slough=comp.fraction_of(_Tissue.SLOUGH),
            eschar=comp.fraction_of(_Tissue.ESCHAR),
            epithelial=comp.fraction_of(_Tissue.EPITHELIAL),
            obscured_fraction=comp.obscured_fraction,
            open_bed_fraction=analysis.open_bed_fraction,
            maroon_purple=analysis.maroon_purple_fraction,
            erythema=analysis.erythema_fraction,
            structure_fraction=float(structure_fraction),
            adipose_fraction=float(adipose_fraction),
            skin_intact=analysis.skin_appears_intact,
            depth_index=depth_index,
            skin_tone=skin_tone,
            has_scale_reference=has_scale_reference,
        )

    # -- labelling ---------------------------------------------------------
    def label(self, features: TeacherFeatures) -> TeacherOutput:
        """Apply the ordered directive gates → soft target + abstention."""
        t = self.config.thresholds
        if self.config.require_scale_reference and not features.has_scale_reference:
            return self._abstain("no scale reference (fiducial) — morphometrics void")

        # 1. Obscuration gate → Unstageable (a genuinely photo-observable rule).
        if features.obscured_fraction >= t.obscure_tau:
            conf = self._confidence(features.obscured_fraction, t.obscure_tau)
            return self._fire("unstageable", conf, "wound base obscured by slough/eschar")

        # 2. Exposed deep structure → Stage 4 (needs a learned tissue head).
        if features.structure_fraction >= t.structure_tau:
            conf = self._confidence(features.structure_fraction, t.structure_tau)
            return self._fire("stage_4", conf,
                              "muscle/tendon/bone visible (depth/undermining unverified)")

        # 3. Exposed adipose → Stage 3 (depth inferred, not measured → capped).
        if features.adipose_fraction >= t.adipose_tau:
            conf = min(0.75, self._confidence(features.adipose_fraction, t.adipose_tau))
            return self._fire("stage_3", conf, "subcutaneous fat visible (depth inferred)")

        # 4. Intact skin + discolouration → DTI vs Stage 1 (low-confidence).
        if features.skin_intact:
            if features.maroon_purple >= t.maroon_min:
                return self._intact_discoloration(
                    "deep_tissue_injury", features, features.maroon_purple, t.maroon_min)
            if features.erythema >= t.erythema_min:
                return self._intact_discoloration(
                    "stage_1", features, features.erythema, t.erythema_min)
            return self._abstain("intact skin without a clear erythema/DTI colour cue")

        # 5. Partial-thickness, viable base, minimal slough → Stage 2.
        if features.open_bed_fraction > 0.0 and features.slough <= t.devitalized_max_stage2:
            margin = min(features.granulation, features.epithelial + features.granulation)
            conf = self._confidence(max(margin, 0.2), 0.0)
            return self._fire("stage_2", min(0.85, conf),
                              "partial-thickness viable base, no obscuring slough")

        # 6. Nothing cleared a node margin.
        return self._abstain("no gate margin cleared (ambiguous tissue composition)")

    # -- helpers -----------------------------------------------------------
    def _confidence(self, value: float, tau: float) -> float:
        gain = self.config.thresholds.logistic_gain
        return _sigmoid(gain * (value - tau - self.config.thresholds.margin_delta) + 1.5)

    def _intact_discoloration(
        self, gate: str, features: TeacherFeatures, value: float, tau: float,
    ) -> TeacherOutput:
        conf = min(0.7, self._confidence(value, tau))
        if (self.config.abstain_dark_skin_low_confidence
                and features.skin_tone == SkinToneBand.IV_VI):
            return self._abstain(
                f"{gate} in Fitzpatrick IV–VI needs colour-card ITA° support (abstain)")
        return self._fire(gate, conf, f"intact skin with {gate} colour cue")

    def _fire(self, gate: str, confidence: float, reason: str) -> TeacherOutput:
        confidence = float(np.clip(confidence, 0.0, 0.98))
        if confidence < self.config.thresholds.min_confidence:
            return self._abstain(f"{gate} below confidence floor ({confidence:.2f})")
        stage_value = self._gate_to_stage(gate)
        idx = _STAGE_INDEX[stage_value]
        probs = self._soft_distribution(idx, confidence)
        return TeacherOutput(
            soft_probs=probs, stage_index=idx, confidence=confidence,
            abstain=False, gate=gate, reason=reason, citations=self._citations(gate),
        )

    def _abstain(self, reason: str) -> TeacherOutput:
        probs = np.full(_N_STAGES, 1.0 / _N_STAGES, dtype=np.float32)
        return TeacherOutput(
            soft_probs=probs, stage_index=-1, confidence=0.0,
            abstain=True, gate="abstain", reason=reason, citations=[],
        )

    @staticmethod
    def _gate_to_stage(gate: str) -> str:
        mapping = {
            "unstageable": PressureInjuryStage.UNSTAGEABLE.value,
            "stage_4": PressureInjuryStage.STAGE_4.value,
            "stage_3": PressureInjuryStage.STAGE_3.value,
            "deep_tissue_injury": PressureInjuryStage.DEEP_TISSUE.value,
            "stage_1": PressureInjuryStage.STAGE_1.value,
            "stage_2": PressureInjuryStage.STAGE_2.value,
        }
        return mapping[gate]

    @staticmethod
    def _soft_distribution(target: int, confidence: float) -> np.ndarray:
        """Peak on ``target`` with ordinal neighbour mass on adjacent Stages."""
        probs = np.full(_N_STAGES, (1.0 - confidence) * 0.05, dtype=np.float32)
        probs[target] = confidence
        if target < 4:  # Stages 1–4 occupy indices 0–3 and are ordered.
            share = (1.0 - confidence) * 0.5
            for neighbour in (target - 1, target + 1):
                if 0 <= neighbour < 4:
                    probs[neighbour] += share
        total = float(probs.sum())
        return (probs / total).astype(np.float32)

    def _citations(self, gate: str) -> list[tuple[str, int]]:
        if self._pack is None:
            return []
        topic = _GATE_TOPIC.get(gate)
        return self._pack.citation(topic) if topic else []


# ---------------------------------------------------------------------------
# Pseudo-label pool (teacher targets are precomputed once — student-independent).
# ---------------------------------------------------------------------------
class PseudoLabelPool(Dataset):
    """Unlabeled images with precomputed teacher soft targets and weak/strong views."""

    def __init__(
        self,
        images: list[Path],
        *,
        input_size: int,
        teacher: DirectiveTeacher,
        masks: dict[str, Path] | None = None,
    ) -> None:
        self._images = images
        self._size = int(input_size)
        self.soft_targets = np.zeros((len(images), _N_STAGES), dtype=np.float32)
        self.weights = np.zeros(len(images), dtype=np.float32)
        self.abstained = np.zeros(len(images), dtype=bool)
        self._precompute(teacher, masks or {})

    def _precompute(self, teacher: DirectiveTeacher, masks: dict[str, Path]) -> None:
        for i, path in enumerate(self._images):
            rgb = data_mod.read_rgb(path, self._size)
            mask_path = masks.get(path.stem)
            if mask_path is not None:
                mask = data_mod.read_mask(mask_path, self._size) > 0
            else:
                mask = np.ones((self._size, self._size), dtype=bool)
            features = teacher.features_from_image(rgb, mask)
            out = teacher.label(features)
            self.soft_targets[i] = out.soft_probs
            self.abstained[i] = out.abstain
            self.weights[i] = 0.0 if out.abstain else out.confidence

    def coverage(self) -> float:
        """Fraction of the pool that received a (non-abstained) pseudo-label."""
        return float((~self.abstained).mean()) if len(self.abstained) else 0.0

    def __len__(self) -> int:
        return len(self._images)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor, int]:
        rgb = data_mod.read_rgb(self._images[index], self._size)
        weak = data_mod.to_input_tensor(data_mod.weak_augment(rgb))
        strong = data_mod.to_input_tensor(data_mod.strong_augment(rgb))
        return weak, strong, index


def _stem_index(masks_dir: Path | None) -> dict[str, Path]:
    if masks_dir is None:
        return {}
    return {p.stem: p for p in data_mod.list_images(masks_dir)}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--config", type=Path, default=None)
    parser.add_argument("--unlabeled-images", type=Path, default=None)
    parser.add_argument("--unlabeled-masks", type=Path, default=None,
                        help="optional wound masks for the pool (pre-masked by SAM/segmenter)")
    parser.add_argument("--labeled-manifest", type=Path, default=None,
                        help="optional expert-labeled manifest (dominant supervision)")
    parser.add_argument("--val-manifest", type=Path, default=None,
                        help="optional labeled val manifest for QWK monitoring")
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--pretrained", dest="pretrained", action="store_true", default=None)
    parser.add_argument("--no-pretrained", dest="pretrained", action="store_false")
    parser.add_argument("--input-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--labeled-batch-size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight-decay", type=float, default=None)
    parser.add_argument("--warmup-epochs", type=int, default=None)
    parser.add_argument("--lambda-sup", type=float, default=None)
    parser.add_argument("--lambda-kd", type=float, default=None)
    parser.add_argument("--lambda-cons", type=float, default=None)
    parser.add_argument("--lambda-fixmatch", type=float, default=None)
    parser.add_argument("--temperature", type=float, default=None)
    parser.add_argument("--fixmatch-threshold", type=float, default=None)
    parser.add_argument("--ema-decay", type=float, default=None)
    parser.add_argument("--num-workers", type=int, default=None)
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
    "input_size": 384,
    "epochs": 40,
    "batch_size": 16,
    "labeled_batch_size": 8,
    "lr": 1e-4,
    "weight_decay": 0.05,
    "warmup_epochs": 3,
    "lambda_sup": 1.0,
    "lambda_kd": 0.5,
    "lambda_cons": 0.3,
    "lambda_fixmatch": 0.0,
    "temperature": 2.0,
    "fixmatch_threshold": 0.95,
    "ema_decay": 0.999,
    "num_workers": 4,
    "amp": True,
    "grad_checkpointing": False,
    "seed": 1234,
    "device": "auto",
    "export_onnx": True,
    "out_dir": "artifacts/student",
}


def build_config(args: argparse.Namespace) -> dict[str, Any]:
    file_config = utils.load_yaml_config(args.config) if args.config else {}
    cli = {k: v for k, v in vars(args).items() if k != "config"}
    merged = utils.merge_cli_over_config({**_DEFAULTS, **file_config}, cli)
    if merged.get("unlabeled_images") is None:
        raise SystemExit("--unlabeled-images is required (via CLI or config)")
    for key in ("unlabeled_images", "unlabeled_masks", "labeled_manifest",
                "val_manifest", "out_dir"):
        if merged.get(key) is not None:
            merged[key] = Path(merged[key])
    return merged


@torch.no_grad()
def _evaluate_qwk(model: torch.nn.Module, loader: DataLoader,
                  device: torch.device) -> float:
    from training.train_grade import quadratic_weighted_kappa

    model.eval()
    trues: list[int] = []
    preds: list[int] = []
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        pred = model(images).argmax(dim=1)
        trues.extend(int(t) for t in targets.tolist())
        preds.extend(int(p) for p in pred.cpu().tolist())
    return quadratic_weighted_kappa(np.asarray(trues), np.asarray(preds), _N_STAGES)


def train_student(cfg: dict[str, Any]) -> int:
    """Distill the rule teacher into a grade student with consistency SSL."""
    utils.set_seed(int(cfg["seed"]))
    device = utils.resolve_device(str(cfg["device"]))
    size = int(cfg["input_size"])

    teacher = DirectiveTeacher()
    pool_images = data_mod.list_images(cfg["unlabeled_images"])
    if not pool_images:
        raise SystemExit(f"no images under {cfg['unlabeled_images']}")
    pool = PseudoLabelPool(
        pool_images, input_size=size, teacher=teacher,
        masks=_stem_index(cfg.get("unlabeled_masks")),
    )
    print(f"pool={len(pool)} images | teacher label coverage={pool.coverage():.1%} "
          f"| citations={'on' if teacher._pack else 'off (no pack)'}")

    unlabeled_loader = DataLoader(
        pool, batch_size=int(cfg["batch_size"]), shuffle=True,
        num_workers=int(cfg["num_workers"]), drop_last=True, pin_memory=device.type == "cuda",
    )

    labeled_loader: DataLoader | None = None
    if cfg.get("labeled_manifest"):
        records = data_mod.read_grade_manifest(cfg["labeled_manifest"])
        labeled_loader = DataLoader(
            data_mod.GradingDataset(records, input_size=size, augment=True),
            batch_size=int(cfg["labeled_batch_size"]), shuffle=True,
            num_workers=int(cfg["num_workers"]), drop_last=True,
            pin_memory=device.type == "cuda",
        )
    val_loader: DataLoader | None = None
    if cfg.get("val_manifest"):
        val_records = data_mod.read_grade_manifest(cfg["val_manifest"])
        val_loader = DataLoader(
            data_mod.GradingDataset(val_records, input_size=size, augment=False),
            batch_size=int(cfg["labeled_batch_size"]), shuffle=False,
            num_workers=int(cfg["num_workers"]), pin_memory=device.type == "cuda",
        )

    model = build_grader(
        model_name=str(cfg["model_name"]), num_classes=_N_STAGES,
        pretrained=bool(cfg["pretrained"]),
        gradient_checkpointing=bool(cfg["grad_checkpointing"]),
    ).to(device)
    ema = utils.ModelEma(model, decay=float(cfg["ema_decay"]))
    print(f"student: {cfg['model_name']} | params={utils.count_parameters(model):,} "
          f"| device={device}")

    kd_loss = DistillationKLLoss(temperature=float(cfg["temperature"])).to(device)
    cons_loss = SoftmaxConsistencyLoss().to(device)
    ce_loss = torch.nn.CrossEntropyLoss().to(device)

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(cfg["lr"]),
                                  weight_decay=float(cfg["weight_decay"]))
    steps_per_epoch = max(1, len(unlabeled_loader))
    scheduler = utils.cosine_warmup_scheduler(
        optimizer, total_steps=steps_per_epoch * int(cfg["epochs"]),
        warmup_steps=steps_per_epoch * int(cfg["warmup_epochs"]),
    )
    scaler = utils.make_grad_scaler(device, enabled=bool(cfg["amp"]))
    ckpt = utils.CheckpointManager(cfg["out_dir"], mode="max")
    writer = utils.make_summary_writer(cfg["out_dir"] / "tb")

    soft = torch.from_numpy(pool.soft_targets).to(device)
    weights = torch.from_numpy(pool.weights).to(device)
    labeled_iter = itertools.cycle(labeled_loader) if labeled_loader else None
    global_step = 0

    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        meter = utils.AverageMeter()
        for weak, strong, idx in unlabeled_loader:
            weak = weak.to(device, non_blocking=True)
            strong = strong.to(device, non_blocking=True)
            idx = idx.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with utils.autocast_context(device, enabled=bool(cfg["amp"])):
                student_strong = model(strong)
                with torch.no_grad():
                    ema_weak = ema.module(weak)
                teacher_soft = soft[idx]
                teacher_w = weights[idx]
                loss = torch.zeros((), device=device)

                if float(cfg["lambda_kd"]) > 0 and bool((teacher_w > 0).any()):
                    loss = loss + float(cfg["lambda_kd"]) * kd_loss(
                        student_strong, teacher_soft, sample_weight=teacher_w)
                if float(cfg["lambda_cons"]) > 0:
                    loss = loss + float(cfg["lambda_cons"]) * cons_loss(student_strong, ema_weak)
                if float(cfg["lambda_fixmatch"]) > 0:
                    pseudo, mask = fixmatch_pseudo_label(
                        ema_weak, threshold=float(cfg["fixmatch_threshold"]))
                    loss = loss + float(cfg["lambda_fixmatch"]) * masked_cross_entropy(
                        student_strong, pseudo, mask)
                if labeled_iter is not None:
                    lab_images, lab_targets = next(labeled_iter)
                    lab_images = lab_images.to(device, non_blocking=True)
                    lab_targets = lab_targets.to(device, non_blocking=True)
                    sup = ce_loss(model(lab_images), lab_targets)
                    loss = loss + float(cfg["lambda_sup"]) * sup

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            ema.update(model)
            meter.update(float(loss.item()), weak.size(0))
            writer.add_scalar("train/loss", float(loss.item()), global_step)
            writer.add_scalar("train/lr", scheduler.get_last_lr()[0], global_step)
            global_step += 1

        if val_loader is not None:
            metric = _evaluate_qwk(ema.module, val_loader, device)
            writer.add_scalar("val/qwk_ema", metric, epoch)
        else:
            metric = -meter.average  # no labels: track (negative) loss as monitor
        is_best = ckpt.save(
            {
                "model_state": model.state_dict(),
                "ema_state": ema.module.state_dict(),
                "model_name": cfg["model_name"], "classes": list(data_mod.GRADE_CLASSES),
                "input_size": size, "epoch": epoch,
            },
            metric=metric,
        )
        flag = " *best" if is_best else ""
        monitor = f"val_qwk={metric:.4f}" if val_loader is not None \
            else f"loss={meter.average:.4f}"
        print(f"epoch {epoch:03d}/{cfg['epochs']}  {monitor}{flag}")

    writer.flush()
    writer.close()
    print(f"best monitor: {ckpt.best_metric:.4f} ({ckpt.best_path})")

    if bool(cfg["export_onnx"]):
        state = torch.load(ckpt.best_path, map_location="cpu")
        model.load_state_dict(state["ema_state"])  # deploy the EMA student
        onnx_path = export_onnx(
            model, cfg["out_dir"] / "model.onnx", input_size=size,
            output_names=("logits",), device=torch.device("cpu"),
        )
        print(f"exported ONNX (EMA student): {onnx_path}  sha256={utils.sha256_file(onnx_path)}")
    return 0


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = build_config(args)
    return train_student(cfg)


if __name__ == "__main__":
    raise SystemExit(main())

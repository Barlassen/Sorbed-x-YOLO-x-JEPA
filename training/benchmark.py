#!/usr/bin/env python3
"""Benchmark the trained grader against the rule baseline and render examples.

Produces, on a held-out validation fold of a stage-graded manifest:

* ``benchmark.json`` — accuracy and quadratic-weighted kappa for the learned
  ConvNeXt grader versus the directive **rule engine** (the classical, no-training
  staging path in :mod:`sorbed`), with the rule engine's abstention rate. This is
  the "learned vs. base" comparison for the paper.
* ``examples.png`` — a montage of real images, each shown as *photo | wound-mask
  overlay* with its true stage, the learned grader's stage + confidence, and the
  rule engine's stage. Lets you eyeball what the pipeline actually produced.

Heavy dependencies (torch, onnxruntime, PIL, the ``sorbed`` runtime) are imported
lazily so this module imports without them.

Example
-------
    python -m training.benchmark \\
        --grade-manifest /data/briefer/datasets/piid/manifest.csv \\
        --grade-onnx artifacts/grade_convnextv2/model.onnx \\
        --seg-onnx artifacts/seg_unetpp/model.onnx \\
        --temperature 3.05 --n-examples 8 --out-dir artifacts/benchmark
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

STAGES = ("stage_1", "stage_2", "stage_3", "stage_4")
_STAGE_INDEX = {s: i for i, s in enumerate(STAGES)}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--grade-manifest", type=Path, required=True,
                        help="CSV image,stage,patient_id (the grading manifest).")
    parser.add_argument("--grade-onnx", type=Path, required=True,
                        help="Trained grader ONNX (CORN head).")
    parser.add_argument("--seg-onnx", type=Path, default=None,
                        help="Trained segmenter ONNX; used for the mask overlays.")
    parser.add_argument("--out-dir", type=Path, default=Path("artifacts/benchmark"))
    parser.add_argument("--input-size", type=int, default=288)
    parser.add_argument("--folds", type=int, default=5)
    parser.add_argument("--fold", type=int, default=0)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--temperature", type=float, default=1.0,
                        help="Temperature applied to the grader logits (calibration).")
    parser.add_argument("--n-examples", type=int, default=8,
                        help="Images to render in the montage (spread across stages).")
    parser.add_argument("--max-images", type=int, default=60,
                        help="Cap on validation images scored (the rule engine is slow); "
                        "0 scores the whole fold. Subsample is deterministic.")
    parser.add_argument("--device", type=str, default="cpu")
    return parser.parse_args(argv)


def _val_records(manifest: Path, folds: int, fold: int, seed: int) -> list[Any]:
    from training import data as data_mod

    records = data_mod.read_grade_manifest(manifest)
    groups = [r.patient_id for r in records]
    _, val_idx = data_mod.group_kfold_indices(groups, n_splits=folds, seed=seed)[fold]
    return [records[i] for i in val_idx]


def _grade_probs(onnx_path: Path, image_path: str, size: int, temperature: float) -> Any:
    import numpy as np
    import onnxruntime as ort
    import torch

    from training.data import read_rgb, to_input_tensor
    from training.losses import corn_logits_to_probs

    sess = getattr(_grade_probs, "_sess", None)
    if sess is None or _grade_probs._path != str(onnx_path):  # type: ignore[attr-defined]
        sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
        _grade_probs._sess = sess  # type: ignore[attr-defined]
        _grade_probs._path = str(onnx_path)  # type: ignore[attr-defined]
    x = to_input_tensor(read_rgb(image_path, size))[None].numpy().astype(np.float32)
    logits = torch.from_numpy(sess.run(None, {sess.get_inputs()[0].name: x})[0]) / temperature
    return corn_logits_to_probs(logits)[0].numpy()


def _rule_stage(image_path: str) -> tuple[str, float, Any]:
    """Return ``(rule_stage_value, confidence, bundle)`` from the sorbed pipeline."""
    from sorbed.pipeline import AnalyzeOptions, analyze_image

    bundle = analyze_image(image_path, options=AnalyzeOptions())
    decision = bundle.analysis.decision
    return str(getattr(decision.stage, "value", decision.stage)), float(decision.confidence), bundle


def _kappa(y_true: list[int], y_pred: list[int]) -> float:
    import numpy as np

    from training.train_grade import quadratic_weighted_kappa

    if not y_true:
        return 0.0
    return quadratic_weighted_kappa(np.asarray(y_true), np.asarray(y_pred), len(STAGES))


def _montage(examples: list[dict[str, Any]], out_path: Path) -> None:
    from PIL import Image, ImageDraw

    from sorbed.visualize.overlay import render_tissue_overlay

    thumb, pad, bar = 224, 8, 46
    rows = []
    for ex in examples:
        photo = Image.open(ex["image"]).convert("RGB").resize((thumb, thumb))
        b = ex["bundle"]
        overlay = render_tissue_overlay(
            b.display_image.to_uint8_rgb(), b.tissue_label_map, b.wound_mask
        ).resize((thumb, thumb))
        row = Image.new("RGB", (thumb * 2 + pad * 3, thumb + bar + pad), (18, 18, 22))
        row.paste(photo, (pad, bar))
        row.paste(overlay, (thumb + pad * 2, bar))
        d = ImageDraw.Draw(row)
        ok = "OK" if ex["true"] == ex["learned"] else "x"
        d.text((pad, 6),
               f"true {ex['true']}   |   grader {ex['learned']} ({ex['conf']:.2f}) [{ok}]"
               f"   |   rule {ex['rule']}",
               fill=(235, 235, 240))
        d.text((pad, thumb + bar - 14), "photo", fill=(150, 150, 160))
        d.text((thumb + pad * 2, thumb + bar - 14), "wound-mask overlay", fill=(150, 150, 160))
        rows.append(row)
    w = max(r.width for r in rows)
    canvas = Image.new("RGB", (w, sum(r.height for r in rows) + pad), (18, 18, 22))
    y = 0
    for r in rows:
        canvas.paste(r, (0, y))
        y += r.height
    out_path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(out_path)


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.seg_onnx is not None:
        os.environ["SORBED_SEGMENTATION_BACKEND"] = "onnx"
        os.environ["SORBED_ONNX_MODEL"] = str(Path(args.seg_onnx).resolve())

    val = _val_records(args.grade_manifest, args.folds, args.fold, args.seed)
    if args.max_images and len(val) > args.max_images:
        import random

        order = random.Random(args.seed).sample(range(len(val)), args.max_images)
        val = [val[i] for i in sorted(order)]
        print(f"benchmark: {len(val)} of the fold's images (deterministic subsample)")
    else:
        print(f"benchmark: {len(val)} validation images (fold {args.fold}/{args.folds})")

    learned_true: list[int] = []
    learned_pred: list[int] = []
    rule_true: list[int] = []
    rule_pred: list[int] = []
    rule_abstain = 0
    per_stage_examples: dict[str, list[dict[str, Any]]] = {s: [] for s in STAGES}

    for i, rec in enumerate(val):
        true_stage = STAGES[rec.stage_index]
        probs = _grade_probs(args.grade_onnx, str(rec.image), args.input_size, args.temperature)
        learned = STAGES[int(probs.argmax())]
        conf = float(probs.max())
        rule_value, _rule_conf, bundle = _rule_stage(str(rec.image))

        learned_true.append(rec.stage_index)
        learned_pred.append(_STAGE_INDEX[learned])
        if rule_value in _STAGE_INDEX:
            rule_true.append(rec.stage_index)
            rule_pred.append(_STAGE_INDEX[rule_value])
        else:
            rule_abstain += 1

        if len(per_stage_examples[true_stage]) < 2:
            per_stage_examples[true_stage].append({
                "image": str(rec.image), "true": true_stage, "learned": learned,
                "conf": conf, "rule": rule_value, "bundle": bundle,
            })
        if (i + 1) % 25 == 0:
            print(f"  ...{i + 1}/{len(val)}")

    n = len(learned_true)
    learned_acc = sum(t == p for t, p in zip(learned_true, learned_pred, strict=True)) / n
    rule_committed = len(rule_true)
    rule_acc = (
        sum(t == p for t, p in zip(rule_true, rule_pred, strict=True)) / rule_committed
        if rule_committed else 0.0
    )
    metrics = {
        "n_val": n,
        "learned_grader": {
            "accuracy": round(learned_acc, 4),
            "quadratic_weighted_kappa": round(_kappa(learned_true, learned_pred), 4),
            "abstention_rate": 0.0,
        },
        "rule_engine_baseline": {
            "accuracy_on_committed": round(rule_acc, 4),
            "quadratic_weighted_kappa_on_committed": round(_kappa(rule_true, rule_pred), 4),
            "abstention_rate": round(rule_abstain / n, 4),
            "committed": rule_committed,
        },
        "temperature": args.temperature,
        "fold": args.fold,
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "benchmark.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    examples: list[dict[str, Any]] = []
    for stage in STAGES:
        examples.extend(per_stage_examples[stage])
    examples = examples[: args.n_examples]
    if examples:
        _montage(examples, args.out_dir / "examples.png")

    print(json.dumps(metrics, indent=2))
    print(f"wrote {args.out_dir / 'benchmark.json'} and {args.out_dir / 'examples.png'}")
    return metrics


def main(argv: list[str] | None = None) -> int:
    run(parse_args(argv))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

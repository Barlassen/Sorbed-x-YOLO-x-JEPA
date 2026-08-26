#!/usr/bin/env python3
"""Assemble a HuggingFace-ready model repository for Sorbed's ONNX models.

Collects the released ONNX weights, a ``config.json`` describing each model's
input/output and preprocessing, and a model card into an output directory, and
optionally pushes them to a HuggingFace repository.

Assembling the directory needs no network and no token. Pushing requires the
``huggingface_hub`` package and a write token in ``HF_TOKEN`` (or a prior
``huggingface-cli login``); uploading to a HuggingFace repository is the one step
that requires an account and token.

Example
-------
    # assemble locally (no token needed)
    python scripts/export_hf.py --models-dir models/release --out hf_export

    # then push (requires HF_TOKEN and huggingface_hub)
    python scripts/export_hf.py --models-dir models/release --out hf_export \\
        --repo-id your-user/sorbed-pressure-injury --push
"""

from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

# Static description of each released model: filename, input, preprocessing.
MODEL_CONFIG = {
    "models": {
        "segmenter": {
            "file": "seg_unetpp.onnx",
            "architecture": "U-Net++ / EfficientNetV2-S (segmentation-models-pytorch)",
            "task": "binary wound-area segmentation",
            "input": {"name": "image", "shape": [1, 3, 768, 768],
                      "layout": "NCHW", "dtype": "float32"},
            "preprocessing": {
                "resize": [768, 768],
                "scale": "0-1",
                "normalize_mean": [0.485, 0.456, 0.406],
                "normalize_std": [0.229, 0.224, 0.225],
            },
            "output": {"shape": [1, 1, 768, 768],
                       "meaning": "foreground logits; sigmoid >= 0.5 = wound"},
            "training_data": "AZH Chronic Wound + MICCAI-2021 FUSeg (foot ulcers)",
            "metrics": {"internal_validation_dice": 0.87},
        },
        "grader": {
            "file": "grade_convnextv2.onnx",
            "architecture": "ConvNeXt-V2 with CORN ordinal head (timm)",
            "task": "pressure-injury stage classification (stages 1-4)",
            "input": {"name": "image", "shape": [1, 3, 288, 288],
                      "layout": "NCHW", "dtype": "float32"},
            "preprocessing": {
                "resize": [288, 288],
                "scale": "0-1",
                "normalize_mean": [0.485, 0.456, 0.406],
                "normalize_std": [0.229, 0.224, 0.225],
            },
            "output": {
                "shape": [1, 3],
                "meaning": "CORN conditional logits; divide by temperature 3.05, "
                "apply sigmoid+cumulative product to obtain stage probabilities for stages 1-4",
            },
            "classes": ["stage_1", "stage_2", "stage_3", "stage_4"],
            "temperature": 3.05,
            "training_data": "PIID (EPUAP stages 1-4; 1,087 images after de-duplication)",
            "metrics": {
                "cv_quadratic_weighted_kappa": "0.919 +/- 0.006 (5-fold patient-independent)",
                "balanced_accuracy": 0.80,
                "ece_after_temperature_scaling": 0.039,
            },
        },
    },
    "scope": (
        "Single-institution internal validation. PIID is EPUAP-staged with no "
        "external test set; the segmenter is trained on diabetic-foot ulcers, "
        "which differ from pressure injuries in site, tissue, and depth. Decision "
        "support, not a medical device."
    ),
    "license": "apache-2.0",
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--models-dir", type=Path, default=Path("models/release"),
                        help="Directory holding the ONNX files and README.md.")
    parser.add_argument("--out", type=Path, default=Path("hf_export"),
                        help="Directory to assemble the HuggingFace repo into.")
    parser.add_argument("--repo-id", type=str, default=None,
                        help="HuggingFace repo id (user_or_org/name) for --push.")
    parser.add_argument("--push", action="store_true",
                        help="Push --out to --repo-id (needs huggingface_hub + HF_TOKEN).")
    parser.add_argument("--private", action="store_true", help="Create the repo as private.")
    return parser.parse_args(argv)


def assemble(models_dir: Path, out: Path) -> list[Path]:
    out.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for model in MODEL_CONFIG["models"].values():
        src = models_dir / model["file"]
        if src.is_file():
            dst = out / model["file"]
            shutil.copy2(src, dst)
            written.append(dst)
        else:
            print(f"warning: {src} not found — skipping (add the weights first)")
    (out / "config.json").write_text(json.dumps(MODEL_CONFIG, indent=2), encoding="utf-8")
    written.append(out / "config.json")
    card = models_dir / "README.md"
    if card.is_file():
        shutil.copy2(card, out / "README.md")
        written.append(out / "README.md")
    return written


def push(out: Path, repo_id: str, *, private: bool) -> None:
    import os

    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", private=private, exist_ok=True)
    api.upload_folder(repo_id=repo_id, repo_type="model", folder_path=str(out))
    print(f"pushed {out} -> https://huggingface.co/{repo_id}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    written = assemble(args.models_dir, args.out)
    for p in written:
        print(f"assembled {p}")
    if args.push:
        if not args.repo_id:
            raise SystemExit("--push requires --repo-id user_or_org/name")
        push(args.out, args.repo_id, private=args.private)
    else:
        print("assembled locally. Add --repo-id and --push (with HF_TOKEN set) to upload.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

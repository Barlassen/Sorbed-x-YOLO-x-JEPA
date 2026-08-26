#!/usr/bin/env python3
"""Download SAM / MedSAM / SAM-2 weights with NO API key and NO login.

Every source here is a **public** endpoint reachable with a plain HTTPS GET from
a server with open internet — no HuggingFace token, no Roboflow/Kaggle key, no
gated access request. Two kinds of weights are offered:

* **transformers-format** (``config.json`` + a weights file + preprocessor) —
  loadable directly by ``training.finetune_medsam`` (which uses the HF
  ``transformers`` ``SamModel``/``SamProcessor`` API). This is the path you want
  for Sorbed's MedSAM mask-factory. Point the fine-tuner at the download dir:

      python -m training.fetch_models medsam-vit-base --out /data/briefer/models
      python -m training.finetune_medsam train \\
          --weights-dir /data/briefer/models/medsam-vit-base ...

* **original-format** SAM / SAM-2 checkpoints from Meta's public CDN (``.pth`` /
  ``.pt``) for the native ``segment-anything`` / ``sam2`` codebases. These need
  conversion before the ``transformers`` path can load them, so prefer the
  transformers-format MedSAM above for Sorbed.

``SAM 3`` exists but its checkpoints are **gated** (HuggingFace access request +
token), so there is no no-auth URL — it is intentionally not offered here.

All URLs were verified against the official ``facebookresearch/segment-anything``
/ ``facebookresearch/sam2`` READMEs and public HuggingFace repos; none are
fabricated. Run ``--list`` to see them without downloading.
"""

from __future__ import annotations

import argparse
import sys
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path

_HF = "https://huggingface.co"
_META = "https://dl.fbaipublicfiles.com"
_UA = "sorbed-fetch-models/1.0 (+https://github.com/ArioMoniri/Sorbed)"


@dataclass(frozen=True)
class ModelSpec:
    """One downloadable model. ``hf_repo`` uses the transformers layout."""

    name: str
    fmt: str  # "transformers" | "original"
    note: str
    hf_repo: str | None = None
    # transformers: config + preprocessor + first weights file that exists.
    hf_files: tuple[str, ...] = ("config.json", "preprocessor_config.json")
    hf_weight_candidates: tuple[str, ...] = ("model.safetensors", "pytorch_model.bin")
    # original: direct CDN URLs.
    urls: tuple[str, ...] = field(default_factory=tuple)


# Transformers-format (no token) — usable directly by finetune_medsam.
_TRANSFORMERS: tuple[ModelSpec, ...] = (
    ModelSpec(
        "medsam-vit-base", "transformers",
        "MedSAM in transformers format (public, apache-2.0). The mask-factory default.",
        hf_repo="flaviagiammarino/medsam-vit-base",
    ),
    ModelSpec(
        "medsam-vit-base-wanglab", "transformers",
        "MedSAM in transformers format from the Wang-lab HF mirror (public).",
        hf_repo="wanglab/medsam-vit-base",
    ),
    ModelSpec(
        "sam-vit-base", "transformers",
        "Meta SAM ViT-B in transformers format (public). Generic SAM baseline.",
        hf_repo="facebook/sam-vit-base",
    ),
    ModelSpec(
        "sam-vit-large", "transformers",
        "Meta SAM ViT-L in transformers format (public).",
        hf_repo="facebook/sam-vit-large",
    ),
    ModelSpec(
        "sam-vit-huge", "transformers",
        "Meta SAM ViT-H in transformers format (public).",
        hf_repo="facebook/sam-vit-huge",
    ),
)

# Original-format Meta CDN checkpoints (need conversion for the transformers path).
_ORIGINAL: tuple[ModelSpec, ...] = (
    ModelSpec("sam-vit-b-orig", "original", "Meta SAM ViT-B original .pth (~375 MB).",
              urls=(f"{_META}/segment_anything/sam_vit_b_01ec64.pth",)),
    ModelSpec("sam-vit-l-orig", "original", "Meta SAM ViT-L original .pth (~1.25 GB).",
              urls=(f"{_META}/segment_anything/sam_vit_l_0b3195.pth",)),
    ModelSpec("sam-vit-h-orig", "original", "Meta SAM ViT-H original .pth (~2.56 GB).",
              urls=(f"{_META}/segment_anything/sam_vit_h_4b8939.pth",)),
    ModelSpec("sam2.1-hiera-base-plus", "original", "Meta SAM 2.1 hiera base+ .pt (~309 MB).",
              urls=(f"{_META}/segment_anything_2/092824/sam2.1_hiera_base_plus.pt",)),
    ModelSpec("sam2.1-hiera-large", "original", "Meta SAM 2.1 hiera large .pt (~856 MB).",
              urls=(f"{_META}/segment_anything_2/092824/sam2.1_hiera_large.pt",)),
    ModelSpec("medsam2-latest", "original",
              "MedSAM-2 (SAM2-based) from wanglab/MedSAM2 (public); needs the sam2 codebase.",
              urls=(f"{_HF}/wanglab/MedSAM2/resolve/main/MedSAM2_latest.pt",)),
)

REGISTRY: dict[str, ModelSpec] = {m.name: m for m in (*_TRANSFORMERS, *_ORIGINAL)}


def _download(url: str, dest: Path, *, dry_run: bool) -> bool:
    """Stream ``url`` to ``dest`` (follows redirects). ``False`` on HTTP 404."""
    if dry_run:
        print(f"    [dry-run] GET {url}")
        return True
    dest.parent.mkdir(parents=True, exist_ok=True)
    request = urllib.request.Request(url, headers={"User-Agent": _UA})
    try:
        with urllib.request.urlopen(request) as response:
            tmp = dest.with_suffix(dest.suffix + ".part")
            total = 0
            with tmp.open("wb") as handle:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    handle.write(chunk)
                    total += len(chunk)
            tmp.replace(dest)
            print(f"    saved {dest.name} ({total / 1e6:.1f} MB)")
            return True
    except urllib.error.HTTPError as exc:
        if exc.code == 404:
            return False
        raise


def _hf_url(repo: str, filename: str) -> str:
    return f"{_HF}/{repo}/resolve/main/{filename}"


def fetch(spec: ModelSpec, out_root: Path, *, dry_run: bool) -> Path:
    """Download one model into ``out_root/<name>/``; return that directory."""
    dest_dir = out_root / spec.name
    print(f"[{spec.fmt}] {spec.name} -> {dest_dir}")
    if spec.fmt == "transformers":
        assert spec.hf_repo is not None
        for filename in spec.hf_files:
            if not _download(_hf_url(spec.hf_repo, filename), dest_dir / filename, dry_run=dry_run):
                print(f"    warning: {filename} not found in {spec.hf_repo}", file=sys.stderr)
        got_weights = False
        for candidate in spec.hf_weight_candidates:
            if _download(_hf_url(spec.hf_repo, candidate), dest_dir / candidate, dry_run=dry_run):
                got_weights = True
                break
        if not got_weights and not dry_run:
            raise SystemExit(
                f"no weights file ({' / '.join(spec.hf_weight_candidates)}) in {spec.hf_repo}"
            )
    else:
        for url in spec.urls:
            filename = url.rsplit("/", 1)[-1]
            if not _download(url, dest_dir / filename, dry_run=dry_run):
                raise SystemExit(f"download failed (404): {url}")
    return dest_dir


def print_list() -> None:
    print("No-auth SAM / MedSAM / SAM-2 weights (public; plain HTTPS, no token)")
    print("=" * 78)
    print("\ntransformers-format (load directly with training.finetune_medsam --weights-dir):")
    for spec in _TRANSFORMERS:
        print(f"  {spec.name:26s} repo={spec.hf_repo}")
        print(f"      {spec.note}")
    print("\noriginal-format (Meta/HF CDN; need conversion for the transformers path):")
    for spec in _ORIGINAL:
        print(f"  {spec.name:26s} {spec.urls[0]}")
        print(f"      {spec.note}")
    print("\nGATED (NOT offered — needs a HuggingFace token): facebook/sam3, facebook/sam3.1")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("models", nargs="*", help="model names to fetch (see --list)")
    parser.add_argument("--out", type=Path, default=Path("models"),
                        help="download root (default: ./models)")
    parser.add_argument("--list", action="store_true", help="list sources and exit")
    parser.add_argument("--dry-run", action="store_true",
                        help="print the URLs that would be fetched without downloading")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.list or not args.models:
        print_list()
        if not args.models:
            print("\nPass one or more model names to download, e.g.:")
            print("  python -m training.fetch_models medsam-vit-base --out /data/briefer/models")
        return 0
    unknown = [m for m in args.models if m not in REGISTRY]
    if unknown:
        raise SystemExit(f"unknown model(s): {', '.join(unknown)} (see --list)")
    for name in args.models:
        dest = fetch(REGISTRY[name], args.out, dry_run=args.dry_run)
        if REGISTRY[name].fmt == "transformers":
            print(f"    usable now: training.finetune_medsam train --weights-dir {dest} ...")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

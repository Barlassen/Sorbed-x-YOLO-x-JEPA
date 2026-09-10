#!/usr/bin/env python3
"""Turn ComplexWoundDB's four expert annotations into a wound-region ruler.

ComplexWoundDB ships 27 clinical wound photos — most of them **pressure injuries**
on the sacrum, trochanter, gluteal region and heel — each annotated independently
by **four** experts. The masks are colour-coded by tissue (white = intact skin /
background; red = granulation; yellow = slough/fibrin; black = necrosis; plus two
minor classes), so the wound *region* is simply "every pixel an expert did not
call background".

That gives Sorbed two things it did not have:

1. **A pressure-domain segmentation ruler.** Every other wound-mask test we own
   is foot (AZH/FUSeg, DFUTissue), and the segmenter's Dice collapses from 0.82
   in-domain to 0.46 out-of-domain — so an in-domain-for-pressure ruler is the
   only way to know how good our masks actually are where Sorbed is used.
2. **A human ceiling.** Four experts disagree with each other by a measurable
   amount; a model cannot meaningfully beat that. This script reports mean
   pairwise Dice between experts alongside the consensus mask it writes.

Consensus is a per-pixel vote: a pixel is wound when at least ``--min-votes`` of
the four experts marked it as non-background (default 3 of 4, a strict majority).

    python -m training.prepare_complexwound_ruler --out data/ruler_complexwound

Writes ``<out>/images`` (sym-links) + ``<out>/masks`` (0/255 PNGs) and an
``agreement.json`` report. Pure data preparation — no GPU, no network.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

_REPO = Path(__file__).resolve().parent.parent
_DEFAULT_ROOT = _REPO / "data/corpus/complexwounddb/ComplexWoundDB"
_BACKGROUND = (255, 255, 255)  # white = intact skin / not wound


def wound_mask(path: Path) -> np.ndarray | None:
    """Boolean wound region from one expert's colour-coded mask."""
    try:
        rgb = np.array(Image.open(path).convert("RGB"))
    except (OSError, ValueError):
        return None
    return ~np.all(rgb == np.array(_BACKGROUND, dtype=np.uint8), axis=-1)


def dice(a: np.ndarray, b: np.ndarray) -> float:
    """Dice between two boolean masks; both-empty counts as perfect agreement."""
    total = int(a.sum() + b.sum())
    if total == 0:
        return 1.0
    return float(2 * np.logical_and(a, b).sum() / total)


def build(root: Path, out: Path, min_votes: int) -> dict[str, object]:
    """Write consensus masks and return the agreement report."""
    images_dir = root / "images"
    masks_root = root / "annotations" / "masks"
    experts = sorted(d for d in masks_root.iterdir() if d.is_dir())
    if not experts:
        raise SystemExit(f"no expert directories under {masks_root}")

    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "masks").mkdir(parents=True, exist_ok=True)

    pairwise: list[float] = []
    per_image: list[dict[str, object]] = []
    written = 0

    for img_path in sorted(images_dir.glob("*.png"), key=lambda p: int(p.stem)):
        votes: list[np.ndarray] = []
        for expert in experts:
            m = wound_mask(expert / img_path.name)
            if m is not None:
                votes.append(m)
        if len(votes) < min_votes:
            continue
        shape = votes[0].shape
        if any(v.shape != shape for v in votes):
            continue  # shape mismatch between experts — skip rather than guess

        # Human ceiling: how much do the experts agree with each other?
        image_pairs = [
            dice(votes[i], votes[j])
            for i in range(len(votes))
            for j in range(i + 1, len(votes))
        ]
        pairwise.extend(image_pairs)

        consensus = np.sum(votes, axis=0) >= min_votes
        Image.fromarray((consensus * 255).astype(np.uint8)).save(out / "masks" / f"{img_path.stem}.png")

        link = out / "images" / img_path.name
        if link.exists() or link.is_symlink():
            link.unlink()
        link.symlink_to(img_path.resolve())

        per_image.append({
            "image": img_path.name,
            "experts": len(votes),
            "mean_pairwise_dice": round(float(np.mean(image_pairs)), 4) if image_pairs else None,
            "consensus_wound_frac": round(float(consensus.mean()), 4),
        })
        written += 1

    report: dict[str, object] = {
        "root": str(root),
        "experts": [e.name for e in experts],
        "min_votes": min_votes,
        "images": written,
        "human_ceiling_mean_pairwise_dice": round(float(np.mean(pairwise)), 4) if pairwise else None,
        "human_ceiling_std": round(float(np.std(pairwise)), 4) if pairwise else None,
        "human_ceiling_min": round(float(np.min(pairwise)), 4) if pairwise else None,
        "per_image": per_image,
    }
    (out / "agreement.json").write_text(json.dumps(report, indent=2))
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="ComplexWoundDB -> wound-region ruler + human ceiling.")
    ap.add_argument("--root", type=Path, default=_DEFAULT_ROOT, help="ComplexWoundDB directory.")
    ap.add_argument("--out", type=Path, required=True, help="Output ruler directory.")
    ap.add_argument("--min-votes", type=int, default=3, help="Experts that must agree a pixel is wound.")
    return ap.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    report = build(args.root, args.out, args.min_votes)
    print(f"ruler -> {args.out}  ({report['images']} images, {len(report['experts'])} experts, "
          f"min_votes={report['min_votes']})")
    print(f"human ceiling (mean pairwise expert Dice): {report['human_ceiling_mean_pairwise_dice']} "
          f"± {report['human_ceiling_std']}  (worst pair {report['human_ceiling_min']})")
    print(f"report -> {args.out / 'agreement.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

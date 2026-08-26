# Model Card — Classical Segmentation Backend

- **Name:** `classical`
- **Backend:** `classical` (weight-free)
- **License:** Apache-2.0 (code); no third-party weights or data
- **Weights:** none — needs no download and runs fully offline
- **Version:** ships with the Sorbed core

## Overview

The classical backend is Sorbed's always-available default wound-boundary
segmenter. It requires no trained weights, no network access, and no GPU, which
guarantees the pipeline produces a result even before any learned model is
fetched. It is an **interpretable baseline, not a clinical-grade detector.**

## Method

Implemented in `src/sorbed/segmentation/classical.py`:

1. Convert the image to CIE-Lab.
2. Sample a healthy-skin / background reference from the image border.
3. Compute, per pixel, the standardized Lab distance from that reference
   ("woundness"), smoothed with a Gaussian.
4. Threshold with Otsu, clean morphologically, fill holes, and keep the largest
   connected component.
5. Refine the boundary with GrabCut, seeded from the coarse mask.

The reported **confidence is computed from the image** — the normalized contrast
between wound and non-wound regions, penalized when the mask spans nearly the
whole frame. It is never a constant. On a low-contrast or blank image the backend
returns a small/empty mask and a low confidence, which makes the staging engine
abstain rather than fabricate a result.

## Intended use

- A dependable fallback that keeps the whole system operational offline.
- A transparent, pixel-explainable baseline for comparison against learned
  backends.
- Bootstrapping / triage where interpretability matters more than peak accuracy.

## Limitations

- **Not clinical-grade.** It does not "understand" wounds; it detects regions
  that differ from border skin. Non-wound anomalies (shadows, tattoos, dressings,
  rulers) can be picked up, and low-contrast wounds can be missed.
- **Lighting sensitivity.** Uneven illumination and specular highlights shift the
  Lab distance and degrade the mask.
- **Skin-tone caveat.** The border-reference and Lab-distance approach behaves
  differently across skin tones and against varied backgrounds; contrast between
  wound and healthy skin is not uniform across the population. Treat results on
  darker skin tones with particular caution, and prefer a learned backend that
  has been evaluated for skin-tone fairness (see `TEMPLATE.md`).
- **Single-region assumption.** Keeping the largest connected component means
  multiple separate wounds are not all captured.

## Provenance

No weights, no dataset, no download. The method derives from the color-space /
classical lineage discussed in `docs/MODELS.md` (see the
[uwm-bigdata wound-segmentation](https://github.com/uwm-bigdata/wound-segmentation)
family of approaches).

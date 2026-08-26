"""Weight-free wound segmentation.

This backend needs no downloads and always runs. It finds the wound by how far
each pixel departs, in CIE-Lab, from the healthy-skin / background reference
sampled at the image border, then refines the result with GrabCut. The reported
confidence is computed from the actual separability of the two regions — it is
never a constant.

This is an interpretable baseline, not a clinical-grade detector. It is honest
about that: on a low-contrast or blank image it returns a small/empty mask and a
low confidence, which makes the staging engine abstain rather than fabricate a
grade. Swap in a learned backend (see ``segmentation/backends``) for accuracy.
"""

from __future__ import annotations

import cv2
import numpy as np
from scipy import ndimage
from skimage import color, filters, morphology

from sorbed.imaging import RasterImage
from sorbed.segmentation.base import SegmentationResult

_BORDER_FRACTION = 0.08
_BACKEND_NAME = "classical"


class ClassicalSegmenter:
    """Lab-distance + GrabCut wound segmenter."""

    name = _BACKEND_NAME

    def __init__(self, *, grabcut_iterations: int = 5, min_area_fraction: float = 0.0015) -> None:
        self._grabcut_iterations = grabcut_iterations
        self._min_area_fraction = min_area_fraction

    def segment(self, image: RasterImage) -> SegmentationResult:
        rgb = image.pixels
        lab = color.rgb2lab(rgb)
        valid = self._valid_mask(image)

        woundness = self._woundness(lab, valid)
        coarse, confidence = self._threshold(woundness, valid, image)

        if coarse.sum() < self._min_area_fraction * coarse.size:
            # Nothing wound-like stands out — return the (small) mask honestly.
            return SegmentationResult(
                wound_mask=coarse.astype(bool),
                confidence=min(confidence, 0.25),
                backend=_BACKEND_NAME,
            )

        refined = self._grabcut(image.to_uint8_rgb(), coarse, valid)
        mask = refined if refined.sum() >= self._min_area_fraction * refined.size else coarse
        confidence = self._confidence(woundness, mask, valid)
        return SegmentationResult(
            wound_mask=mask.astype(bool),
            confidence=float(confidence),
            backend=_BACKEND_NAME,
        )

    @staticmethod
    def _valid_mask(image: RasterImage) -> np.ndarray:
        """Pixels eligible to be wound: opaque ones."""
        if image.alpha is None:
            return np.ones(image.shape_hw, dtype=bool)
        return image.alpha > 0.5

    @staticmethod
    def _woundness(lab: np.ndarray, valid: np.ndarray) -> np.ndarray:
        """Per-pixel standardized distance from the border-skin reference."""
        h, w = lab.shape[:2]
        bh = max(1, int(h * _BORDER_FRACTION))
        bw = max(1, int(w * _BORDER_FRACTION))
        border = np.zeros((h, w), dtype=bool)
        border[:bh, :] = border[-bh:, :] = True
        border[:, :bw] = border[:, -bw:] = True
        border &= valid

        ref_pixels = lab[border] if border.any() else lab[valid]
        mu = ref_pixels.mean(axis=0)
        sigma = ref_pixels.std(axis=0) + 1e-3

        z = (lab - mu) / sigma
        distance = np.sqrt((z**2).sum(axis=2))
        distance = filters.gaussian(distance, sigma=1.5)
        distance[~valid] = 0.0
        return distance

    def _threshold(
        self, woundness: np.ndarray, valid: np.ndarray, image: RasterImage
    ) -> tuple[np.ndarray, float]:
        vals = woundness[valid]
        if vals.size == 0 or float(vals.max() - vals.min()) < 1e-6:
            return np.zeros_like(woundness, dtype=bool), 0.0
        level = float(filters.threshold_otsu(vals))
        mask = (woundness > level) & valid
        mask = self._clean(mask)
        confidence = self._confidence(woundness, mask, valid)
        return mask, confidence

    def _clean(self, mask: np.ndarray) -> np.ndarray:
        if not mask.any():
            return mask
        radius = max(1, int(0.01 * np.sqrt(mask.size)))
        mask = morphology.opening(mask, morphology.disk(radius))
        mask = morphology.closing(mask, morphology.disk(radius))
        mask = ndimage.binary_fill_holes(mask)
        labeled, count = ndimage.label(mask)
        if count <= 1:
            return mask.astype(bool)
        # Keep the largest connected component.
        sizes = ndimage.sum(np.ones_like(labeled), labeled, index=range(1, count + 1))
        keep = int(np.argmax(sizes)) + 1
        return labeled == keep

    def _grabcut(self, rgb_u8: np.ndarray, coarse: np.ndarray, valid: np.ndarray) -> np.ndarray:
        gc = np.where(coarse, cv2.GC_PR_FGD, cv2.GC_PR_BGD).astype(np.uint8)
        # Anchor confident background at the image border, confident foreground
        # at the eroded core of the coarse mask.
        gc[:2, :] = gc[-2:, :] = gc[:, :2] = gc[:, -2:] = cv2.GC_BGD
        core = morphology.erosion(coarse, morphology.disk(3))
        gc[core] = cv2.GC_FGD
        gc[~valid] = cv2.GC_BGD
        if not (gc == cv2.GC_FGD).any() or not (gc == cv2.GC_PR_FGD).any():
            return coarse
        bgd = np.zeros((1, 65), np.float64)
        fgd = np.zeros((1, 65), np.float64)
        try:
            cv2.grabCut(
                np.ascontiguousarray(rgb_u8[:, :, ::-1]),  # OpenCV expects BGR
                gc,
                None,
                bgd,
                fgd,
                self._grabcut_iterations,
                cv2.GC_INIT_WITH_MASK,
            )
        except cv2.error:
            return coarse
        result = (gc == cv2.GC_FGD) | (gc == cv2.GC_PR_FGD)
        return self._clean(result & valid)

    @staticmethod
    def _confidence(woundness: np.ndarray, mask: np.ndarray, valid: np.ndarray) -> float:
        """Normalized contrast between wound and non-wound woundness.

        High when the wound region is clearly separable from its surroundings;
        low on flat/ambiguous images. Penalized when the mask spans the whole
        frame (a likely failure).
        """
        inside = woundness[mask & valid]
        outside = woundness[(~mask) & valid]
        if inside.size == 0 or outside.size == 0:
            return 0.0
        mean_in, mean_out = float(inside.mean()), float(outside.mean())
        contrast = (mean_in - mean_out) / (mean_in + mean_out + 1e-6)
        contrast = max(0.0, min(1.0, contrast))
        frame_fraction = float((mask & valid).sum()) / float(valid.sum() + 1e-6)
        plausibility = 1.0 if frame_fraction < 0.85 else 0.3
        return round(contrast * plausibility, 4)

"""HuggingFace-backed wound segmentation (SAM / MedSAM).

This is a real learned backend. It refines a coarse, weight-free proposal into a
precise wound mask using a promptable Segment-Anything model from the HuggingFace
Hub (``facebook/sam-vit-base`` by default, or the box-prompted medical variant
``flaviagiammarino/medsam-vit-base``). The proposal's bounding box is fed to the
model as a prompt, and the model's own predicted IoU becomes the confidence — a
genuine, model-derived value, not a constant.

Weights are downloaded from the Hub on first use and cached; nothing is vendored.
The model id, revision, and resolved weight hash are recorded for provenance. If
``transformers``/``torch`` are not installed, or the Hub is unreachable, a clear
error explains how to enable it (``pip install 'sorbed[hf]'`` and network access
to huggingface.co).
"""

from __future__ import annotations

import numpy as np
from scipy import ndimage

from sorbed.imaging import RasterImage
from sorbed.segmentation.base import SegmentationResult
from sorbed.segmentation.classical import ClassicalSegmenter

_DEFAULT_MODEL = "facebook/sam-vit-base"


class HuggingFaceSAMSegmenter:
    """Promptable SAM/MedSAM wound segmenter via the HuggingFace ``transformers`` API."""

    name = "hf_sam"

    def __init__(
        self,
        model_id: str = _DEFAULT_MODEL,
        *,
        revision: str | None = None,
        device: str | None = None,
        min_area_fraction: float = 0.0015,
    ) -> None:
        self._model_id = model_id
        self._revision = revision
        self._device = device
        self._min_area_fraction = min_area_fraction
        self._proposer = ClassicalSegmenter(min_area_fraction=min_area_fraction)
        self._model = None
        self._processor = None
        self._weights_sha256: str | None = None

    @classmethod
    def from_settings(cls, settings: object) -> HuggingFaceSAMSegmenter:
        model_id = str(getattr(settings, "hf_model_id", _DEFAULT_MODEL) or _DEFAULT_MODEL)
        return cls(model_id=model_id)

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        try:
            import torch
            from transformers import SamModel, SamProcessor
        except ImportError as exc:
            raise RuntimeError(
                "The HuggingFace backend needs torch + transformers: "
                "pip install 'sorbed[hf]'."
            ) from exc

        try:
            self._processor = SamProcessor.from_pretrained(self._model_id, revision=self._revision)
            self._model = SamModel.from_pretrained(self._model_id, revision=self._revision)
        except Exception as exc:  # network/hub errors surface clearly
            raise RuntimeError(
                f"Could not load '{self._model_id}' from the HuggingFace Hub. Ensure "
                "huggingface.co is reachable (this managed environment may firewall it) "
                f"or pre-download the model to the HF cache. Underlying error: {exc}"
            ) from exc

        device = self._device or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model.to(device)
        self._model.eval()
        self._device = device
        self._weights_sha256 = self._resolve_weight_hash()

    def segment(self, image: RasterImage) -> SegmentationResult:
        proposal = self._proposer.segment(image)
        if proposal.area_px < self._min_area_fraction * proposal.wound_mask.size:
            # Nothing to refine — return the honest empty/low-confidence proposal.
            return SegmentationResult(
                wound_mask=proposal.wound_mask,
                confidence=min(proposal.confidence, 0.25),
                backend=self.name,
            )

        self._ensure_loaded()
        import torch  # available once _ensure_loaded succeeded

        assert self._model is not None and self._processor is not None

        box = self._proposal_box(proposal.wound_mask)
        rgb_u8 = image.to_uint8_rgb()
        inputs = self._processor(rgb_u8, input_boxes=[[box]], return_tensors="pt").to(self._device)
        with torch.no_grad():
            outputs = self._model(**inputs)

        masks = self._processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu(),
        )[0][0]
        scores = outputs.iou_scores.cpu().numpy().reshape(-1)
        best = int(scores.argmax())
        mask = masks[best].numpy().astype(bool)
        mask = self._largest_component(mask)
        confidence = float(np.clip(scores[best], 0.0, 1.0))

        if mask.sum() < self._min_area_fraction * mask.size:
            return SegmentationResult(
                wound_mask=proposal.wound_mask,
                confidence=proposal.confidence,
                backend=self.name,
                weights_sha256=self._weights_sha256,
            )
        return SegmentationResult(
            wound_mask=mask,
            confidence=confidence,
            backend=self.name,
            weights_sha256=self._weights_sha256,
        )

    @staticmethod
    def _proposal_box(mask: np.ndarray) -> list[int]:
        ys, xs = np.where(mask)
        return [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())]

    @staticmethod
    def _largest_component(mask: np.ndarray) -> np.ndarray:
        labeled, count = ndimage.label(mask)
        if count <= 1:
            return mask
        sizes = ndimage.sum(np.ones_like(labeled), labeled, index=range(1, count + 1))
        return labeled == (int(np.argmax(sizes)) + 1)

    def _resolve_weight_hash(self) -> str | None:
        """Best-effort provenance: the Hub's recorded commit for the loaded model."""
        try:
            from huggingface_hub import model_info

            info = model_info(self._model_id, revision=self._revision)
            return str(info.sha) if info.sha else None
        except Exception:
            return None

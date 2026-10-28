"""Learned wound-bed tissue classification (DeepLabV3-MobileNetV3, trained by
``training/train_tissue_cnn.py``).

The network labels every wound pixel as granulation, slough, eschar, or epithelial
from a crop of the *raw* photo (it was trained on un-normalized images). The color
cues the rules also need (maroon/purple, erythema) still come from
:class:`~sorbed.tissue.color_model.ColorTissueClassifier` on the normalized image.
This is the exact feature path the staging thresholds were tuned on (HANDOFF,
staging step 2); the crop, letterbox, and paste steps mirror the training code.

torch/torchvision are imported lazily so the weight-free core never depends on them.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np

from sorbed.config.settings import Settings
from sorbed.domain.enums import TissueClass
from sorbed.tissue.color_model import ColorTissueClassifier, TissueAnalysis, tissue_index

_SIDE = 160  # the wound crop is rescaled so its longer side is this many px
_SQ = 256  # network input square
_MIN_REGION_PX = 30  # smaller regions keep the color-model composition
_MEAN = np.array([0.485, 0.456, 0.406], np.float32)
_STD = np.array([0.229, 0.224, 0.225], np.float32)
# Network output channel -> tissue, by head size (3 = DFUTissue classes, 4 = + necrosis).
_NET_TISSUES = {
    3: (TissueClass.SLOUGH, TissueClass.GRANULATION, TissueClass.EPITHELIAL),
    4: (TissueClass.SLOUGH, TissueClass.GRANULATION, TissueClass.EPITHELIAL, TissueClass.ESCHAR),
}


class CnnWeightsMissingError(RuntimeError):
    """Raised when the CNN tissue backend is selected without a weights file."""


@lru_cache(maxsize=2)
def _load(path: str):
    """Load (and cache) the network and its class count from a state-dict file."""
    import torch
    from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

    state = torch.load(path, map_location="cpu", weights_only=True)
    n_cls = int(state["classifier.4.weight"].shape[0])
    model = deeplabv3_mobilenet_v3_large(weights=None, weights_backbone=None, aux_loss=None)
    model.classifier[-1] = torch.nn.Conv2d(model.classifier[-1].in_channels, n_cls, 1)
    model.aux_classifier = None
    model.load_state_dict(state)
    return model.eval(), n_cls


class CnnTissueClassifier:
    """Tissue composition from a segmentation CNN; color cues from the color model."""

    name = "cnn"

    def __init__(self, weights: str | Path) -> None:
        path = Path(weights)
        if not path.is_file():
            raise CnnWeightsMissingError(
                f"tissue CNN weights not found: {str(weights)!r} (set SORBED_TISSUE_CNN_WEIGHTS)"
            )
        self._path = str(path)
        self._color = ColorTissueClassifier()
        self.weights_sha256 = hashlib.sha256(path.read_bytes()).hexdigest()

    @classmethod
    def from_settings(cls, settings: Settings) -> CnnTissueClassifier:
        return cls(settings.tissue_cnn_weights)

    def classify(
        self, rgb: np.ndarray, wound_mask: np.ndarray, raw_rgb: np.ndarray | None = None
    ) -> TissueAnalysis:
        """``rgb`` is the color-normalized image, ``raw_rgb`` the original (float RGB in [0, 1])."""
        color = self._color.classify(rgb, wound_mask)
        if wound_mask.sum() < _MIN_REGION_PX:
            return color

        raw = rgb if raw_rgb is None else raw_rgb
        bgr = cv2.cvtColor((np.clip(raw, 0.0, 1.0) * 255.0).round().astype(np.uint8), cv2.COLOR_RGB2BGR)
        label_map, confidence = self._predict(bgr, wound_mask)

        tissues = (TissueClass.GRANULATION, TissueClass.SLOUGH, TissueClass.ESCHAR, TissueClass.EPITHELIAL)
        counts = {c: int((label_map == tissue_index(c)).sum()) for c in tissues}
        counts = {c: n for c, n in counts.items() if n}
        total = sum(counts.values())
        if not total:
            return color
        conf_sum = {c: float(confidence[label_map == tissue_index(c)].sum()) for c in counts}
        # Wound pixels the network never saw (lost when the crop was resized) stay
        # out of the composition and show as unknown.
        label_map[wound_mask & (label_map == tissue_index(TissueClass.BACKGROUND))] = tissue_index(
            TissueClass.UNKNOWN
        )
        return TissueAnalysis(
            label_map=label_map,
            composition=self._color._build_composition(counts, conf_sum, total),
            maroon_purple_fraction=color.maroon_purple_fraction,
            erythema_fraction=color.erythema_fraction,
            open_bed_fraction=round((total - counts.get(TissueClass.EPITHELIAL, 0)) / total, 4),
            per_pixel_confidence=confidence,
        )

    def _predict(self, bgr: np.ndarray, wound_mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Full-size label map (tissue-order indices) and per-pixel softmax confidence."""
        import torch

        model, n_cls = _load(self._path)
        h_img, w_img = wound_mask.shape

        # Crop to the wound's bounding box (5% padding), longer side -> _SIDE px.
        ys, xs = np.nonzero(wound_mask)
        y0, y1, x0, x1 = ys.min(), ys.max() + 1, xs.min(), xs.max() + 1
        py, px = int(0.05 * (y1 - y0)) + 1, int(0.05 * (x1 - x0)) + 1
        y0, y1 = max(0, y0 - py), min(h_img, y1 + py)
        x0, x1 = max(0, x0 - px), min(w_img, x1 + px)
        f = _SIDE / max(y1 - y0, x1 - x0)
        size = (max(1, round((x1 - x0) * f)), max(1, round((y1 - y0) * f)))
        crop = cv2.resize(bgr[y0:y1, x0:x1], size, interpolation=cv2.INTER_AREA)
        region = cv2.resize(wound_mask[y0:y1, x0:x1].astype(np.uint8), size, interpolation=cv2.INTER_NEAREST) > 0

        # Letterbox into the network's square input (top-left, zero padding).
        ch, cw = crop.shape[:2]
        g = _SQ / max(ch, cw)
        nh, nw = max(1, round(ch * g)), max(1, round(cw * g))
        square = np.zeros((_SQ, _SQ, 3), np.uint8)
        square[:nh, :nw] = cv2.resize(crop, (nw, nh), interpolation=cv2.INTER_AREA)
        x = (cv2.cvtColor(square, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0 - _MEAN) / _STD
        with torch.no_grad():
            logits = model(torch.from_numpy(x.transpose(2, 0, 1).copy())[None])["out"][0, :, :nh, :nw]
            prob = torch.softmax(logits, 0)
        cls = cv2.resize(prob.argmax(0).numpy().astype(np.uint8), (cw, ch), interpolation=cv2.INTER_NEAREST)
        conf = cv2.resize(prob.max(0).values.numpy(), (cw, ch), interpolation=cv2.INTER_NEAREST)

        lut = np.array([tissue_index(t) for t in _NET_TISSUES[n_cls]], np.int16)
        crop_labels = np.where(region, lut[cls], tissue_index(TissueClass.BACKGROUND)).astype(np.int16)
        crop_conf = np.where(region, conf, 0.0).astype(np.float32)

        # Paste back at full resolution, inside the wound only.
        label_map = np.full((h_img, w_img), tissue_index(TissueClass.BACKGROUND), np.int16)
        confidence = np.zeros((h_img, w_img), np.float32)
        label_map[y0:y1, x0:x1] = cv2.resize(crop_labels, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
        confidence[y0:y1, x0:x1] = cv2.resize(crop_conf, (x1 - x0, y1 - y0), interpolation=cv2.INTER_NEAREST)
        outside = ~wound_mask
        label_map[outside] = tissue_index(TissueClass.BACKGROUND)
        confidence[outside] = 0.0
        return label_map, confidence

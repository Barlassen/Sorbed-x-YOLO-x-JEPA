"""ONNX Runtime wound-segmentation backend.

This backend runs a trained segmentation network (typically a U-Net exported by
``scripts/train_segmenter.py``) through ONNX Runtime. It is deliberately model-
agnostic about output layout: it accepts a single foreground logit
``(1, 1, H, W)`` / ``(1, H, W)`` / ``(H, W)`` or a two-class map ``(1, 2, H, W)``,
and turns it into a boolean wound mask plus an evidence-derived confidence.

The pipeline in :func:`sorbed.pipeline.backends.build_segmenter` instantiates this
class via :meth:`OnnxSegmenter.from_settings` when ``segmentation_backend`` is
``"onnx"`` or ``"smp_unet"``. The model path is read from the ``SORBED_ONNX_MODEL``
environment variable (the frozen :class:`~sorbed.config.settings.Settings` object
has no field for it), and its SHA-256 is recorded on every result so an analysis
can be tied back to the exact weights that produced it.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort
from scipy import ndimage

from sorbed.config.settings import Settings
from sorbed.imaging import RasterImage
from sorbed.segmentation.base import SegmentationResult

_BACKEND_NAME = "onnx"
_ENV_MODEL = "SORBED_ONNX_MODEL"
_ENV_MEAN = "SORBED_ONNX_MEAN"
_ENV_STD = "SORBED_ONNX_STD"
_ENV_INPUT = "SORBED_ONNX_INPUT"

# ImageNet channel statistics — the default normalization for ImageNet-pretrained
# encoders (EfficientNet / ResNet), which is what the training script uses.
_IMAGENET_MEAN = (0.485, 0.456, 0.406)
_IMAGENET_STD = (0.229, 0.224, 0.225)

# Fallback spatial size used only when the model declares dynamic H/W axes.
_DEFAULT_INPUT = 512


def _sha256_file(path: Path) -> str:
    """Streaming SHA-256 of a file, so large weights are not read into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_triple(value: str) -> tuple[float, float, float]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if len(parts) != 3:
        raise RuntimeError(
            f"expected three comma-separated numbers, got {value!r}"
        )
    a, b, c = (float(p) for p in parts)
    return a, b, c


def _select_providers() -> list[str]:
    """Prefer CUDA when the runtime exposes it, always keep CPU as a fallback."""
    available = set(ort.get_available_providers())
    providers: list[str] = []
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    providers.append("CPUExecutionProvider")
    return providers


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def _softmax(x: np.ndarray, axis: int) -> np.ndarray:
    shifted = x - x.max(axis=axis, keepdims=True)
    exp = np.exp(shifted)
    return exp / exp.sum(axis=axis, keepdims=True)


class OnnxSegmenter:
    """Run a trained ONNX segmentation model and return a boolean wound mask."""

    name = _BACKEND_NAME

    def __init__(
        self,
        model_path: Path,
        *,
        mean: tuple[float, float, float] = _IMAGENET_MEAN,
        std: tuple[float, float, float] = _IMAGENET_STD,
        default_input: int = _DEFAULT_INPUT,
        min_area_fraction: float = 0.0015,
    ) -> None:
        self._model_path = Path(model_path)
        if not self._model_path.is_file():
            raise RuntimeError(f"ONNX model not found: {self._model_path}")
        self._mean = np.asarray(mean, dtype=np.float32).reshape(3, 1, 1)
        self._std = np.asarray(std, dtype=np.float32).reshape(3, 1, 1)
        self._default_input = int(default_input)
        self._min_area_fraction = float(min_area_fraction)
        self._weights_sha256 = _sha256_file(self._model_path)

        self._session = ort.InferenceSession(
            str(self._model_path), providers=_select_providers()
        )
        self._input_name = self._session.get_inputs()[0].name
        self._output_name = self._session.get_outputs()[0].name
        self._input_hw = self._resolve_input_hw()

    @classmethod
    def from_settings(cls, settings: Settings) -> OnnxSegmenter:
        raw = os.environ.get(_ENV_MODEL, "").strip()
        if not raw:
            raise RuntimeError(
                f"the {settings.segmentation_backend!r} segmentation backend needs a "
                f"trained model: set the {_ENV_MODEL} environment variable to an "
                "exported .onnx file, or run `sorbed models pull` to fetch a "
                "registered one."
            )
        model_path = Path(raw).expanduser()
        if not model_path.is_file():
            raise RuntimeError(
                f"{_ENV_MODEL} points to a missing file: {model_path}. Export a model "
                "with scripts/train_segmenter.py or run `sorbed models pull`."
            )

        mean = _IMAGENET_MEAN
        std = _IMAGENET_STD
        if (raw_mean := os.environ.get(_ENV_MEAN, "").strip()):
            mean = _parse_triple(raw_mean)
        if (raw_std := os.environ.get(_ENV_STD, "").strip()):
            std = _parse_triple(raw_std)
        default_input = _DEFAULT_INPUT
        if (raw_input := os.environ.get(_ENV_INPUT, "").strip()):
            default_input = int(raw_input)

        return cls(
            model_path,
            mean=mean,
            std=std,
            default_input=default_input,
            min_area_fraction=settings.min_wound_area_fraction,
        )

    @property
    def weights_sha256(self) -> str:
        return self._weights_sha256

    def _resolve_input_hw(self) -> tuple[int, int]:
        """Determine the model's expected (H, W), falling back on dynamic axes."""
        shape = list(self._session.get_inputs()[0].shape)
        height: int = self._default_input
        width: int = self._default_input
        if len(shape) == 4:  # (N, C, H, W)
            h_axis, w_axis = shape[2], shape[3]
            if isinstance(h_axis, int) and h_axis > 0:
                height = h_axis
            if isinstance(w_axis, int) and w_axis > 0:
                width = w_axis
        return height, width

    def segment(self, image: RasterImage) -> SegmentationResult:
        orig_h, orig_w = image.shape_hw
        in_h, in_w = self._input_hw

        tensor = self._preprocess(image.pixels, in_h, in_w)
        outputs = self._session.run([self._output_name], {self._input_name: tensor})
        prob_model = self._probabilities(np.asarray(outputs[0], dtype=np.float32))

        # Wound probability at the original resolution.
        prob = cv2.resize(
            prob_model, (orig_w, orig_h), interpolation=cv2.INTER_LINEAR
        ).astype(np.float32)

        valid = self._valid_mask(image)
        mask = (prob >= 0.5) & valid
        mask = self._largest_component(mask)

        confidence = self._confidence(prob, mask)
        return SegmentationResult(
            wound_mask=mask.astype(bool),
            confidence=confidence,
            backend=_BACKEND_NAME,
            weights_sha256=self._weights_sha256,
        )

    def _preprocess(self, pixels: np.ndarray, in_h: int, in_w: int) -> np.ndarray:
        """RGB float [0,1] (H,W,3) -> normalized NCHW batch of size 1."""
        resized = cv2.resize(pixels, (in_w, in_h), interpolation=cv2.INTER_LINEAR)
        resized = np.clip(resized, 0.0, 1.0).astype(np.float32)
        chw = np.transpose(resized, (2, 0, 1))  # (3, H, W)
        normalized = (chw - self._mean) / self._std
        return normalized[np.newaxis, ...].astype(np.float32)

    @staticmethod
    def _probabilities(out: np.ndarray) -> np.ndarray:
        """Reduce a raw model output to a single (H, W) wound-probability map.

        Supports single-logit outputs ``(1,1,H,W)`` / ``(1,H,W)`` / ``(H,W)`` via a
        sigmoid, and two-class outputs ``(1,2,H,W)`` / ``(2,H,W)`` via a softmax on
        the class axis (foreground = channel index 1).
        """
        arr = np.asarray(out, dtype=np.float32)
        # Drop a leading batch axis of size 1 (N, ...) -> (...).
        if arr.ndim == 4:
            if arr.shape[0] != 1:
                raise RuntimeError(f"unsupported batch size in output {arr.shape}")
            arr = arr[0]  # (C, H, W)
        if arr.ndim == 3:
            channels = arr.shape[0]
            if channels == 1:
                return _sigmoid(arr[0])
            if channels == 2:
                return _softmax(arr, axis=0)[1]
            raise RuntimeError(f"unsupported channel count in output {arr.shape}")
        if arr.ndim == 2:
            return _sigmoid(arr)
        raise RuntimeError(f"unsupported model output shape {arr.shape}")

    @staticmethod
    def _valid_mask(image: RasterImage) -> np.ndarray:
        """Opaque pixels are eligible to be wound; transparent ones are excluded."""
        if image.alpha is None:
            return np.ones(image.shape_hw, dtype=bool)
        return image.alpha > 0.5

    def _largest_component(self, mask: np.ndarray) -> np.ndarray:
        if not mask.any():
            return mask
        if mask.sum() < self._min_area_fraction * mask.size:
            # Too little signal to be a real wound — return it honestly and let the
            # confidence and staging engine decide.
            return mask
        labeled, count = ndimage.label(mask)
        if count <= 1:
            return mask
        sizes = ndimage.sum(np.ones_like(labeled), labeled, index=range(1, count + 1))
        keep = int(np.argmax(sizes)) + 1
        return labeled == keep

    @staticmethod
    def _confidence(prob: np.ndarray, mask: np.ndarray) -> float:
        """Mean predicted wound probability inside the mask (never a constant)."""
        selected = prob[mask]
        if selected.size == 0:
            return 0.0
        value = float(selected.mean())
        return max(0.0, min(1.0, value))

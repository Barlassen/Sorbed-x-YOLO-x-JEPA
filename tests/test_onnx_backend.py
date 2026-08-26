"""End-to-end test for the ONNX Runtime segmentation backend.

Builds a tiny but *real* ONNX graph (no torch required) that maps a
``(1, 3, H, W)`` input to a ``(1, 1, H, W)`` foreground-logit output, saves it,
and runs :class:`OnnxSegmenter` against a generated :class:`RasterImage`. It
asserts the backend returns a valid :class:`SegmentationResult`: a boolean mask of
the original shape, a confidence in ``[0, 1]``, and the model's SHA-256 recorded.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = pytest.mark.requires_ml

onnx = pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from onnx import TensorProto, helper  # noqa: E402

from sorbed.domain.image import Calibration, ImageMetadata  # noqa: E402
from sorbed.imaging import RasterImage, build_metadata  # noqa: E402
from sorbed.segmentation.backends.onnx_backend import OnnxSegmenter  # noqa: E402
from sorbed.segmentation.base import SegmentationResult  # noqa: E402

_H = 32
_W = 32


def _build_tiny_onnx(path, in_h: int, in_w: int) -> None:
    """A real graph: reduce RGB channels to one, then scale — a valid segmenter.

    ``logits = (mean_over_channels(input) - 0.5) * 8`` gives a per-pixel logit that
    is positive where the input is bright and negative where it is dark, so the
    downstream sigmoid+threshold produces a genuine, input-dependent mask.
    """
    input_t = helper.make_tensor_value_info(
        "input", TensorProto.FLOAT, [1, 3, in_h, in_w]
    )
    output_t = helper.make_tensor_value_info(
        "logits", TensorProto.FLOAT, [1, 1, in_h, in_w]
    )

    half = helper.make_tensor("half", TensorProto.FLOAT, [1], [0.5])
    scale = helper.make_tensor("scale", TensorProto.FLOAT, [1], [8.0])

    nodes = [
        # Opset 17: ReduceMean takes `axes` as an attribute (channel axis = 1).
        helper.make_node(
            "ReduceMean", ["input"], ["chan_mean"], axes=[1], keepdims=1
        ),
        helper.make_node("Sub", ["chan_mean", "half"], ["centered"]),
        helper.make_node("Mul", ["centered", "scale"], ["logits"]),
    ]
    graph = helper.make_graph(
        nodes,
        "tiny_segmenter",
        [input_t],
        [output_t],
        initializer=[half, scale],
    )
    model = helper.make_model(
        graph, opset_imports=[helper.make_opsetid("", 17)]
    )
    onnx.checker.check_model(model)
    onnx.save(model, str(path))


def _make_raster(height: int, width: int) -> RasterImage:
    rng = np.random.default_rng(7)
    pixels = np.full((height, width, 3), 0.15, dtype=np.float32)
    # A bright square that the tiny model should light up as "wound".
    pixels[height // 4 : height * 3 // 4, width // 4 : width * 3 // 4, :] = 0.9
    pixels += rng.normal(0.0, 0.01, size=pixels.shape).astype(np.float32)
    pixels = np.clip(pixels, 0.0, 1.0).astype(np.float32)

    metadata: ImageMetadata = build_metadata(
        pixels=pixels,
        source_format="png",
        bit_depth=8,
        channels=3,
    )
    calibration = Calibration()  # UNCALIBRATED; scale is irrelevant to segmentation
    return RasterImage(pixels=pixels, metadata=metadata, calibration=calibration)


def test_onnx_segmenter_end_to_end(tmp_path, monkeypatch):
    model_path = tmp_path / "tiny.onnx"
    _build_tiny_onnx(model_path, _H, _W)

    from sorbed.config.settings import get_settings

    monkeypatch.setenv("SORBED_ONNX_MODEL", str(model_path))
    monkeypatch.setenv("SORBED_SEGMENTATION_BACKEND", "onnx")
    settings = get_settings()

    segmenter = OnnxSegmenter.from_settings(settings)
    assert segmenter.name == "onnx"

    # Original image is larger than the model input to exercise resize-back.
    image = _make_raster(_H * 2, _W * 2)
    result = segmenter.segment(image)

    assert isinstance(result, SegmentationResult)
    assert result.wound_mask.dtype == np.bool_
    assert result.wound_mask.shape == image.shape_hw
    assert 0.0 <= result.confidence <= 1.0
    assert result.backend == "onnx"
    assert result.weights_sha256 and len(result.weights_sha256) == 64
    # The bright central square must be detected, and the mask must not be all-True.
    assert result.wound_mask.any()
    assert not result.wound_mask.all()
    assert result.confidence > 0.5


def test_from_settings_requires_model_env(monkeypatch):
    from sorbed.config.settings import get_settings

    monkeypatch.delenv("SORBED_ONNX_MODEL", raising=False)
    monkeypatch.setenv("SORBED_SEGMENTATION_BACKEND", "onnx")
    settings = get_settings()
    with pytest.raises(RuntimeError, match="SORBED_ONNX_MODEL"):
        OnnxSegmenter.from_settings(settings)

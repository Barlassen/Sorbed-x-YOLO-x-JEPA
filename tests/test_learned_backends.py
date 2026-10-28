"""Tests for the YOLO old-first wound mask and the CNN tissue backend.

Neither needs trained weights: the YOLO model is replaced by a stub returning fixed
instance masks, and the tissue CNN gets a randomly initialized network of the
trained architecture. They check the wiring, not accuracy (that is
training/check_product_pipeline.py).
"""

from __future__ import annotations

import importlib.util
from types import SimpleNamespace

import numpy as np
import pytest

from sorbed.config.settings import Settings
from sorbed.domain.enums import CalibrationStatus, TissueClass
from sorbed.domain.image import Calibration
from sorbed.imaging import RasterImage, build_metadata
from sorbed.pipeline.backends import build_tissue_classifier
from sorbed.segmentation.backends import yolo_backend
from sorbed.segmentation.backends.yolo_backend import YoloSegmenter
from sorbed.tissue.color_model import ColorTissueClassifier, tissue_index
from tests.synth import GRANULATION, SLOUGH, make_wound

_HAS_TORCHVISION = importlib.util.find_spec("torchvision") is not None
_H, _W = 64, 80


def _raster(rgb: np.ndarray) -> RasterImage:
    meta = build_metadata(pixels=rgb, source_format="png", bit_depth=8, channels=3)
    return RasterImage(pixels=rgb, metadata=meta, calibration=Calibration(
        status=CalibrationStatus.UNCALIBRATED))


class _Array:
    """The slice of the torch.Tensor API the backend touches."""

    def __init__(self, a) -> None:
        self._a = np.asarray(a)

    def cpu(self) -> _Array:
        return self

    def numpy(self) -> np.ndarray:
        return self._a

    def __len__(self) -> int:
        return len(self._a)


class _StubYolo:
    def __init__(self, masks: list[np.ndarray], scores: list[float]) -> None:
        self._masks, self._scores = masks, scores

    def predict(self, bgr, **_):
        if not self._masks:
            return [SimpleNamespace(masks=None, boxes=None)]
        return [SimpleNamespace(masks=SimpleNamespace(data=_Array(np.stack(self._masks))),
                                boxes=SimpleNamespace(conf=_Array(self._scores)))]


def _box(y0: int, y1: int, x0: int, x1: int) -> np.ndarray:
    m = np.zeros((_H, _W), np.float32)
    m[y0:y1, x0:x1] = 1.0
    return m


def test_fallback_model_runs_only_when_main_finds_nothing(monkeypatch):
    main, fallback = _box(5, 25, 5, 25), _box(30, 50, 40, 70)
    models = {"main": _StubYolo([], []), "fallback": _StubYolo([fallback], [0.6])}
    monkeypatch.setattr(yolo_backend, "_load_model", models.__getitem__)
    seg = YoloSegmenter("main", fallback_model_id="fallback", keep_largest=False)
    image = _raster(np.zeros((_H, _W, 3), np.float32))

    result = seg.segment(image)
    assert result.area_px == int(fallback.sum())
    assert result.confidence == pytest.approx(0.6)

    models["main"] = _StubYolo([main], [0.3])
    result = seg.segment(image)
    assert result.area_px == int(main.sum())
    assert result.confidence == pytest.approx(0.3)


def test_union_and_area_weighted_confidence(monkeypatch):
    big, small = _box(5, 25, 5, 25), _box(40, 50, 60, 70)  # 400 px at 0.8, 100 px at 0.2
    monkeypatch.setattr(yolo_backend, "_load_model", lambda _: _StubYolo([big, small], [0.8, 0.2]))
    image = _raster(np.zeros((_H, _W, 3), np.float32))

    union = YoloSegmenter("m", keep_largest=False).segment(image)
    assert union.area_px == 500
    assert union.confidence == pytest.approx((400 * 0.8 + 100 * 0.2) / 500)
    assert YoloSegmenter("m", keep_largest=True).segment(image).area_px == 400


def test_yolo_settings_reach_the_segmenter():
    seg = YoloSegmenter.from_settings(Settings(
        yolo_seg_model="a.pt", yolo_seg_fallback_model="b.pt", yolo_seg_confidence=0.01,
        yolo_seg_keep_largest=False))
    assert (seg._model_id, seg._fallback_model_id, seg._confidence_floor, seg._keep_largest) == (
        "a.pt", "b.pt", 0.01, False)
    assert YoloSegmenter.from_settings(Settings())._fallback_model_id is None


def test_cnn_backend_without_weights_fails_clearly():
    with pytest.raises(RuntimeError, match="SORBED_TISSUE_CNN_WEIGHTS"):
        build_tissue_classifier(Settings(tissue_backend="cnn", tissue_cnn_weights="missing.pt"))


@pytest.mark.skipif(not _HAS_TORCHVISION, reason="needs torch + torchvision (the [ml] extra)")
def test_cnn_backend_labels_only_the_wound(tmp_path):
    import torch
    from torchvision.models.segmentation import deeplabv3_mobilenet_v3_large

    torch.manual_seed(0)
    net = deeplabv3_mobilenet_v3_large(weights=None, weights_backbone=None, aux_loss=None)
    net.classifier[-1] = torch.nn.Conv2d(net.classifier[-1].in_channels, 4, 1)
    weights = tmp_path / "tissue.pt"
    torch.save(net.state_dict(), weights)
    clf = build_tissue_classifier(Settings(tissue_backend="cnn", tissue_cnn_weights=str(weights)))
    wound = make_wound(fill=GRANULATION, inner=((28, 20), SLOUGH), seed=3)

    t = clf.classify(wound.rgb, wound.wound_mask, raw_rgb=wound.rgb)
    color = ColorTissueClassifier().classify(wound.rgb, wound.wound_mask)
    assert clf.name == "cnn" and len(clf.weights_sha256) == 64
    assert sum(t.composition.fractions.values()) == pytest.approx(1.0)
    assert set(t.composition.fractions) <= {TissueClass.GRANULATION, TissueClass.SLOUGH,
                                            TissueClass.ESCHAR, TissueClass.EPITHELIAL}
    assert (t.label_map[~wound.wound_mask] == tissue_index(TissueClass.BACKGROUND)).all()
    assert (t.label_map[wound.wound_mask] != tissue_index(TissueClass.BACKGROUND)).all()
    assert (t.maroon_purple_fraction, t.erythema_fraction) == (
        color.maroon_purple_fraction, color.erythema_fraction)

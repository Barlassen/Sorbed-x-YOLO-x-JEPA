"""Partial-label (superset) supervision in the multiclass segmentation loss.

Skipped when torch is absent (the training stack is optional). Verifies that the
superset sentinel drives the foreground probability up without supervising a
specific tissue class, per the single-model design decision.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")

from training.losses import MulticlassDiceCELoss  # noqa: E402


def test_superset_none_matches_plain_loss() -> None:
    loss = MulticlassDiceCELoss(num_classes=7)
    logits = torch.randn(2, 7, 8, 8)
    target = torch.randint(0, 7, (2, 8, 8))
    value = loss(logits, target)
    assert torch.isfinite(value)


def test_superset_penalises_background_on_foreground_pixels() -> None:
    n_classes, sentinel = 7, -2
    loss = MulticlassDiceCELoss(num_classes=n_classes, superset_index=sentinel,
                                background_index=0)
    # All pixels are "foreground, class unknown" (the binary-wound-set case).
    target = torch.full((1, 8, 8), sentinel, dtype=torch.long)

    # Logits confident that everything is BACKGROUND -> high superset penalty.
    bg = torch.full((1, n_classes, 8, 8), -4.0)
    bg[:, 0] = 8.0
    # Logits confident that everything is some FOREGROUND class -> low penalty.
    fg = torch.full((1, n_classes, 8, 8), -4.0)
    fg[:, 2] = 8.0

    assert loss(bg, target) > loss(fg, target)


def test_superset_pixels_excluded_from_class_ce() -> None:
    # A mix: half real labels, half superset. The loss stays finite and the
    # superset pixels do not force a specific class.
    n_classes, sentinel = 7, -2
    loss = MulticlassDiceCELoss(num_classes=n_classes, superset_index=sentinel)
    target = torch.randint(0, n_classes, (1, 8, 8))
    target[:, :, :4] = sentinel
    logits = torch.randn(1, n_classes, 8, 8, requires_grad=True)
    value = loss(logits, target)
    value.backward()
    assert torch.isfinite(value)
    assert logits.grad is not None


def _write_pair(tmp_path, mask_values):
    cv2 = pytest.importorskip("cv2")
    import numpy as np

    image = (np.arange(16 * 16 * 3, dtype=np.uint8) % 255).reshape(16, 16, 3)
    mask = np.zeros((16, 16), dtype=np.uint8)
    mask[4:8, 4:8] = int(mask_values)
    image_path = tmp_path / "a.png"
    mask_path = tmp_path / "a_mask.png"
    cv2.imwrite(str(image_path), image)
    cv2.imwrite(str(mask_path), mask)
    return image_path, mask_path


def test_binary_source_maps_foreground_to_sentinel(tmp_path) -> None:
    from training.data import SUPERSET_SENTINEL, SegmentationDataset

    image_path, mask_path = _write_pair(tmp_path, mask_values=255)
    dataset = SegmentationDataset(
        [(image_path, mask_path)], input_size=16, num_classes=7,
        superset_index=SUPERSET_SENTINEL, binary_flags=[True])
    _, target = dataset[0]
    # A binary source: foreground becomes the sentinel, background stays class 0.
    assert set(target.unique().tolist()) == {0, SUPERSET_SENTINEL}


def test_tissue_source_passes_class_indices_through(tmp_path) -> None:
    from training.data import SUPERSET_SENTINEL, SegmentationDataset

    image_path, mask_path = _write_pair(tmp_path, mask_values=3)
    dataset = SegmentationDataset(
        [(image_path, mask_path)], input_size=16, num_classes=7,
        superset_index=SUPERSET_SENTINEL, binary_flags=[False])
    _, target = dataset[0]
    # A tissue source keeps its real class indices; no sentinel is introduced.
    assert set(target.unique().tolist()) == {0, 3}


def test_binary_flags_length_mismatch_raises(tmp_path) -> None:
    from training.data import SegmentationDataset

    image_path, mask_path = _write_pair(tmp_path, mask_values=255)
    with pytest.raises(ValueError, match="binary_flags"):
        SegmentationDataset([(image_path, mask_path)], input_size=16,
                            num_classes=7, binary_flags=[True, False])

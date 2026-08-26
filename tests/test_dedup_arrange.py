"""Perceptual-hash dedup and tissue-mask arrange tools.

Skipped when Pillow/opencv/torch-free core deps are absent. These exercise the
real image paths on tiny synthetic fixtures.
"""

from __future__ import annotations

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")
PIL = pytest.importorskip("PIL")

from training.datasets import ManifestRecord  # noqa: E402


def _write_img(path, array) -> None:
    cv2.imwrite(str(path), array)


def test_dhash_identical_images_zero_distance(tmp_path) -> None:
    from training.dedup import dhash, hamming

    arr = (np.arange(64 * 64, dtype=np.uint8) % 255).reshape(64, 64)
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    _write_img(a, arr)
    _write_img(b, arr.copy())
    ha, hb = dhash(a), dhash(b)
    assert ha is not None and hb is not None
    assert hamming(ha, hb) == 0


def test_dhash_distinct_images_large_distance(tmp_path) -> None:
    from training.dedup import dhash, hamming

    left = np.zeros((64, 64), np.uint8)
    left[:, :32] = 255  # bright left half
    right = np.zeros((64, 64), np.uint8)
    right[:32, :] = 255  # bright top half — different gradient structure
    a = tmp_path / "a.png"
    b = tmp_path / "b.png"
    _write_img(a, left)
    _write_img(b, right)
    assert hamming(dhash(a), dhash(b)) > 6


def test_deduplicate_keeps_priority_source(tmp_path) -> None:
    from training.dedup import deduplicate

    arr = (np.arange(64 * 64, dtype=np.uint8) % 255).reshape(64, 64)
    p1 = tmp_path / "dup_a.png"
    p2 = tmp_path / "dup_b.png"
    p3 = tmp_path / "unique.png"
    _write_img(p1, arr)
    _write_img(p2, arr.copy())
    other = arr.copy()
    other[:32] = 0
    _write_img(p3, other)
    records = [
        ManifestRecord(str(p1), "", "mixed", "", "azh:1", 0, "azh", "x"),
        ManifestRecord(str(p2), "", "mixed", "", "wsnet:1", 0, "wsnet", "x"),
        ManifestRecord(str(p3), "", "mixed", "", "azh:2", 0, "azh", "x"),
    ]
    kept, report = deduplicate(records, hamming_radius=6, source_priority=["azh", "wsnet"])
    kept_paths = {r.image_path for r in kept}
    assert str(p3) in kept_paths          # the unique image survives
    assert str(p1) in kept_paths          # azh preferred over wsnet duplicate
    assert str(p2) not in kept_paths      # the duplicate is dropped
    assert report["dropped"] == 1


def test_arrange_remaps_grayscale_to_unified(tmp_path) -> None:
    from training.arrange_tissue import arrange

    images = tmp_path / "src" / "images"
    masks = tmp_path / "src" / "masks"
    images.mkdir(parents=True)
    masks.mkdir(parents=True)
    _write_img(images / "w1.png", (np.random.rand(16, 16, 3) * 255).astype(np.uint8))
    native = np.zeros((16, 16), np.uint8)
    native[:8, :8] = 1  # granulation in the map below
    native[:8, 8:] = 4  # eschar
    _write_img(masks / "w1.png", native)

    mapping = tmp_path / "map.yaml"
    mapping.write_text(
        "mode: grayscale\nunmapped: error\nmap:\n  0: background\n"
        "  1: granulation\n  4: eschar\n",
        encoding="utf-8",
    )
    out = tmp_path / "out"
    report = arrange(images, masks, mapping, out)
    assert report["pairs"] == 1
    remapped = cv2.imread(str(out / "masks" / "w1.png"), cv2.IMREAD_GRAYSCALE)
    # granulation -> index 2, eschar -> index 4, background -> 0.
    assert set(np.unique(remapped).tolist()) == {0, 2, 4}


def test_arrange_errors_on_unmapped_value(tmp_path) -> None:
    from training.arrange_tissue import MappingError, arrange

    images = tmp_path / "src" / "images"
    masks = tmp_path / "src" / "masks"
    images.mkdir(parents=True)
    masks.mkdir(parents=True)
    _write_img(images / "w1.png", (np.random.rand(16, 16, 3) * 255).astype(np.uint8))
    native = np.zeros((16, 16), np.uint8)
    native[:8, :8] = 9  # not in the mapping
    _write_img(masks / "w1.png", native)
    mapping = tmp_path / "map.yaml"
    mapping.write_text("mode: grayscale\nmap:\n  0: background\n", encoding="utf-8")
    with pytest.raises(MappingError, match="no mapping"):
        arrange(images, masks, mapping, tmp_path / "out")

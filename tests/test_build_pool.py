"""Synthetic tests only: no patient data. Exercises the JEPA pool builder's
de-duplication, test-set leakage exclusion, symlink naming, and symlink-skipping
scan — the data rules the honest evaluation depends on."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
from PIL import Image

from training.build_pool import build, scan_images, write_pool


def _save(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8)).save(path)


def _distinct(seed: int) -> np.ndarray:
    """A 32x32 RGB image with a content-dependent (so hash-distinct) pattern."""
    rng = np.random.default_rng(seed)
    return rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)


class BuildPoolTests(unittest.TestCase):
    def _src(self, root: Path, tag: str, rank: int = 0):
        return (root, tag, rank)

    def test_exact_duplicate_across_sources_kept_once(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            img = _distinct(1)
            _save(tmp / "a" / "same.png", img)
            _save(tmp / "b" / "same_copy.png", img)  # byte-identical content
            _save(tmp / "b" / "other.png", _distinct(2))
            kept, report = build(
                [self._src(tmp / "a", "a"), self._src(tmp / "b", "b")],
                [],
                hamming_radius=2, leak_hamming=2,
                follow_symlinks=False, ignore_dirnames=set(),
            )
            self.assertEqual(report["scanned_keep"], 3)
            self.assertEqual(report["dropped_near_duplicates"], 1)
            self.assertEqual(report["kept"], 2)

    def test_priority_source_wins_the_duplicate(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            img = _distinct(3)
            _save(tmp / "low" / "x.png", img)
            _save(tmp / "high" / "x.png", img)
            kept, report = build(
                [self._src(tmp / "low", "low", 1), self._src(tmp / "high", "high", 0)],
                [],
                hamming_radius=2, leak_hamming=2,
                follow_symlinks=False, ignore_dirnames=set(),
            )
            self.assertEqual(report["kept"], 1)
            self.assertEqual(kept[0].tag, "high")

    def test_leakage_copy_is_dropped_whole(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            test_img = _distinct(4)
            _save(tmp / "test" / "t.png", test_img)
            _save(tmp / "train" / "leak.png", test_img)   # same as a test photo
            _save(tmp / "train" / "clean.png", _distinct(5))
            kept, report = build(
                [self._src(tmp / "train", "train")],
                [tmp / "test"],
                hamming_radius=2, leak_hamming=2,
                follow_symlinks=False, ignore_dirnames=set(),
            )
            self.assertEqual(report["leakage_dropped"], 1)
            self.assertEqual(report["kept"], 1)
            self.assertNotIn("leak", {k.path.stem for k in kept})

    def test_distinct_images_all_survive_with_unique_links(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for i in range(5):
                _save(tmp / "s" / f"img_{i}.png", _distinct(100 + i))
            kept, report = build(
                [self._src(tmp / "s", "s")], [],
                hamming_radius=2, leak_hamming=2,
                follow_symlinks=False, ignore_dirnames=set(),
            )
            self.assertEqual(report["kept"], 5)
            images = write_pool(kept, tmp / "pool", dry_run=False)
            links = sorted(images.iterdir())
            self.assertEqual(len(links), 5)
            self.assertEqual(len({p.name for p in links}), 5)  # unique names
            self.assertTrue(all(p.is_symlink() for p in links))

    def test_scan_skips_symlinks_and_ignored_dirs(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _save(tmp / "images" / "real.png", _distinct(7))
            _save(tmp / "labels" / "mask.png", _distinct(8))       # ignored dir
            (tmp / "images" / "link.png").symlink_to(tmp / "images" / "real.png")
            found = scan_images(tmp, follow_symlinks=False, ignore_dirnames={"labels"})
            names = {p.name for p in found}
            self.assertIn("real.png", names)
            self.assertNotIn("mask.png", names)   # ignored dir skipped
            self.assertNotIn("link.png", names)   # symlink skipped


if __name__ == "__main__":
    unittest.main()

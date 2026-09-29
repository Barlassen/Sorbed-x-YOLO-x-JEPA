"""Synthetic tests only: no patient data. Covers the stage probe's input building
(must mirror JEPA pre-training), split listing, the checkpoint-load guard, and the
paired bootstrap."""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from training.train_jepa import Encoder, JepaConfig
from training.validate_stage_probe import (
    build_encoder,
    list_split,
    paired_delta,
    prepare_image,
)


def _save(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array.astype(np.uint8)).save(path)


class StageProbeTests(unittest.TestCase):
    def test_list_split_reads_stage_folders(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for stage in (1, 3):
                _save(root / "train" / str(stage) / f"a{stage}.png", np.zeros((8, 8, 3)))
            _save(root / "train" / "notes" / "x.png", np.zeros((8, 8, 3)))  # non-stage dir ignored
            items = list_split(root, "train")
            self.assertEqual(sorted(s for _, s in items), [1, 3])

    def test_prepare_crops_to_mask_and_stacks_depth_relief(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _save(tmp / "img" / "w.png", np.random.default_rng(0).integers(0, 255, (64, 96, 3)))
            mask = np.zeros((64, 96), np.uint8); mask[20:30, 40:50] = 255
            _save(tmp / "masks" / "w.png", mask)
            _save(tmp / "depth" / "w.png", np.tile(np.arange(96, dtype=np.uint8), (64, 1)))
            cfg = JepaConfig(img_size=32, in_chans=5)
            t, cropped = prepare_image(tmp / "img" / "w.png", cfg, tmp / "masks", tmp / "depth",
                                       crop=True, relief=True)
            self.assertTrue(cropped)
            self.assertEqual(tuple(t.shape), (5, 32, 32))

    def test_empty_mask_falls_back_to_full_frame(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            _save(tmp / "img" / "e.png", np.full((40, 40, 3), 128))
            _save(tmp / "masks" / "e.png", np.zeros((40, 40)))
            cfg = JepaConfig(img_size=32, in_chans=3)
            t, cropped = prepare_image(tmp / "img" / "e.png", cfg, tmp / "masks", None,
                                       crop=True, relief=False)
            self.assertFalse(cropped)
            self.assertEqual(tuple(t.shape), (3, 32, 32))

    def test_checkpoint_guard_rejects_mismatched_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            bad = Path(tmp) / "bad.pt"
            torch.save({"context_encoder.whatever": torch.zeros(1)}, bad)
            cfg = JepaConfig(img_size=32, in_chans=3)
            with self.assertRaises(SystemExit):
                build_encoder(cfg, "cpu", init="jepa", weights=str(bad), seed=0)

    def test_checkpoint_guard_accepts_real_encoder_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            good = Path(tmp) / "good.pt"
            cfg = JepaConfig(img_size=32, in_chans=3)
            torch.save(Encoder(cfg).state_dict(), good)
            enc = build_encoder(cfg, "cpu", init="jepa", weights=str(good), seed=0)
            self.assertFalse(enc.training)

    def test_pooling_changes_feature_width(self):
        from training.validate_stage_probe import encode

        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            for i in range(2):
                _save(tmp / "img" / f"p{i}.png", np.random.default_rng(i).integers(0, 255, (40, 40, 3)))
            cfg = JepaConfig(img_size=32, in_chans=3)
            enc = build_encoder(cfg, "cpu", init="random", weights=None, seed=0)
            items = [(tmp / "img" / f"p{i}.png", 1) for i in range(2)]
            dims = {}
            for pool in ("mean", "meanstd", "max"):
                X, _, _ = encode(enc, items, cfg, "cpu", None, None, False, False, pool)
                dims[pool] = X.shape[1]
            self.assertEqual(dims["meanstd"], 2 * dims["mean"])
            self.assertEqual(dims["max"], dims["mean"])

    def test_paired_delta_is_positive_when_a_beats_b(self):
        y = np.array([1, 2, 3, 4] * 10)
        perfect = y.copy()
        wrong = np.roll(y, 1)
        d = paired_delta(y, perfect, wrong, [1, 2, 3, 4], n_boot=200, seed=0)
        self.assertGreater(d["macro_f1"]["lo"], 0.0)
        self.assertEqual(d["qwk"]["p_gt0"], 1.0)


if __name__ == "__main__":
    unittest.main()

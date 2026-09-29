"""Synthetic tests only: no patient data, real reviews, or clinical accuracy claims."""
import hashlib
import json
import tempfile
import unittest
from argparse import Namespace
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from training.evaluate_surface_seg import evaluate
from training.predict_surface_seg import predict
from training.surface_labels import aggregate_counts, build_target, require_review, surface_counts
from training.train_surface_seg import (
    SurfaceDataset, SurfaceNet, run, select_reviewed_splits, surface_loss,
)


class SurfaceTests(unittest.TestCase):
    def test_lesion_remainder_is_not_intact(self):
        lesion = np.ones((3, 3), dtype=bool)
        open_region = np.eye(3, dtype=bool)
        target = build_target(lesion, [(2, open_region)])
        self.assertTrue(np.all(target[~open_region] == 255))
        self.assertFalse(np.any(target == 1))

    def test_conflicting_labels_rejected(self):
        region = np.ones((2, 2), dtype=bool)
        with self.assertRaises(ValueError):
            build_target(region, [(1, region), (2, region)])
        self.assertTrue(np.all(build_target(region, [(3, region), (3, region)]) == 3))

    def test_unknown_reference_is_ignored_but_prediction_abstention_is_not(self):
        reference = np.array([[2, 2, 255]], dtype=np.uint8)
        pred = np.array([[2, 255, 1]], dtype=np.uint8)
        scores = surface_counts(pred, reference)
        self.assertAlmostEqual(scores['classes']['open_surface']['dice'], 2 / 3)
        self.assertEqual(scores['classes']['intact_lesion']['fp'], 0)
        self.assertEqual(scores['abstained_known_pixels'], 1)

    def test_empty_pairs_do_not_inflate_dice(self):
        empty = np.zeros((2, 2), dtype=np.uint8)
        scores = surface_counts(empty, empty)
        self.assertIsNone(scores['classes']['covered_surface']['dice'])
        self.assertIsNone(aggregate_counts([scores])['covered_surface']['pooled_dice'])
        pred = empty.copy()
        pred[0, 0] = 3
        self.assertEqual(surface_counts(pred, empty)['classes']['covered_surface']['dice'], 0)

    def test_mask_shape_and_schema_mismatch_rejected(self):
        with self.assertRaises(ValueError):
            surface_counts(np.zeros((2, 2), dtype=np.uint8), np.zeros((3, 3), dtype=np.uint8))
        with self.assertRaises(ValueError):
            surface_counts(np.array([[9]], dtype=np.uint8), np.array([[0]], dtype=np.uint8))

    def test_unreviewed_labels_rejected(self):
        for purpose in ['training', 'evaluation']:
            with self.assertRaises(ValueError):
                require_review({'image_id': 'synthetic', 'eligible_for_training': True,
                                'eligible_for_ground_truth': True}, purpose)

    def test_no_unknown_gradients_and_finite_all_ignore(self):
        logits = torch.randn(1, 4, 2, 2, requires_grad=True)
        y = torch.tensor([[[2, 255], [0, 255]]])
        surface_loss(logits, y).backward()
        self.assertTrue(torch.all(logits.grad[:, :, :, 1] == 0))
        z = torch.randn(1, 4, 2, 2, requires_grad=True)
        loss = surface_loss(z, torch.full((1, 2, 2), 255))
        self.assertEqual(float(loss.detach()), 0)
        loss.backward()
        self.assertTrue(torch.all(z.grad == 0))

    def test_single_network_handles_mixed_surface_targets_and_odd_geometry(self):
        net = SurfaceNet()
        output = net(torch.rand(1, 3, 33, 35))
        self.assertEqual(tuple(output.shape), (1, 4, 33, 35))
        target = torch.arange(33 * 35).reshape(1, 33, 35) % 4
        surface_loss(output, target).backward()
        self.assertIsNotNone(net.enc1[0].weight.grad)

    @staticmethod
    def record(name, split, group, digest):
        return {'schema_version': 'surface-2.0', 'image_id': name, 'case_id': name,
                'split': split, 'patient_group_id': group, 'source_sha256': digest,
                'surface_mask_sha256': 'synthetic', 'review_status': 'clinician_reviewed',
                'eligible_for_training': True, 'eligible_for_ground_truth': True,
                'reviewer': {'id': 'SYNTHETIC_TEST_ONLY', 'date': '2000-01-01'}}

    def test_patient_and_duplicate_leakage_rejected(self):
        a = self.record('a', 'train', 'p1', 'hash-a')
        for b in [self.record('b', 'val', 'p1', 'hash-b'),
                  self.record('b', 'val', 'p2', 'hash-a')]:
            with self.assertRaisesRegex(ValueError, 'leakage'):
                select_reviewed_splits([a, b])

    def test_unassigned_groups_rejected(self):
        with self.assertRaises(ValueError):
            select_reviewed_splits([{'split': 'unassigned'}])

    def test_synthetic_train_predict_evaluate_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            records_dir = root / 'records'
            records_dir.mkdir()
            records = []
            for i, split in enumerate(['train', 'val', 'test']):
                rgb = np.zeros((40, 48, 3), dtype=np.uint8)
                rgb[:, :, 0] = 30 + i * 40
                rgb[10:30, 10:30, 1] = 150
                mask = np.zeros((40, 48), dtype=np.uint8)
                mask[4:15, 4:15] = 1
                mask[16:28, 4:20] = 2
                mask[10:30, 28:40] = 3
                mask[:3] = 255
                image_path, mask_path = root / f'image{i}.png', root / f'mask{i}.png'
                Image.fromarray(rgb).save(image_path)
                Image.fromarray(mask).save(mask_path)
                r = self.record(f'synthetic{i}', split, f'group{i}',
                                hashlib.sha256(image_path.read_bytes()).hexdigest())
                r.update(source_path=str(image_path), surface_mask_path=str(mask_path),
                         surface_mask_sha256=hashlib.sha256(mask_path.read_bytes()).hexdigest())
                (records_dir / f'{i}.json').write_text(json.dumps(r))
                records.append(r)
            ds = SurfaceDataset(records[:1], 32, 'training')
            self.assertEqual(set(torch.unique(ds[0][1]).tolist()), {0, 1, 2, 3, 255})
            run(Namespace(records=records_dir, out=root / 'run', epochs=1, batch=1,
                          size=32, seed=0, device='cpu', lr=1e-3))
            count = predict(root / 'run/best.pt', records_dir, root / 'pred')
            self.assertEqual(count, 1)
            with Image.open(root / 'pred/synthetic2.png') as im:
                self.assertEqual(im.size, (48, 40))
            report = evaluate(records_dir, root / 'pred', root / 'pred/surface_classes.json')
            self.assertEqual(report['images'], 1)
            self.assertEqual(report['reference_kind'], 'reviewed_reference')
            self.assertEqual(set(report['classes']), {'intact_lesion', 'open_surface', 'covered_surface'})


if __name__ == '__main__':
    torch.set_num_threads(2)
    unittest.main()

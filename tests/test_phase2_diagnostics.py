"""
test_phase2_diagnostics.py

Comprehensive Unit Tests for Phase 2:
1. analyze_pseudolabels.py:
   - Split file parsing (8,584 samples, hidden ground-truth integrity)
   - Mathematical precision, coverage, and N_accepted computation
   - Corner cases: 0% accepted, 100% accepted, all correct, all incorrect
   - Synthetic distribution generation & probability normalization
   - End-to-end report generation (JSON, CSV, Plot)
2. recalibrate_bn.py:
   - Detection of all BatchNorm layers (BatchNorm3d)
   - Parameter immutability / bitwise invariance verification
   - Mathematical verification of cumulative sample averaging (momentum=None)
   - End-to-end recalibration protocol execution and checkpoint persistence
"""

import os
import sys
import json
import csv
import tempfile
import unittest
import numpy as np
import torch
import torch.nn as nn

from models.resnet3d import ResNet3D18, F2FixMatchModel
from analyze_pseudolabels import (
    parse_unlabeled_split,
    compute_pseudolabel_metrics,
    evaluate_synthetic_unlabeled,
    save_csv_report,
    save_plots
)
from recalibrate_bn import (
    find_bn_modules,
    snapshot_model_parameters,
    verify_parameter_invariance,
    recalibrate_bn_statistics,
    save_recalibrated_checkpoint
)


class TestAnalyzePseudolabels(unittest.TestCase):
    def test_parse_real_split_file(self):
        split_file = "data/splits/ucf101_train_10_percent_unlabeled_split_1_rawframes.txt"
        self.assertTrue(os.path.exists(split_file), f"Split file {split_file} must exist.")
        samples = parse_unlabeled_split(split_file)
        self.assertEqual(len(samples), 8584, "Split must contain exactly 8,584 unlabeled samples.")
        
        # Verify sample structure and ground-truth range [0, 100]
        for s in samples[:10]:
            self.assertIn("rel_path", s)
            self.assertIn("total_frames", s)
            self.assertIn("gt_label", s)
            self.assertGreaterEqual(s["gt_label"], 0)
            self.assertLessEqual(s["gt_label"], 100)

    def test_compute_pseudolabel_metrics_all_correct(self):
        # 10 samples, 5 classes, all perfectly predicted with 0.95 confidence
        num_samples = 10
        num_classes = 5
        probs = np.zeros((num_samples, num_classes), dtype=np.float32)
        gt_labels = np.array([i % num_classes for i in range(num_samples)], dtype=np.int64)
        for i in range(num_samples):
            probs[i, gt_labels[i]] = 0.95
            probs[i, (gt_labels[i] + 1) % num_classes] = 0.05

        tau_values = [0.1, 0.5, 0.9, 0.98]
        metrics = compute_pseudolabel_metrics(probs, gt_labels, tau_values)
        sweeps = metrics["tau_sweep"]

        # At tau=0.5: all 10 accepted, 100% precision, 100% coverage
        m_05 = next(m for m in sweeps if m["tau"] == 0.5)
        self.assertEqual(m_05["n_accepted"], 10)
        self.assertEqual(m_05["coverage_percent"], 100.0)
        self.assertEqual(m_05["n_correct"], 10)
        self.assertEqual(m_05["precision_percent"], 100.0)

        # At tau=0.98: 0 accepted, 0% coverage, 0% precision
        m_098 = next(m for m in sweeps if m["tau"] == 0.98)
        self.assertEqual(m_098["n_accepted"], 0)
        self.assertEqual(m_098["coverage_percent"], 0.0)
        self.assertEqual(m_098["n_correct"], 0)
        self.assertEqual(m_098["precision_percent"], 0.0)

    def test_compute_pseudolabel_metrics_mixed(self):
        # 4 samples:
        # sample 0: max_prob=0.85, pred=0, gt=0 (correct)
        # sample 1: max_prob=0.80, pred=1, gt=2 (incorrect)
        # sample 2: max_prob=0.50, pred=2, gt=2 (correct)
        # sample 3: max_prob=0.20, pred=3, gt=3 (correct)
        probs = np.array([
            [0.85, 0.05, 0.05, 0.05],
            [0.05, 0.80, 0.10, 0.05],
            [0.10, 0.10, 0.50, 0.30],
            [0.20, 0.20, 0.40, 0.20]  # pred is class 2 with 0.40, gt is 3
        ], dtype=np.float32)
        gt_labels = np.array([0, 2, 2, 3], dtype=np.int64)

        metrics = compute_pseudolabel_metrics(probs, gt_labels, [0.3, 0.7])
        sweeps = metrics["tau_sweep"]

        # At tau=0.7: samples 0 and 1 accepted (2/4 = 50% coverage).
        # sample 0 correct, sample 1 incorrect -> 1/2 = 50% precision.
        m_07 = next(m for m in sweeps if m["tau"] == 0.7)
        self.assertEqual(m_07["n_accepted"], 2)
        self.assertEqual(m_07["coverage_percent"], 50.0)
        self.assertEqual(m_07["n_correct"], 1)
        self.assertEqual(m_07["precision_percent"], 50.0)

    def test_compute_pseudolabel_metrics_empty_inputs(self):
        probs = np.zeros((0, 101), dtype=np.float32)
        gt_labels = np.array([], dtype=np.int64)
        metrics = compute_pseudolabel_metrics(probs, gt_labels, [0.1, 0.3, 0.5])
        self.assertEqual(metrics["num_samples"], 0)
        self.assertFalse(metrics["has_ground_truth"])
        self.assertEqual(len(metrics["tau_sweep"]), 0)
        self.assertEqual(metrics["confidence_distribution"]["mean"], 0.0)

    def test_compute_pseudolabel_metrics_no_ground_truth(self):
        # Samples where all ground-truth labels are -1 (unlabeled split without GT)
        num_samples = 5
        probs = np.full((num_samples, 10), 0.1, dtype=np.float32)
        probs[:, 0] = 0.8  # confident prediction
        gt_labels = np.full(num_samples, -1, dtype=np.int64)

        metrics = compute_pseudolabel_metrics(probs, gt_labels, [0.3, 0.7])
        self.assertFalse(metrics["has_ground_truth"])
        sweeps = metrics["tau_sweep"]
        for s in sweeps:
            self.assertIsNone(s["precision"])
            self.assertIsNone(s["precision_percent"])

        # Verify CSV export outputs "N/A" rather than "0.00"
        with tempfile.TemporaryDirectory() as td:
            csv_path = os.path.join(td, "no_gt.csv")
            save_csv_report(sweeps, csv_path)
            with open(csv_path, "r") as f:
                content = f.read()
                self.assertIn("N/A", content)

    def test_synthetic_unlabeled_generation(self):
        samples = [{"sample_index": i, "rel_path": f"cls/v_{i}", "total_frames": 100, "gt_label": i % 101} for i in range(100)]
        probs, gt = evaluate_synthetic_unlabeled(model=None, samples=samples, device=torch.device("cpu"), seed=123)
        self.assertEqual(probs.shape, (100, 101))
        self.assertEqual(len(gt), 100)

        # Verify probabilities sum to 1.0 per sample
        sums = np.sum(probs, axis=1)
        np.testing.assert_allclose(sums, np.ones(100), atol=1e-5)

        # Verify max probabilities span valid interval
        max_p = np.max(probs, axis=1)
        self.assertTrue(np.all(max_p >= 0.10))
        self.assertTrue(np.all(max_p <= 1.0))

    def test_full_pipeline_artifacts(self):
        with tempfile.TemporaryDirectory() as td:
            out_json = os.path.join(td, "test_metrics.json")
            out_csv = os.path.join(td, "test_metrics.csv")
            out_plot = os.path.join(td, "test_plot.png")

            probs = np.random.dirichlet(np.ones(101), size=50)
            gt = np.random.randint(0, 101, size=50)
            metrics = compute_pseudolabel_metrics(probs, gt, [0.1, 0.3, 0.5, 0.8])

            with open(out_json, "w") as f:
                json.dump(metrics, f, indent=2)
            save_csv_report(metrics["tau_sweep"], out_csv)
            save_plots(metrics, out_plot)

            self.assertTrue(os.path.exists(out_json))
            self.assertTrue(os.path.exists(out_csv))
            self.assertTrue(os.path.exists(out_plot))
            self.assertGreater(os.path.getsize(out_plot), 1000)

            with open(out_csv, "r") as f:
                reader = csv.reader(f)
                header = next(reader)
                self.assertIn("coverage_percent", header)
                self.assertIn("precision_percent", header)


class TestRecalibrateBN(unittest.TestCase):
    def test_find_bn_modules_count(self):
        model_f2 = F2FixMatchModel(num_classes=101)
        bn_f2 = find_bn_modules(model_f2)
        # ResNet-18 has exactly 20 BatchNorm3d modules in the backbone
        self.assertEqual(len(bn_f2), 20)

        model_resnet = ResNet3D18(num_classes=101)
        bn_resnet = find_bn_modules(model_resnet)
        self.assertEqual(len(bn_resnet), 20)

    def test_parameter_invariance_detection(self):
        model = ResNet3D18(num_classes=10)
        snapshot = snapshot_model_parameters(model)

        # Unmodified model passes invariance check
        is_inv, count, msg = verify_parameter_invariance(model, snapshot)
        self.assertTrue(is_inv)
        self.assertIn("All parameters bitwise identical", msg)

        # Mutating a single parameter weight triggers immediate failure
        with torch.no_grad():
            model.fc.weight[0, 0] += 0.01
        is_inv_mut, _, msg_mut = verify_parameter_invariance(model, snapshot)
        self.assertFalse(is_inv_mut)
        self.assertIn("Parameter fc.weight modified", msg_mut)

    def test_exact_cumulative_moving_average_math(self):
        # Test that momentum=None computes exact cumulative average across batches
        bn = nn.BatchNorm3d(3, momentum=None)
        bn.train()

        b1 = torch.full((2, 3, 2, 2, 2), 2.0)
        b2 = torch.full((2, 3, 2, 2, 2), 6.0)

        _ = bn(b1)
        self.assertEqual(bn.num_batches_tracked.item(), 1)
        np.testing.assert_allclose(bn.running_mean.numpy(), [2.0, 2.0, 2.0], atol=1e-5)

        _ = bn(b2)
        self.assertEqual(bn.num_batches_tracked.item(), 2)
        # (2.0 + 6.0) / 2 = 4.0
        np.testing.assert_allclose(bn.running_mean.numpy(), [4.0, 4.0, 4.0], atol=1e-5)

    def test_recalibrate_bn_statistics_synthetic(self):
        model = F2FixMatchModel(num_classes=10)
        device = torch.device("cpu")
        summary = recalibrate_bn_statistics(
            model=model,
            device=device,
            is_synthetic=True,
            num_batches=5,
            batch_size=2,
            log_interval=5
        )

        self.assertEqual(summary["total_batches_processed"], 5)
        self.assertTrue(summary["parameter_invariance_verified"])
        self.assertEqual(summary["num_bn_layers"], 20)
        self.assertGreater(len(summary["layers"]), 0)

        # Model must be returned in eval mode
        self.assertFalse(model.training)

        # All BN modules must have had their running stats tracked
        for name, bn in find_bn_modules(model):
            self.assertEqual(bn.num_batches_tracked.item(), 5)
            self.assertEqual(bn.momentum, 0.1)  # Momentum restored

    def test_save_recalibrated_checkpoint_structure(self):
        model = ResNet3D18(num_classes=10)
        fake_checkpoint = {
            "epoch": 42,
            "best_acc": 85.5,
            "optimizer_state_dict": {"state": {}},
            "model_state_dict": model.state_dict()
        }
        fake_summary = {
            "num_bn_layers": 20,
            "total_batches_processed": 10,
            "mode": "synthetic",
            "aggregate_mean_l1_shift": 0.05,
            "aggregate_var_l1_shift": 0.12
        }

        with tempfile.TemporaryDirectory() as td:
            out_ckpt = os.path.join(td, "recal_test.pth")
            save_recalibrated_checkpoint(fake_checkpoint, model, fake_summary, out_ckpt)
            self.assertTrue(os.path.exists(out_ckpt))

            loaded = torch.load(out_ckpt, map_location="cpu", weights_only=False)
            self.assertEqual(loaded["epoch"], 42)
            self.assertEqual(loaded["best_acc"], 85.5)
            self.assertTrue(loaded["precise_bn"]["recalibrated"])
            self.assertEqual(loaded["precise_bn"]["batches"], 10)
            self.assertIn("model_state_dict", loaded)

    def test_recalibrate_bn_zero_bn_layers(self):
        # A 3D model with 0 BatchNorm modules should not crash
        no_bn_model = nn.Sequential(nn.Conv3d(3, 4, 3, padding=1), nn.ReLU(), nn.AdaptiveAvgPool3d(1), nn.Flatten(), nn.Linear(4, 2))
        summary = recalibrate_bn_statistics(
            model=no_bn_model,
            device=torch.device("cpu"),
            is_synthetic=True,
            num_batches=3,
            batch_size=2,
            log_interval=3
        )
        self.assertEqual(summary["num_bn_layers"], 0)
        self.assertEqual(summary["total_batches_processed"], 3)
        self.assertEqual(summary["aggregate_mean_l1_shift"], 0.0)
        self.assertEqual(summary["max_mean_shift_layer"], "None")

    def test_f2_fixmatch_forward_method(self):
        # Verify standard PyTorch forward interface
        model = F2FixMatchModel(num_classes=10)
        model.eval()
        dummy_x = torch.randn(2, 3, 8, 224, 224)
        out_forward = model(dummy_x)
        out_eval = model.forward_eval(dummy_x)
        self.assertEqual(out_forward.shape, (2, 10))
        self.assertTrue(torch.equal(out_forward, out_eval))


if __name__ == "__main__":
    unittest.main()

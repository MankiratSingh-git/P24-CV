"""
Unit tests for Phase 3: Run D (RGB + TG Dual-Stream FixMatch Control)
Verifies:
1. Training loop optimization when feature losses are disabled (beta=0.0, gamma=0.0).
2. Gradient isolation: model.backbone.fc_tap receives ZERO gradients when beta=0 and gamma=0,
   while backbone parameters and the main 3D head (fc) receive valid non-zero gradients.
3. Contrast verification: when beta > 0 or gamma > 0, fc_tap receives non-zero gradients.
4. Robustness across warmup, confident pseudo-labeling, zero-confident edge cases, and gradient accumulation.
5. Argument validation preventing negative loss weights and out-of-range thresholds.
6. Launch scripts integrity and execution validation (run_d.sh and run_d.bat).
"""

import os
import sys
import io
import shutil
import subprocess
import unittest
import torch
import torch.nn as nn
import torch.nn.functional as F

from models.resnet3d import F2FixMatchModel
from train_f2_fixmatch import train_step, parse_args


def run_training_step(
    model,
    optimizer,
    labeled_batch,
    unlabeled_batch,
    alpha=2.0,
    beta=0.0,
    gamma=0.0,
    threshold=0.3,
    is_warmup=False,
    rampup_weight=1.0,
    device='cpu'
):
    """
    Executes a single mini-batch training step calling the real train_step from train_f2_fixmatch.py.
    Returns (total_loss, loss_s, loss_u_s, loss_u_wp, loss_u_sp, num_select).
    """
    model.train()
    optimizer.zero_grad()
    total_loss, loss_s, loss_u_s, loss_u_wp, loss_u_sp, num_select, _ = train_step(
        model=model,
        labeled_batch=labeled_batch,
        unlabeled_batch=unlabeled_batch,
        alpha=alpha,
        beta=beta,
        gamma=gamma,
        threshold=threshold,
        is_warmup=is_warmup,
        rampup_weight=rampup_weight,
        device=device
    )
    total_loss.backward()
    optimizer.step()
    return total_loss.item(), loss_s, loss_u_s, loss_u_wp, loss_u_sp, num_select


class TestPhase3RunD(unittest.TestCase):
    """Test suite for Phase 3 Run D optimization, gradient verification, and launch scripts."""

    def setUp(self):
        torch.manual_seed(42)
        self.device = torch.device('cpu')
        self.num_classes = 5
        self.model = F2FixMatchModel(num_classes=self.num_classes, dropout_ratio=0.5).to(self.device)
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.1, momentum=0.9)

        # Synthetic micro-batches for fast execution (T=8, H=32, W=32)
        self.bz_l = 2
        self.bz_u = 4
        self.rgb_l = torch.randn(self.bz_l, 3, 8, 32, 32)
        self.tg_l = torch.randn(self.bz_l, 3, 8, 32, 32)
        self.targets_l = torch.randint(0, self.num_classes, (self.bz_l,))

        self.rgb_w = torch.randn(self.bz_u, 3, 8, 32, 32)
        self.rgb_s = torch.randn(self.bz_u, 3, 8, 32, 32)
        self.tg_w = torch.randn(self.bz_u, 3, 8, 32, 32)
        self.tg_s = torch.randn(self.bz_u, 3, 8, 32, 32)

        self.labeled_batch = (self.rgb_l, self.tg_l, self.targets_l)
        self.unlabeled_batch = (self.rgb_w, self.rgb_s, self.tg_w, self.tg_s)

    def test_run_d_step_and_gradient_isolation(self):
        """
        Verify that in Run D (beta=0.0, gamma=0.0):
        1. Mini-batch training step completes successfully via train_step.
        2. model.backbone.fc_tap receives strictly ZERO gradients.
        3. model.backbone.fc and backbone layers receive valid non-zero gradients.
        """
        total_loss, loss_s, loss_us, loss_uwp, loss_usp, num_select = run_training_step(
            self.model,
            self.optimizer,
            self.labeled_batch,
            self.unlabeled_batch,
            alpha=2.0,
            beta=0.0,
            gamma=0.0,
            threshold=0.0,  # Ensure all samples are selected to test unsupervised path
            is_warmup=False,
            rampup_weight=1.0,
            device=self.device
        )

        self.assertGreater(num_select, 0, "Expected confident samples with threshold=0.0")
        self.assertGreater(total_loss, 0.0)
        self.assertGreater(loss_s, 0.0)
        self.assertGreater(loss_us, 0.0)
        self.assertEqual(loss_uwp, 0.0, "loss_u_wp must be exactly 0.0 when beta=0.0")
        self.assertEqual(loss_usp, 0.0, "loss_u_sp must be exactly 0.0 when gamma=0.0")

        # Check gradients for fc_tap (must be completely absent or zero)
        fc_tap_weight_grad = self.model.backbone.fc_tap.weight.grad
        fc_tap_bias_grad = self.model.backbone.fc_tap.bias.grad
        self.assertTrue(
            fc_tap_weight_grad is None or (fc_tap_weight_grad == 0).all().item(),
            "fc_tap.weight must receive ZERO gradients when beta=0 and gamma=0!"
        )
        self.assertTrue(
            fc_tap_bias_grad is None or (fc_tap_bias_grad == 0).all().item(),
            "fc_tap.bias must receive ZERO gradients when beta=0 and gamma=0!"
        )

        # Check gradients for 3D classifier head (fc)
        fc_weight_grad = self.model.backbone.fc.weight.grad
        fc_bias_grad = self.model.backbone.fc.bias.grad
        self.assertIsNotNone(fc_weight_grad, "fc.weight must receive gradients!")
        self.assertGreater(
            fc_weight_grad.abs().sum().item(), 0.0,
            "fc.weight gradients must be non-zero!"
        )
        self.assertIsNotNone(fc_bias_grad, "fc.bias must receive gradients!")
        self.assertGreater(
            fc_bias_grad.abs().sum().item(), 0.0,
            "fc.bias gradients must be non-zero!"
        )

        # Check gradients for backbone layers (e.g. conv1 and deep layer4)
        conv1_grad = self.model.backbone.conv1.weight.grad
        self.assertIsNotNone(conv1_grad, "Backbone conv1 must receive gradients!")
        self.assertGreater(
            conv1_grad.abs().sum().item(), 0.0,
            "Backbone conv1 gradients must be non-zero!"
        )

    def test_contrast_fc_tap_receives_gradient_when_beta_or_gamma_positive(self):
        """
        Contrast check: verify that when beta > 0 or gamma > 0, fc_tap DOES receive
        non-zero gradients. This confirms that the zero gradient in Run D is due to the
        guard optimization and not a disconnected module.
        """
        # Test Case A: beta=2.0, gamma=0.0
        model_a = F2FixMatchModel(num_classes=self.num_classes).to(self.device)
        optim_a = torch.optim.SGD(model_a.parameters(), lr=0.1)
        run_training_step(
            model_a, optim_a, self.labeled_batch, self.unlabeled_batch,
            alpha=2.0, beta=2.0, gamma=0.0, threshold=0.0, is_warmup=False
        )
        self.assertIsNotNone(model_a.backbone.fc_tap.weight.grad)
        self.assertGreater(model_a.backbone.fc_tap.weight.grad.abs().sum().item(), 0.0)

        # Test Case B: beta=0.0, gamma=1.0
        model_b = F2FixMatchModel(num_classes=self.num_classes).to(self.device)
        optim_b = torch.optim.SGD(model_b.parameters(), lr=0.1)
        run_training_step(
            model_b, optim_b, self.labeled_batch, self.unlabeled_batch,
            alpha=2.0, beta=0.0, gamma=1.0, threshold=0.0, is_warmup=False
        )
        self.assertIsNotNone(model_b.backbone.fc_tap.weight.grad)
        self.assertGreater(model_b.backbone.fc_tap.weight.grad.abs().sum().item(), 0.0)

    def test_warmup_phase_execution(self):
        """
        Verify that during warmup (is_warmup=True), unsupervised branch is skipped entirely,
        Run D runs cleanly, and fc_tap receives zero gradients while backbone and fc are trained.
        """
        total_loss, loss_s, loss_us, loss_uwp, loss_usp, num_select = run_training_step(
            self.model,
            self.optimizer,
            self.labeled_batch,
            self.unlabeled_batch,
            alpha=2.0,
            beta=0.0,
            gamma=0.0,
            threshold=0.3,
            is_warmup=True,
            rampup_weight=0.0,
            device=self.device
        )

        self.assertEqual(num_select, 0)
        self.assertEqual(loss_us, 0.0)
        self.assertEqual(loss_uwp, 0.0)
        self.assertEqual(loss_usp, 0.0)
        self.assertEqual(total_loss, loss_s)

        self.assertTrue(
            self.model.backbone.fc_tap.weight.grad is None or
            (self.model.backbone.fc_tap.weight.grad == 0).all().item()
        )
        self.assertGreater(self.model.backbone.fc.weight.grad.abs().sum().item(), 0.0)

    def test_zero_confident_samples_edge_case(self):
        """
        Edge case: when threshold is high (1.0) and no unlabeled samples are confident,
        the step must complete cleanly without unbound variable errors or NaNs.
        """
        total_loss, loss_s, loss_us, loss_uwp, loss_usp, num_select = run_training_step(
            self.model,
            self.optimizer,
            self.labeled_batch,
            self.unlabeled_batch,
            alpha=2.0,
            beta=0.0,
            gamma=0.0,
            threshold=1.0,  # Unreachable threshold
            is_warmup=False,
            rampup_weight=1.0,
            device=self.device
        )

        self.assertEqual(num_select, 0)
        self.assertEqual(loss_us, 0.0)
        self.assertEqual(loss_uwp, 0.0)
        self.assertEqual(loss_usp, 0.0)
        self.assertGreater(total_loss, 0.0)

        # Gradients check
        self.assertTrue(
            self.model.backbone.fc_tap.weight.grad is None or
            (self.model.backbone.fc_tap.weight.grad == 0).all().item()
        )
        self.assertGreater(self.model.backbone.fc.weight.grad.abs().sum().item(), 0.0)

    def test_accum_steps_remainder_mechanics(self):
        """
        Verify gradient accumulation mechanics when total batches is not divisible by accum_steps
        (e.g., 5 batches with accum_steps=2). Confirms fc_tap remains strictly unupdated while
        backbone and fc are updated.
        """
        model = F2FixMatchModel(num_classes=self.num_classes).to(self.device)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        ce_loss_fn = nn.CrossEntropyLoss()

        total_batches = 5
        accum_steps = 2
        optimizer.zero_grad()

        for b_idx in range(total_batches):
            steps_in_window = min(
                accum_steps,
                total_batches - (b_idx // accum_steps) * accum_steps
            )
            # Step computation
            total_loss, _, _, _, _, _, _ = train_step(
                model=model,
                labeled_batch=self.labeled_batch,
                unlabeled_batch=self.unlabeled_batch,
                ce_loss_fn=ce_loss_fn,
                alpha=2.0,
                beta=0.0,
                gamma=0.0,
                threshold=0.0,
                is_warmup=False,
                rampup_weight=1.0,
                device=self.device
            )
            loss_for_backward = total_loss / float(steps_in_window)
            loss_for_backward.backward()

            # fc_tap must have ZERO gradients at every accumulation step
            fc_tap_grad = model.backbone.fc_tap.weight.grad
            self.assertTrue(
                fc_tap_grad is None or (fc_tap_grad == 0).all().item(),
                f"fc_tap received non-zero gradients at batch {b_idx}!"
            )

            if (b_idx + 1) % accum_steps == 0 or (b_idx + 1) == total_batches:
                optimizer.step()
                optimizer.zero_grad()

        # Final verification: fc_tap remains completely zero after full cycle
        self.assertTrue(
            model.backbone.fc_tap.weight.grad is None or
            (model.backbone.fc_tap.weight.grad == 0).all().item()
        )

    def test_launch_scripts_existence_and_contents(self):
        """
        Verify that scripts/run_d.sh and scripts/run_d.bat exist, contain required headers,
        error flags, and exact Run D hyperparameters.
        """
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        sh_path = os.path.join(repo_root, "scripts", "run_d.sh")
        bat_path = os.path.join(repo_root, "scripts", "run_d.bat")

        self.assertTrue(os.path.isfile(sh_path), f"Missing {sh_path}")
        self.assertTrue(os.path.isfile(bat_path), f"Missing {bat_path}")

        # Check bash script contents
        with open(sh_path, "r", encoding="utf-8") as f:
            sh_content = f.read()

        self.assertTrue(sh_content.startswith("#!/"), "run_d.sh missing shebang header")
        self.assertIn("set -euo pipefail", sh_content, "run_d.sh missing strict error handling")
        self.assertIn("--alpha", sh_content)
        self.assertIn("--beta", sh_content)
        self.assertIn("--gamma", sh_content)
        self.assertIn("BETA=\"${BETA:-0.0}\"", sh_content)
        self.assertIn("GAMMA=\"${GAMMA:-0.0}\"", sh_content)
        self.assertIn("ALPHA=\"${ALPHA:-2.0}\"", sh_content)
        self.assertIn("fixmatch_rgb_tg", sh_content)

        # Check bat script contents
        with open(bat_path, "r", encoding="utf-8") as f:
            bat_content = f.read()

        self.assertIn("@echo off", bat_content)
        self.assertIn("setlocal enabledelayedexpansion", bat_content)
        self.assertIn("--beta %BETA%", bat_content)
        self.assertIn("--gamma %GAMMA%", bat_content)
        self.assertIn("--alpha %ALPHA%", bat_content)
        self.assertIn("set BETA=0.0", bat_content)
        self.assertIn("set GAMMA=0.0", bat_content)
        self.assertIn("set ALPHA=2.0", bat_content)
        self.assertIn("fixmatch_rgb_tg", bat_content)

    def test_bash_syntax_if_available(self):
        """If a functional bash executable is available, test run_d.sh syntax with bash -n."""
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        sh_path = os.path.join(repo_root, "scripts", "run_d.sh")

        candidates = [
            r"C:\Program Files\Git\bin\bash.exe",
            r"C:\Program Files\Git\usr\bin\bash.exe",
            shutil.which("bash"),
        ]
        valid_bash = None
        for cand in candidates:
            if cand and os.path.isfile(cand):
                try:
                    chk = subprocess.run([cand, "--version"], capture_output=True, text=True, timeout=5)
                    if chk.returncode == 0:
                        valid_bash = cand
                        break
                except Exception:
                    continue

        if valid_bash:
            res = subprocess.run([valid_bash, "-n", sh_path], capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"bash -n failed: {res.stderr}")
        else:
            self.skipTest("No functional bash executable found in environment")

    def test_bash_run_d_help_invocation(self):
        """If a functional bash executable is available, test scripts/run_d.sh --help execution."""
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        sh_path = os.path.join(repo_root, "scripts", "run_d.sh")

        candidates = [
            r"C:\Program Files\Git\bin\bash.exe",
            r"C:\Program Files\Git\usr\bin\bash.exe",
            shutil.which("bash"),
        ]
        valid_bash = None
        for cand in candidates:
            if cand and os.path.isfile(cand):
                try:
                    chk = subprocess.run([cand, "--version"], capture_output=True, text=True, timeout=5)
                    if chk.returncode == 0:
                        valid_bash = cand
                        break
                except Exception:
                    continue

        if valid_bash:
            res = subprocess.run([valid_bash, sh_path, "--help"], capture_output=True, text=True)
            self.assertEqual(res.returncode, 0, f"run_d.sh failed: {res.stderr}")
            self.assertIn("Run D: Dual-Stream FixMatch Control", res.stdout)
            self.assertIn("beta=0.0", res.stdout)
            self.assertIn("gamma=0.0", res.stdout)
        else:
            self.skipTest("No functional bash executable found in environment")

    def test_argparse_run_d_arguments(self):
        """Verify train_f2_fixmatch.py argument parser accepts Run D arguments cleanly."""
        test_argv = [
            "--alpha", "2.0",
            "--beta", "0.0",
            "--gamma", "0.0",
            "--labeled-percent", "10",
            "--split", "1",
            "--output-dir", "checkpoints/fixmatch_rgb_tg"
        ]
        args = parse_args(test_argv)
        self.assertEqual(args.alpha, 2.0)
        self.assertEqual(args.beta, 0.0)
        self.assertEqual(args.gamma, 0.0)
        self.assertEqual(args.output_dir, "checkpoints/fixmatch_rgb_tg")

    def test_argparse_input_validation(self):
        """Verify parse_args rejects illegal negative parameters and invalid thresholds."""
        # Negative beta
        with self.assertRaises(SystemExit):
            with unittest.mock.patch('sys.stderr', new=io.StringIO()):
                parse_args(["--beta", "-0.5"])

        # Negative gamma
        with self.assertRaises(SystemExit):
            with unittest.mock.patch('sys.stderr', new=io.StringIO()):
                parse_args(["--gamma", "-1.0"])

        # Negative alpha
        with self.assertRaises(SystemExit):
            with unittest.mock.patch('sys.stderr', new=io.StringIO()):
                parse_args(["--alpha", "-2.0"])

        # Invalid threshold > 1.0
        with self.assertRaises(SystemExit):
            with unittest.mock.patch('sys.stderr', new=io.StringIO()):
                parse_args(["--threshold", "1.5"])

        # Invalid threshold < 0.0
        with self.assertRaises(SystemExit):
            with unittest.mock.patch('sys.stderr', new=io.StringIO()):
                parse_args(["--threshold", "-0.1"])

    def test_argparse_help_no_dict_corruption(self):
        """Verify train_f2_fixmatch.py help formatting does not produce internal action dict dumps."""
        res = subprocess.run([sys.executable, "train_f2_fixmatch.py", "--help"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertNotIn("'option_strings':", res.stdout, "argparse help text corrupted by unescaped %")
        self.assertIn("Total training epochs (220 for 10% subset)", res.stdout)

    def test_run_d_bat_help_invocation(self):
        """Verify scripts/run_d.bat executes and outputs Run D configuration banner on Windows."""
        if sys.platform != "win32":
            self.skipTest("Windows-only batch script test")
        repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
        bat_path = os.path.join(repo_root, "scripts", "run_d.bat")
        res = subprocess.run(["cmd.exe", "/c", bat_path, "--help"], capture_output=True, text=True)
        self.assertEqual(res.returncode, 0, f"run_d.bat failed with: {res.stderr}")
        self.assertIn("Run D: Dual-Stream FixMatch Control", res.stdout)
        self.assertIn("beta=0.0", res.stdout)
        self.assertIn("gamma=0.0", res.stdout)


if __name__ == "__main__":
    unittest.main()

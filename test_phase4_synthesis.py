"""
test_phase4_synthesis.py: Unit tests for Phase 4 Synthesis & Mathematical Disentanglement

Verifies:
1. Mathematical attribution calculations (Delta_dual = D - C, Delta_feat = F - D, Delta_total = F - C).
2. Pre-declared case classification logic (Case A, Case B, Case C, Case D) from implementation plan.
3. Metric loading and ingestion robustness across schema variants (eval_metrics_full30view, metrics.json).
4. Missing file handling and error reporting when runs are absent without --mock-d.
5. CLI argument parsing, input validation, and flag combinations.
6. Markdown report generation formatting, ASCII bar visualizations, and Model Update Rule inclusion.
7. Idempotent updating of RESULTS.md without duplicating sections or corrupting existing content.
8. End-to-end programmatic and subprocess CLI executions.
"""

import os
import sys
import json
import tempfile
import unittest
from unittest.mock import patch
import subprocess

from synthesize_results import (
    load_metrics,
    calculate_attributions,
    classify_case,
    make_ascii_bar,
    generate_markdown_report,
    update_results_md,
    format_results_section_4,
    synthesize,
    parse_args,
    MODEL_UPDATE_RULE,
)


class TestPhase4Synthesis(unittest.TestCase):
    """Comprehensive test suite for Phase 4 synthesis, attribution, and reporting."""

    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)

        # Baseline reference values
        self.c_acc = 38.49
        self.f_acc = 59.90

    # -------------------------------------------------------------------------
    # 1. Mathematical Attribution Calculations
    # -------------------------------------------------------------------------
    def test_calculate_attributions_standard(self):
        """Verify attribution math for standard values (D=55.0)."""
        d_acc = 55.00
        attr = calculate_attributions(self.c_acc, d_acc, self.f_acc)

        expected_dual = round(55.00 - 38.49, 2)  # +16.51
        expected_feat = round(59.90 - 55.00, 2)  # +4.90
        expected_total = round(59.90 - 38.49, 2) # +21.41

        self.assertAlmostEqual(attr["delta_dual"], expected_dual, places=2)
        self.assertAlmostEqual(attr["delta_feat"], expected_feat, places=2)
        self.assertAlmostEqual(attr["delta_total"], expected_total, places=2)

        # Total should equal sum of components
        self.assertAlmostEqual(attr["delta_dual"] + attr["delta_feat"], attr["delta_total"], places=2)

        # Share percentages should sum to 100%
        self.assertAlmostEqual(attr["share_dual"] + attr["share_feat"], 100.0, places=1)

    def test_calculate_attributions_zero_total_delta(self):
        """Edge case: F == C (zero total gain). Must not raise ZeroDivisionError."""
        attr = calculate_attributions(40.0, 40.0, 40.0)
        self.assertEqual(attr["delta_total"], 0.0)
        self.assertEqual(attr["delta_dual"], 0.0)
        self.assertEqual(attr["delta_feat"], 0.0)
        self.assertEqual(attr["share_dual"], 0.0)
        self.assertEqual(attr["share_feat"], 0.0)

    def test_calculate_attributions_negative_deltas(self):
        """Verify handling when Run D underperforms C or F underperforms D."""
        # D < C (dual stream hurts)
        attr = calculate_attributions(40.0, 35.0, 50.0)
        self.assertEqual(attr["delta_dual"], -5.0)
        self.assertEqual(attr["delta_feat"], 15.0)
        self.assertEqual(attr["delta_total"], 10.0)
        self.assertEqual(attr["share_dual"], -50.0)
        self.assertEqual(attr["share_feat"], 150.0)

    def test_calculate_attributions_negative_total_gain(self):
        """When F < C (negative total SSL gain), percentage shares must be 0.0% to avoid inverted proportions."""
        attr = calculate_attributions(c_top1=40.0, d_top1=38.0, f_top1=35.0)
        self.assertEqual(attr["delta_total"], -5.0)
        self.assertEqual(attr["delta_dual"], -2.0)
        self.assertEqual(attr["delta_feat"], -3.0)
        self.assertEqual(attr["share_dual"], 0.0)
        self.assertEqual(attr["share_feat"], 0.0)

    def test_make_ascii_bar_edge_cases(self):
        """Verify ASCII progress bar doesn't crash on zero/negative max_val or zero width."""
        bar_zero = make_ascii_bar(10.0, max_val=0.0, width=20)
        self.assertEqual(bar_zero, f"[{' ' * 20}]")

        bar_neg = make_ascii_bar(10.0, max_val=-5.0, width=15)
        self.assertEqual(bar_neg, f"[{' ' * 15}]")

        bar_zero_width = make_ascii_bar(10.0, max_val=20.0, width=0)
        self.assertEqual(bar_zero_width, "[]")

    # -------------------------------------------------------------------------
    # 2. Pre-Declared Case Classification Logic
    # -------------------------------------------------------------------------
    def test_classify_case_a(self):
        """
        Case A: C=38.5 -> D approx 55 -> F=59.9
        Dual-stream configuration is dominant (+16.5 pts); feature module is meaningful (+4.9 pts).
        """
        res = classify_case(c_top1=38.49, d_top1=55.00, f_top1=59.90)
        self.assertEqual(res["case_id"], "Case A")
        self.assertIn("Dual-Stream Dominant", res["title"])
        self.assertGreater(res["delta_dual"], res["delta_feat"])
        self.assertGreaterEqual(res["delta_feat"], 2.0)

    def test_classify_case_b(self):
        """
        Case B: C=38.5 -> D approx 42 -> F=59.9
        Feature-level machinery is primary driver (F - D approx +17.9 pts > D - C approx +3.5 pts).
        """
        res = classify_case(c_top1=38.49, d_top1=42.00, f_top1=59.90)
        self.assertEqual(res["case_id"], "Case B")
        self.assertIn("Feature-Level Machinery", res["title"])
        self.assertGreater(res["delta_feat"], res["delta_dual"])

    def test_classify_case_c(self):
        """
        Case C: C=38.5 -> D approx 55 -> F approx 56
        Dual-stream dominant; feature-level module adds negligible benefit (< 2.0 pts).
        """
        res = classify_case(c_top1=38.49, d_top1=55.00, f_top1=56.00)
        self.assertEqual(res["case_id"], "Case C")
        self.assertIn("Feature Module Negligible", res["title"])
        self.assertLess(res["delta_feat"], 2.0)

    def test_classify_case_d(self):
        """
        Case D: C=38.5 -> D approx 40 -> F approx 41
        Implementation does not reproduce claimed F^2 benefit over FixMatch-RGB (F - C < 6.0 pts).
        """
        res = classify_case(c_top1=38.49, d_top1=40.00, f_top1=41.00)
        self.assertEqual(res["case_id"], "Case D")
        self.assertIn("Does Not Reproduce", res["title"])
        self.assertLess(res["delta_total"], 6.0)

    def test_classify_custom_thresholds(self):
        """Verify custom threshold sensitivity."""
        # Default min_f2_gain is 6.0. If F - C = 5.0, default is Case D.
        res_default = classify_case(c_top1=35.0, d_top1=38.0, f_top1=40.0)
        self.assertEqual(res_default["case_id"], "Case D")

        # With min_f2_gain=4.0, F - C = 5.0 >= 4.0; delta_feat = 2.0 >= 2.0; delta_dual = 3.0 >= 2.0 -> Case A
        res_custom = classify_case(
            c_top1=35.0, d_top1=38.0, f_top1=40.0, min_f2_gain=4.0, negligible_feat_threshold=1.5
        )
        self.assertEqual(res_custom["case_id"], "Case A")

    # -------------------------------------------------------------------------
    # 3. Metrics Ingestion & Schema Support
    # -------------------------------------------------------------------------
    def test_load_metrics_schema_variants(self):
        """Verify load_metrics correctly extracts metrics across diverse JSON schemas."""
        # Schema 1: eval_metrics_full30view format
        path1 = os.path.join(self.temp_dir.name, "m1.json")
        with open(path1, "w") as f:
            json.dump({
                "top1_accuracy": 59.9,
                "top5_accuracy": 83.32,
                "eval_mode": "full",
                "num_eval_videos": 3783,
                "checkpoint": "model_full.pth"
            }, f)
        m1 = load_metrics(path1)
        self.assertEqual(m1["top1"], 59.9)
        self.assertEqual(m1["top5"], 83.32)
        self.assertEqual(m1["num_eval_videos"], 3783)

        # Schema 2: metrics.json format (best_val_top1)
        path2 = os.path.join(self.temp_dir.name, "m2.json")
        with open(path2, "w") as f:
            json.dump({
                "best_val_top1": 71.69,
                "best_val_top5": 89.14,
                "epochs": 120,
                "checkpoint": "model_100.pth"
            }, f)
        m2 = load_metrics(path2)
        self.assertEqual(m2["top1"], 71.69)
        self.assertEqual(m2["top5"], 89.14)

    def test_load_metrics_missing_top1_key(self):
        """Verify ValueError raised when no valid top-1 key exists."""
        bad_path = os.path.join(self.temp_dir.name, "bad.json")
        with open(bad_path, "w") as f:
            json.dump({"random_key": 42}, f)
        with self.assertRaises(ValueError):
            load_metrics(bad_path)

    def test_load_metrics_non_dict_json(self):
        """Verify ValueError raised when JSON root is not a dictionary."""
        list_path = os.path.join(self.temp_dir.name, "list.json")
        with open(list_path, "w") as f:
            json.dump([1, 2, 3], f)
        with self.assertRaises(ValueError) as ctx:
            load_metrics(list_path)
        self.assertIn("must contain a JSON object", str(ctx.exception))

    def test_load_metrics_accuracy_out_of_bounds(self):
        """Verify ValueError raised when accuracy is outside [0.0, 100.0]."""
        high_path = os.path.join(self.temp_dir.name, "high.json")
        with open(high_path, "w") as f:
            json.dump({"top1_accuracy": 105.0}, f)
        with self.assertRaises(ValueError) as ctx:
            load_metrics(high_path)
        self.assertIn("must be in range [0.0, 100.0]", str(ctx.exception))

        low_path = os.path.join(self.temp_dir.name, "low.json")
        with open(low_path, "w") as f:
            json.dump({"top1_accuracy": -5.0}, f)
        with self.assertRaises(ValueError) as ctx:
            load_metrics(low_path)
        self.assertIn("must be in range [0.0, 100.0]", str(ctx.exception))

    def test_load_metrics_top5_less_than_top1(self):
        """Verify ValueError raised when top5 < top1."""
        inv_path = os.path.join(self.temp_dir.name, "inv.json")
        with open(inv_path, "w") as f:
            json.dump({"top1_accuracy": 60.0, "top5_accuracy": 40.0}, f)
        with self.assertRaises(ValueError) as ctx:
            load_metrics(inv_path)
        self.assertIn("cannot be less than Top-1", str(ctx.exception))

    def test_synthesize_metrics_with_none_top5(self):
        """Verify synthesize handles metrics files where top-5 is absent (None) without crashing."""
        no_top5_path = os.path.join(self.temp_dir.name, "no_top5.json")
        with open(no_top5_path, "w") as f:
            json.dump({"top1_accuracy": 50.0}, f)

        report_out = os.path.join(self.temp_dir.name, "report_no_top5.md")
        results_out = os.path.join(self.temp_dir.name, "RESULTS_no_top5.md")
        with open(results_out, "w") as f:
            f.write("# Results\n")

        # Point Run F to the file missing top5
        res = synthesize(
            run_f_path=no_top5_path,
            mock_d_top1=45.0,
            output_report_path=report_out,
            results_md_path=results_out,
            update_results=True,
        )
        self.assertIsNone(res["ladder"]["F"]["top5"])
        with open(report_out, "r", encoding="utf-8") as f:
            rep = f.read()
        self.assertIn("N/A", rep)
        with open(results_out, "r", encoding="utf-8") as f:
            res_content = f.read()
        self.assertIn("N/A", res_content)

    # -------------------------------------------------------------------------
    # 4. Missing File Handling
    # -------------------------------------------------------------------------
    def test_missing_run_d_without_mock_raises_filenotfound(self):
        """Verify FileNotFoundError raised with clear guidance when Run D is missing and not mocked."""
        non_existent_d = os.path.join(self.temp_dir.name, "non_existent_run_d.json")
        with self.assertRaises(FileNotFoundError) as ctx:
            synthesize(
                run_d_path=non_existent_d,
                mock_d_top1=None,
                update_results=False,
            )
        self.assertIn("Run D evaluation metrics not found", str(ctx.exception))
        self.assertIn("--mock-d", str(ctx.exception))

    def test_missing_run_a_raises_filenotfound(self):
        """Verify FileNotFoundError raised when Run A is missing."""
        non_existent_a = os.path.join(self.temp_dir.name, "missing_run_a.json")
        with self.assertRaises(FileNotFoundError):
            synthesize(
                run_a_path=non_existent_a,
                mock_d_top1=55.0,
                update_results=False,
            )

    # -------------------------------------------------------------------------
    # 5. CLI Argument Parsing & Help String Validation
    # -------------------------------------------------------------------------
    def test_parse_args_mock_d(self):
        """Verify CLI parser processes --mock-d and related arguments."""
        test_args = [
            "synthesize_results.py",
            "--mock-d", "55.0",
            "--mock-d-top5", "79.5",
            "--threshold-min-f2", "7.0",
            "--no-update-results",
        ]
        with patch("sys.argv", test_args):
            args = parse_args()
            self.assertEqual(args.mock_d, 55.0)
            self.assertEqual(args.mock_d_top5, 79.5)
            self.assertEqual(args.threshold_min_f2, 7.0)
            self.assertTrue(args.no_update_results)

    def test_mock_d_out_of_bounds(self):
        """Verify ValueError raised when mock_d accuracy is out of range [0.0, 100.0]."""
        with self.assertRaises(ValueError) as ctx:
            synthesize(mock_d_top1=110.0, update_results=False)
        self.assertIn("must be in range [0.0, 100.0]", str(ctx.exception))

        with self.assertRaises(ValueError) as ctx:
            synthesize(mock_d_top1=-5.0, update_results=False)
        self.assertIn("must be in range [0.0, 100.0]", str(ctx.exception))

    def test_mock_d_top5_less_than_top1(self):
        """Verify ValueError raised when mock_d_top5 < mock_d_top1."""
        with self.assertRaises(ValueError) as ctx:
            synthesize(mock_d_top1=55.0, mock_d_top5=45.0, update_results=False)
        self.assertIn("cannot be less than", str(ctx.exception))

    def test_cli_failure_exit_code_on_missing_run_d(self):
        """Verify CLI exits with code 1 and actionable error when Run D is absent without mock."""
        cmd = [
            sys.executable,
            "synthesize_results.py",
            "--run-d", os.path.join(self.temp_dir.name, "absent_d.json"),
            "--no-update-results",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 1)
        self.assertIn("[Error] Synthesis failed", res.stderr)
        self.assertIn("--mock-d", res.stderr)

    def test_cli_help_no_percent_formatting_error(self):
        """Verify synthesize_results.py --help succeeds without % format string errors."""
        res = subprocess.run(
            [sys.executable, "synthesize_results.py", "--help"],
            capture_output=True,
            text=True,
        )
        self.assertEqual(res.returncode, 0)
        self.assertIn("Phase 4: Synthesis", res.stdout)
        self.assertIn("--mock-d", res.stdout)

    # -------------------------------------------------------------------------
    # 6. Report Formatting & Model Update Rule
    # -------------------------------------------------------------------------
    def test_markdown_report_structure(self):
        """Verify generated markdown report contains all 5 rungs, attributions, and update rule."""
        ladder = {
            "A": {"top1": 71.69, "top5": 89.14, "checkpoint": "ckpt_a.pth"},
            "B": {"top1": 31.32, "top5": 54.11, "checkpoint": "ckpt_b.pth"},
            "C": {"top1": 38.49, "top5": 64.00, "checkpoint": "ckpt_c.pth"},
            "D": {"top1": 55.00, "top5": 79.00, "checkpoint": "ckpt_d.pth"},
            "F": {"top1": 59.90, "top5": 83.32, "checkpoint": "ckpt_f.pth"},
        }
        attributions = calculate_attributions(38.49, 55.00, 59.90)
        classification = classify_case(38.49, 55.00, 59.90)

        report = generate_markdown_report(
            ladder=ladder,
            attributions=attributions,
            classification=classification,
            mock_d_used=True,
        )

        # Check required contents
        self.assertIn("Run A", report)
        self.assertIn("Run B", report)
        self.assertIn("Run C", report)
        self.assertIn("Run D", report)
        self.assertIn("Run F", report)
        self.assertIn(MODEL_UPDATE_RULE, report)
        self.assertIn("\\Delta_{\\text{dual}}", report)
        self.assertIn("\\Delta_{\\text{feat}}", report)
        self.assertIn("Case A", report)
        self.assertIn("Simulated / Mocked (`--mock-d`)", report)

    # -------------------------------------------------------------------------
    # 7. RESULTS.md Updating & Idempotency
    # -------------------------------------------------------------------------
    def test_update_results_md_idempotency(self):
        """Verify that updating RESULTS.md multiple times replaces Section 4 without duplication."""
        initial_results_text = (
            "# Benchmark Results\n\n"
            "## 1. Summary\nSome initial content.\n\n"
            "## 2. Methodology\nMethodology details.\n"
        )
        test_results_path = os.path.join(self.temp_dir.name, "TEST_RESULTS.md")
        with open(test_results_path, "w", encoding="utf-8") as f:
            f.write(initial_results_text)

        sec4_v1 = "## 4. Phase 4 Synthesis & Disentanglement Analysis\n\nVersion 1 Content"
        sec4_v2 = "## 4. Phase 4 Synthesis & Disentanglement Analysis\n\nVersion 2 Content"

        # First update (appends)
        update_results_md(test_results_path, sec4_v1)
        with open(test_results_path, "r", encoding="utf-8") as f:
            c1 = f.read()
        self.assertIn("Version 1 Content", c1)
        self.assertIn("## 1. Summary", c1)
        self.assertEqual(c1.count("## 4. Phase 4 Synthesis & Disentanglement Analysis"), 1)

        # Second update (replaces in place, no duplication)
        update_results_md(test_results_path, sec4_v2)
        with open(test_results_path, "r", encoding="utf-8") as f:
            c2 = f.read()
        self.assertNotIn("Version 1 Content", c2)
        self.assertIn("Version 2 Content", c2)
        self.assertIn("## 1. Summary", c2)
        self.assertEqual(c2.count("## 4. Phase 4 Synthesis & Disentanglement Analysis"), 1)

    # -------------------------------------------------------------------------
    # 8. Physical Run D Integration Test
    # -------------------------------------------------------------------------
    def test_physical_run_d_metrics_file_flow(self):
        """Verify synthesize runs cleanly with a physical Run D evaluation metrics JSON file."""
        run_d_json_path = os.path.join(self.temp_dir.name, "eval_metrics_run_d.json")
        with open(run_d_json_path, "w") as f:
            json.dump({
                "top1_accuracy": 54.85,
                "top5_accuracy": 79.12,
                "eval_mode": "full",
                "num_eval_videos": 3783,
                "checkpoint": "checkpoints/fixmatch_rgb_tg/best_model.pth"
            }, f)

        report_out = os.path.join(self.temp_dir.name, "test_report.md")
        results_out = os.path.join(self.temp_dir.name, "TEST_RESULTS.md")
        with open(results_out, "w") as f:
            f.write("# Empirical Experimental Results\n")

        res = synthesize(
            run_d_path=run_d_json_path,
            mock_d_top1=None,
            output_report_path=report_out,
            results_md_path=results_out,
            update_results=True,
        )

        self.assertFalse(res["mock_d_used"])
        self.assertEqual(res["ladder"]["D"]["top1"], 54.85)
        self.assertTrue(os.path.isfile(report_out))
        self.assertTrue(os.path.isfile(results_out))

        with open(report_out, "r", encoding="utf-8") as f:
            report_text = f.read()
        self.assertIn("54.85%", report_text)
        self.assertIn("Physical Run (30-View Matched)", report_text)
        self.assertNotIn("Simulated / Mocked", report_text)

    # -------------------------------------------------------------------------
    # 9. Subprocess Execution & JSON Summary Export
    # -------------------------------------------------------------------------
    def test_subprocess_cli_with_json_summary(self):
        """Verify full CLI execution via subprocess exporting report and JSON summary."""
        report_path = os.path.join(self.temp_dir.name, "cli_report.md")
        json_summary_path = os.path.join(self.temp_dir.name, "summary.json")

        cmd = [
            sys.executable,
            "synthesize_results.py",
            "--mock-d", "55.0",
            "--output-report", report_path,
            "--json-summary", json_summary_path,
            "--no-update-results",
        ]
        res = subprocess.run(cmd, capture_output=True, text=True)
        self.assertEqual(res.returncode, 0)
        self.assertIn("PHASE 4 SYNTHESIS", res.stdout)
        self.assertTrue(os.path.isfile(report_path))
        self.assertTrue(os.path.isfile(json_summary_path))

        with open(json_summary_path, "r", encoding="utf-8") as f:
            summary = json.load(f)
        self.assertEqual(summary["classification"]["case_id"], "Case A")
        self.assertAlmostEqual(summary["attributions"]["delta_dual"], 16.51, places=2)
        self.assertEqual(summary["model_update_rule"], MODEL_UPDATE_RULE)


if __name__ == "__main__":
    unittest.main()

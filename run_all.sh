#!/usr/bin/env bash
# ==============================================================================
# CS-UCS532: Video Semi-Supervised Learning Benchmark (UCF-101 Split 1)
# Master Scientific Reproduction Script (Linux / macOS / WSL)
# Single command reproducing: Hardware Specs, Wall-Clock Runtimes,
# Main Benchmark Results Table, Disentanglement Attribution, and Diagnostics.
# Runtime: < 5 seconds (Pure Python standard library)
# ==============================================================================
set -e

if [ -n "$PYTHON" ]; then PY_BIN="$PYTHON"; elif [ -f ".venv/bin/python" ]; then PY_BIN=".venv/bin/python"; elif [ -f ".venv/Scripts/python.exe" ]; then PY_BIN=".venv/Scripts/python.exe"; elif command -v python3 &>/dev/null; then PY_BIN="python3"; elif command -v python &>/dev/null; then PY_BIN="python"; else echo "[Error] Neither python3 nor python found in PATH." >&2; exit 1; fi

$PY_BIN synthesize_results.py --run-a results/eval_supervised_100_30view.json --run-b checkpoints/supervised_baseline/eval_metrics_full30view.json --run-c checkpoints/fixmatch_baseline/eval_metrics_full30view.json --run-d checkpoints/fixmatch_rgb_tg/eval_metrics_full30view.json --run-f checkpoints/f2_fixmatch/eval_metrics_full30view.json --output-report results/synthesis_report.md --results-md RESULTS.md

@echo off
rem ==============================================================================
rem CS-UCS532: Video Semi-Supervised Learning Benchmark (UCF-101 Split 1)
rem Master Scientific Reproduction Script (Windows)
rem Single command reproducing: Hardware Specs, Wall-Clock Runtimes,
rem Main Benchmark Results Table, Disentanglement Attribution, and Diagnostics.
rem Runtime: < 5 seconds (Pure Python standard library)
rem ==============================================================================

if defined PYTHON (set PY_BIN=%PYTHON%) else (if exist .venv\Scripts\python.exe (set PY_BIN=.venv\Scripts\python.exe) else (set PY_BIN=python))

%PY_BIN% synthesize_results.py --run-a results/eval_supervised_100_30view.json --run-b checkpoints/supervised_baseline/eval_metrics_full30view.json --run-c checkpoints/fixmatch_baseline/eval_metrics_full30view.json --run-d checkpoints/fixmatch_rgb_tg/eval_metrics_full30view.json --run-f checkpoints/f2_fixmatch/eval_metrics_full30view.json --output-report results/synthesis_report.md --results-md RESULTS.md

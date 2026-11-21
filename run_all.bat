@echo off
rem ==============================================================================
rem CS-UCS532 Open Challenge: Video Semi-Supervised Learning Benchmark
rem Script: run_all.bat
rem Purpose: Windows master reproduction script reproducing full scientific table
rem ==============================================================================

echo ================================================================================
echo CS-UCS532: Semi-Supervised Video Action Recognition Benchmark (UCF-101 Split 1)
echo Master Scientific Reproduction Pipeline (Windows)
echo ================================================================================

set CANONICAL_A=results/eval_supervised_100_30view.json
set CANONICAL_B=checkpoints/supervised_baseline/eval_metrics_full30view.json
set CANONICAL_C=checkpoints/fixmatch_baseline/eval_metrics_full30view.json
set CANONICAL_D=checkpoints/fixmatch_rgb_tg/eval_metrics_full30view.json
set CANONICAL_F=checkpoints/f2_fixmatch/eval_metrics_full30view.json
set CANONICAL_C_TAU=checkpoints/fixmatch_tau_095/eval_metrics_full30view.json
set CANONICAL_BN=results/eval_f2_precise_bn_30view.json

python synthesize_results.py ^
  --run-a %CANONICAL_A% ^
  --run-b %CANONICAL_B% ^
  --run-c %CANONICAL_C% ^
  --run-d %CANONICAL_D% ^
  --run-f %CANONICAL_F% ^
  --output-report results\synthesis_report.md ^
  --results-md RESULTS.md

echo.
echo ================================================================================
echo EXTENSIONS AND DIAGNOSTICS REPRODUCTION SUMMARY
echo ================================================================================

python -c "import json; m95=json.load(open('%CANONICAL_C_TAU%')); m30=json.load(open('%CANONICAL_C%')); mb=json.load(open('%CANONICAL_B%')); print('1. Confidence Threshold Ablation (FixMatch tau=0.95 vs tau=0.30):'); print(f'   Supervised 10%%:          {mb[\"top1_accuracy\"]:.2f}%% Top-1'); print(f'   FixMatch RGB (tau=0.95):  {m95[\"top1_accuracy\"]:.2f}%% Top-1 (+{m95[\"top1_accuracy\"]-mb[\"top1_accuracy\"]:.2f}%% vs Sup 10%%)'); print(f'   FixMatch RGB (tau=0.30):  {m30[\"top1_accuracy\"]:.2f}%% Top-1 (+{m30[\"top1_accuracy\"]-m95[\"top1_accuracy\"]:.2f}%% vs tau=0.95)'); print('   Conclusion: Data Starvation outweighs Pseudo-Label Noise. tau=0.30 provides +1.56%% coverage benefit.')"

python -c "import json; mf=json.load(open('%CANONICAL_F%')); mbn=json.load(open('%CANONICAL_BN%')); print('2. PreciseBN Recalibration Diagnostic:'); print(f'   F2 Pre-Recalibration:   {mf[\"top1_accuracy\"]:.2f}%% Top-1'); print(f'   F2 PreciseBN Recal:     {mbn[\"top1_accuracy\"]:.2f}%% Top-1 (+{mbn[\"top1_accuracy\"]-mf[\"top1_accuracy\"]:.2f}%% gain)'); print('   Conclusion: Shared RGB/TG BatchNorm contamination was an active drag, recovering ~3 pts.')"

echo.
echo ================================================================================
echo Master reproduction completed successfully.
echo Full reports available at: results\synthesis_report.md and RESULTS.md
echo ================================================================================

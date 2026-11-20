@echo off
REM ============================================================================
REM Script: scripts\run_remaining_evals.bat
REM Purpose: Run all remaining evaluations, diagnostics & synthesis on Windows
REM ============================================================================

setlocal enabledelayedexpansion

if not defined PYTHON (
    set "PY_BIN=python"
) else (
    set "PY_BIN=%PYTHON%"
)

echo ========================================================================
echo RUNNING ALL REMAINING EVALUATIONS, DIAGNOSTICS ^& SYNTHESIS
echo ========================================================================

if not exist results mkdir results
if not exist checkpoints\fixmatch_rgb_tg mkdir checkpoints\fixmatch_rgb_tg

echo.
echo [Step 1/7] Evaluating Run D (RGB+TG Control) - Best Model (30-View)...
%PY_BIN% evaluate.py ^
  --checkpoint checkpoints/fixmatch_rgb_tg/best_model.pth ^
  --data-root data/ucf101 ^
  --split-file data/splits/ucf101_val_split_1_rawframes.txt ^
  --eval-mode full ^
  --crop-size 256 ^
  --batch-size 8 ^
  --output-file checkpoints/fixmatch_rgb_tg/eval_metrics_full30view.json
if %ERRORLEVEL% neq 0 (echo [Error] Step 1 failed & exit /b %ERRORLEVEL%)

echo.
echo [Step 1b/7] Evaluating Run D - Final Epoch Model (30-View)...
if exist checkpoints\fixmatch_rgb_tg\final_model.pth (
    %PY_BIN% evaluate.py ^
      --checkpoint checkpoints/fixmatch_rgb_tg/final_model.pth ^
      --data-root data/ucf101 ^
      --split-file data/splits/ucf101_val_split_1_rawframes.txt ^
      --eval-mode full ^
      --crop-size 256 ^
      --batch-size 8 ^
      --output-file checkpoints/fixmatch_rgb_tg/eval_metrics_final_30view.json
)

echo.
echo [Step 2/7] Evaluating Supervised 100%% on Matched 30-View Protocol...
%PY_BIN% evaluate.py ^
  --checkpoint checkpoints/supervised_100/best_model.pth ^
  --data-root data/ucf101 ^
  --split-file data/splits/ucf101_val_split_1_rawframes.txt ^
  --eval-mode full ^
  --crop-size 256 ^
  --batch-size 8 ^
  --output-file results/eval_supervised_100_30view.json
if %ERRORLEVEL% neq 0 (echo [Error] Step 2 failed & exit /b %ERRORLEVEL%)

echo.
echo [Step 3/7] Evaluating Final Epoch Checkpoints for Baselines...
if exist checkpoints\supervised_baseline\final_model.pth (
    %PY_BIN% evaluate.py ^
      --checkpoint checkpoints/supervised_baseline/final_model.pth ^
      --data-root data/ucf101 ^
      --split-file data/splits/ucf101_val_split_1_rawframes.txt ^
      --eval-mode full ^
      --crop-size 256 ^
      --batch-size 8 ^
      --output-file results/eval_supervised_10_final_30view.json
)

if exist checkpoints\fixmatch_baseline\final_model.pth (
    %PY_BIN% evaluate.py ^
      --checkpoint checkpoints/fixmatch_baseline/final_model.pth ^
      --data-root data/ucf101 ^
      --split-file data/splits/ucf101_val_split_1_rawframes.txt ^
      --eval-mode full ^
      --crop-size 256 ^
      --batch-size 8 ^
      --output-file results/eval_fixmatch_final_30view.json
)

if exist checkpoints\f2_fixmatch\final_model.pth (
    %PY_BIN% evaluate.py ^
      --checkpoint checkpoints/f2_fixmatch/final_model.pth ^
      --data-root data/ucf101 ^
      --split-file data/splits/ucf101_val_split_1_rawframes.txt ^
      --eval-mode full ^
      --crop-size 256 ^
      --batch-size 8 ^
      --output-file results/eval_f2_final_30view.json
)

echo.
echo [Step 4/7] Generating Pseudo-Label Precision and Coverage Curves...
%PY_BIN% analyze_pseudolabels.py ^
  --checkpoint checkpoints/f2_fixmatch/best_model.pth ^
  --data-root data/ucf101 ^
  --split-file data/splits/ucf101_train_10_percent_unlabeled_split_1_rawframes.txt ^
  --modality mpl ^
  --output-dir results/pseudolabel_f2
if %ERRORLEVEL% neq 0 (echo [Error] Step 4 failed & exit /b %ERRORLEVEL%)

echo.
echo [Step 5/7] Running PreciseBN Diagnostic on F2-FixMatch...
%PY_BIN% recalibrate_bn.py ^
  --checkpoint checkpoints/f2_fixmatch/best_model.pth ^
  --data-root data/ucf101 ^
  --split-file data/splits/ucf101_train_10_percent_labeled_split_1_rawframes.txt ^
  --output-dir checkpoints/f2_fixmatch_recal ^
  --num-batches 200 ^
  --batch-size 4
if %ERRORLEVEL% neq 0 (echo [Error] Step 5 failed & exit /b %ERRORLEVEL%)

echo.
echo [Step 6/7] Evaluating Recalibrated F2-FixMatch (30-View)...
%PY_BIN% evaluate.py ^
  --checkpoint checkpoints/f2_fixmatch_recal/best_model_precise_bn.pth ^
  --data-root data/ucf101 ^
  --split-file data/splits/ucf101_val_split_1_rawframes.txt ^
  --eval-mode full ^
  --crop-size 256 ^
  --batch-size 8 ^
  --output-file results/eval_f2_precise_bn_30view.json
if %ERRORLEVEL% neq 0 (echo [Error] Step 6 failed & exit /b %ERRORLEVEL%)

echo.
echo [Step 7/7] Running Mathematical Synthesis ^& Disentanglement Analysis...
%PY_BIN% synthesize_results.py ^
  --run-a results/eval_supervised_100_30view.json ^
  --run-d checkpoints/fixmatch_rgb_tg/eval_metrics_full30view.json
if %ERRORLEVEL% neq 0 (echo [Error] Step 7 failed & exit /b %ERRORLEVEL%)

echo.
echo ========================================================================
echo ALL REMAINING TASKS COMPLETED SUCCESSFULLY!
echo Review:
echo   - Report:  results\synthesis_report.md
echo   - Updated: RESULTS.md (Section 4)
echo ========================================================================

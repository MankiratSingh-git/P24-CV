@echo off
rem ==============================================================================
rem Script: scripts\eval_tau_095.bat
rem Purpose: Evaluate FixMatch trained with tau=0.95 on the matched 30-view protocol.
rem ==============================================================================

if not exist checkpoints\fixmatch_tau_095 mkdir checkpoints\fixmatch_tau_095

echo ========================================================================
echo Evaluating FixMatch (tau=0.95) on Matched 30-View Protocol
echo ========================================================================

if exist checkpoints\fixmatch_tau_095\best_model.pth (
    echo.
    echo [1/2] Evaluating FixMatch (tau=0.95) - Best Model (30-View)...
    python evaluate.py ^
      --checkpoint checkpoints\fixmatch_tau_095\best_model.pth ^
      --data-root data\ucf101 ^
      --split-file data\splits\ucf101_val_split_1_rawframes.txt ^
      --eval-mode full ^
      --crop-size 256 ^
      --batch-size 8 ^
      --output-file checkpoints\fixmatch_tau_095\eval_metrics_full30view.json
) else (
    echo [Warning] checkpoints\fixmatch_tau_095\best_model.pth not found!
)

if exist checkpoints\fixmatch_tau_095\final_model.pth (
    echo.
    echo [2/2] Evaluating FixMatch (tau=0.95) - Final Epoch Model (30-View)...
    python evaluate.py ^
      --checkpoint checkpoints\fixmatch_tau_095\final_model.pth ^
      --data-root data\ucf101 ^
      --split-file data\splits\ucf101_val_split_1_rawframes.txt ^
      --eval-mode full ^
      --crop-size 256 ^
      --batch-size 8 ^
      --output-file checkpoints\fixmatch_tau_095\eval_metrics_final_30view.json
)

echo.
echo ========================================================================
echo FixMatch (tau=0.95) Evaluation Complete!
echo ========================================================================

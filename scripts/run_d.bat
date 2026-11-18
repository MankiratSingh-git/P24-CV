@echo off
rem ==============================================================================
rem Run D: Dual-Stream FixMatch Control (RGB + TG, beta=0.0, gamma=0.0)
rem Platform: Windows
rem Paper: "Leveraging Frame- and Feature-level Progressive Augmentation for
rem         Semi-supervised Action Recognition" (Tu et al., ACM TOMM 2025)
rem ==============================================================================
setlocal enabledelayedexpansion

rem Change directory to project root (parent directory of scripts/)
cd /d "%~dp0\.."

rem Default parameters (can be pre-set via environment variables)
if "%PYTHON%"=="" set PYTHON=python
if "%DATA_ROOT%"=="" set DATA_ROOT=data\ucf101
if "%SPLITS_DIR%"=="" set SPLITS_DIR=data\splits
if "%OUTPUT_DIR%"=="" set OUTPUT_DIR=checkpoints\fixmatch_rgb_tg
if "%SPLIT%"=="" set SPLIT=1
if "%LABELED_PERCENT%"=="" set LABELED_PERCENT=10
if "%BATCH_SIZE%"=="" set BATCH_SIZE=4
if "%ACCUM_STEPS%"=="" set ACCUM_STEPS=2
if "%MU%"=="" set MU=4
if "%EPOCHS%"=="" set EPOCHS=220
if "%WARMUP_EPOCHS%"=="" set WARMUP_EPOCHS=40
if "%RAMPUP_EPOCHS%"=="" set RAMPUP_EPOCHS=30
if "%LR%"=="" set LR=0.2
if "%THRESHOLD%"=="" set THRESHOLD=0.3
if "%ALPHA%"=="" set ALPHA=2.0
if "%BETA%"=="" set BETA=0.0
if "%GAMMA%"=="" set GAMMA=0.0
if "%SEED%"=="" set SEED=42
if "%NUM_WORKERS%"=="" set NUM_WORKERS=4
if "%EVAL_FREQ%"=="" set EVAL_FREQ=5

echo ======================================================================
echo Run D: Dual-Stream FixMatch Control (RGB + TG)
echo Platform:    Windows
echo Python:      %PYTHON%
echo Data Root:   %DATA_ROOT%
echo Splits Dir:  %SPLITS_DIR%
echo Output Dir:  %OUTPUT_DIR%
echo Hyperparameters:
echo   Split: %SPLIT% ^| Labeled %%: %LABELED_PERCENT%%% ^| Seed: %SEED%
echo   Batch Size: B_l=%BATCH_SIZE%, mu=%MU% ^| Accum Steps: %ACCUM_STEPS%
echo   Epochs: %EPOCHS% ^| Warmup: %WARMUP_EPOCHS% ^| Rampup: %RAMPUP_EPOCHS%
echo   LR: %LR% ^| Threshold tau: %THRESHOLD%
echo   Loss Weights: alpha=%ALPHA%, beta=%BETA%, gamma=%GAMMA%
echo ======================================================================

if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

"%PYTHON%" train_f2_fixmatch.py ^
    --data-root "%DATA_ROOT%" ^
    --splits-dir "%SPLITS_DIR%" ^
    --split %SPLIT% ^
    --labeled-percent %LABELED_PERCENT% ^
    --batch-size %BATCH_SIZE% ^
    --accum-steps %ACCUM_STEPS% ^
    --mu %MU% ^
    --epochs %EPOCHS% ^
    --warmup-epochs %WARMUP_EPOCHS% ^
    --rampup-epochs %RAMPUP_EPOCHS% ^
    --lr %LR% ^
    --threshold %THRESHOLD% ^
    --alpha %ALPHA% ^
    --beta %BETA% ^
    --gamma %GAMMA% ^
    --seed %SEED% ^
    --num-workers %NUM_WORKERS% ^
    --eval-freq %EVAL_FREQ% ^
    --output-dir "%OUTPUT_DIR%" ^
    %*

if %ERRORLEVEL% neq 0 (
    echo ======================================================================
    echo [ERROR] Run D training failed with exit code %ERRORLEVEL%.
    echo ======================================================================
    exit /b %ERRORLEVEL%
)

echo ======================================================================
echo [SUCCESS] Run D training completed successfully.
echo Artifacts saved in %OUTPUT_DIR%
echo ======================================================================

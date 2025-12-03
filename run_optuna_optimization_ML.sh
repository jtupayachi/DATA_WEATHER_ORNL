#!/bin/bash
# ==================== Bayesian Hyperparameter Optimization Script - ML Models ====================
# This script runs Optuna-based Bayesian optimization for LightGBM and XGBoost models
# Optimizes for MCC (Matthews Correlation Coefficient)
#
# Usage:
#   ./run_optuna_optimization_ML.sh                    # Run with defaults (LightGBM, 50 trials)
#   ./run_optuna_optimization_ML.sh --model xgboost   # Run XGBoost only
#   ./run_optuna_optimization_ML.sh --model both      # Run both LightGBM and XGBoost
#   ./run_optuna_optimization_ML.sh --n_trials 100 --gpu  # Custom settings with GPU
#
# ====================================================================================

set -e  # Exit on error

# ==================== DEFAULT CONFIGURATION ====================
MODEL="both"           # Options: lightgbm, xgboost, both
N_TRIALS=50            # Number of Optuna trials (Bayesian optimization iterations)
N_SPLITS=3             # Cross-validation splits
HORIZON="6hours"       # Forecast horizon
USE_GPU=""             # GPU flag (empty = CPU, --gpu = GPU)
OUTPUT_DIR=""          # Output directory (auto-generated if empty)

# ==================== PARSE COMMAND LINE ARGUMENTS ====================
while [[ $# -gt 0 ]]; do
    case $1 in
        --model)
            MODEL="$2"
            shift 2
            ;;
        --n_trials)
            N_TRIALS="$2"
            shift 2
            ;;
        --n_splits)
            N_SPLITS="$2"
            shift 2
            ;;
        --horizon)
            HORIZON="$2"
            shift 2
            ;;
        --gpu)
            USE_GPU="--gpu"
            shift
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model MODEL       Model type: lightgbm, xgboost, or both (default: both)"
            echo "  --n_trials N        Number of Optuna trials (default: 50)"
            echo "  --n_splits N        CV splits (default: 3)"
            echo "  --horizon HORIZON   Forecast horizon: 15min, 30min, 1hour, 3hours, 6hours, 12hours, 24hours (default: 6hours)"
            echo "  --gpu               Use GPU for training (LightGBM and XGBoost GPU support)"
            echo "  --output DIR        Output directory (default: auto-generated)"
            echo "  --help, -h          Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 --model xgboost --n_trials 100 --gpu"
            exit 0
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use --help for usage information"
            exit 1
            ;;
    esac
done

# ==================== SCRIPT DIRECTORY ====================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ==================== CHECK DEPENDENCIES ====================
echo "=========================================="
echo "Checking dependencies..."
echo "=========================================="

# Check Python
if ! command -v python3 &> /dev/null; then
    echo "❌ Python3 not found!"
    exit 1
fi
echo "✅ Python3: $(python3 --version)"

# Check required packages
python3 -c "import optuna" 2>/dev/null || {
    echo "⚠️  Optuna not found. Installing..."
    pip install optuna
}

python3 -c "import lightgbm" 2>/dev/null || {
    echo "❌ LightGBM not found! Please install with: pip install lightgbm"
    exit 1
}

python3 -c "import xgboost" 2>/dev/null || {
    echo "❌ XGBoost not found! Please install with: pip install xgboost"
    exit 1
}

echo "✅ All dependencies satisfied"

# ==================== GPU CHECK ====================
if [ -n "$USE_GPU" ]; then
    echo ""
    echo "=========================================="
    echo "GPU Configuration"
    echo "=========================================="
    python3 -c "
import lightgbm as lgb
import xgboost as xgb
print('LightGBM version:', lgb.__version__)
print('XGBoost version:', xgb.__version__)
# Note: GPU support depends on how the packages were compiled
"
fi

# ==================== DISPLAY CONFIGURATION ====================
echo ""
echo "=========================================="
echo "Optimization Configuration (ML Models)"
echo "=========================================="
echo "Model type:        $MODEL"
echo "Number of trials:  $N_TRIALS"
echo "CV splits:         $N_SPLITS"
echo "Forecast horizon:  $HORIZON"
if [ -n "$USE_GPU" ]; then
    echo "Use GPU:           Yes"
else
    echo "Use GPU:           No"
fi
if [ -n "$OUTPUT_DIR" ]; then
    echo "Output directory:  $OUTPUT_DIR"
else
    echo "Output directory:  (auto-generated)"
fi
echo "=========================================="
echo ""

# ==================== SEARCH SPACE INFO ====================
echo "LightGBM Search Space:"
echo "  - num_leaves: [16, 128]"
echo "  - max_depth: [3, 12]"
echo "  - learning_rate: [0.01, 0.3]"
echo "  - n_estimators: [100, 300, 500, 1000, 1500]"
echo "  - feature_fraction: [0.5, 1.0]"
echo "  - bagging_fraction: [0.5, 1.0]"
echo "  - reg_alpha/lambda: [1e-8, 10.0]"
echo ""
echo "XGBoost Search Space:"
echo "  - max_depth: [3, 12]"
echo "  - learning_rate: [0.01, 0.3]"
echo "  - n_estimators: [100, 300, 500, 1000, 1500]"
echo "  - subsample: [0.5, 1.0]"
echo "  - colsample_bytree: [0.5, 1.0]"
echo "  - reg_alpha/lambda: [1e-8, 10.0]"
echo "  - gamma: [0.0, 1.0]"
echo ""
echo "Feature Engineering Search Space:"
echo "  - lag_config: [short, medium, long]"
echo "  - use_rolling: [True, False]"
echo "  - rolling_windows: [[4,12], [4,12,24], [4,12,24,96]]"
echo ""

# ==================== CONFIRMATION ====================
read -p "Start optimization? [Y/n] " -n 1 -r
echo
if [[ $REPLY =~ ^[Nn]$ ]]; then
    echo "Aborted."
    exit 0
fi

# ==================== RUN OPTIMIZATION ====================
echo ""
echo "=========================================="
echo "Starting Bayesian Optimization (ML Models)..."
echo "=========================================="
echo "Start time: $(date)"
echo ""

# Build command
CMD="python3 JT_Extreme_event_labeler_ML_optuna.py"
CMD="$CMD --model $MODEL"
CMD="$CMD --n_trials $N_TRIALS"
CMD="$CMD --n_splits $N_SPLITS"
CMD="$CMD --horizon $HORIZON"

if [ -n "$USE_GPU" ]; then
    CMD="$CMD --gpu"
fi

if [ -n "$OUTPUT_DIR" ]; then
    CMD="$CMD --output $OUTPUT_DIR"
fi

# Run with logging
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="optuna_ml_optimization_${MODEL}_${TIMESTAMP}.log"

echo "Command: $CMD"
echo "Log file: $LOG_FILE"
echo ""

# Run and tee to log file
$CMD 2>&1 | tee "$LOG_FILE"

# ==================== COMPLETION ====================
echo ""
echo "=========================================="
echo "✅ Optimization Complete!"
echo "=========================================="
echo "End time: $(date)"
echo "Log file: $LOG_FILE"
echo ""

# Find and display results directory
RESULTS_DIR=$(ls -td optuna_ml_results_* 2>/dev/null | head -1)
if [ -n "$RESULTS_DIR" ]; then
    echo "Results saved to: $RESULTS_DIR"
    echo ""
    echo "Files:"
    ls -la "$RESULTS_DIR"
fi

echo ""
echo "=========================================="
echo "Next Steps:"
echo "=========================================="
echo "1. Review best parameters: cat $RESULTS_DIR/best_params.json"
echo "2. View optimization results: cat $RESULTS_DIR/optimization_results.csv"
echo "3. Run training with best params using the main ML script"
echo ""

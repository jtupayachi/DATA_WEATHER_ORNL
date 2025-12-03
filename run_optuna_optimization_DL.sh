#!/bin/bash
# ==================== Bayesian Hyperparameter Optimization Script ====================
# This script runs Optuna-based Bayesian optimization for GRU/LSTM models
# Optimizes for MCC (Matthews Correlation Coefficient)
#
# Usage:
#   ./run_optuna_optimization.sh              # Run with defaults (GRU, 50 trials)
#   ./run_optuna_optimization.sh --model lstm # Run LSTM only
#   ./run_optuna_optimization.sh --model both # Run both GRU and LSTM
#   ./run_optuna_optimization.sh --n_trials 100 --gpu 1  # Custom settings
#
# ====================================================================================

set -e  # Exit on error

# ==================== DEFAULT CONFIGURATION ====================
MODEL="both"           # Options: gru, lstm, both
N_TRIALS=50            # Number of Optuna trials (Bayesian optimization iterations)
N_SPLITS=3             # Cross-validation splits
EPOCHS=100             # Max epochs per trial
PATIENCE=15            # Early stopping patience
HORIZON="6hours"       # Forecast horizon
GPU=0                  # GPU device ID
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
        --epochs)
            EPOCHS="$2"
            shift 2
            ;;
        --patience)
            PATIENCE="$2"
            shift 2
            ;;
        --horizon)
            HORIZON="$2"
            shift 2
            ;;
        --gpu)
            GPU="$2"
            shift 2
            ;;
        --output)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --help|-h)
            echo "Usage: $0 [OPTIONS]"
            echo ""
            echo "Options:"
            echo "  --model MODEL       Model type: gru, lstm, or both (default: both)"
            echo "  --n_trials N        Number of Optuna trials (default: 50)"
            echo "  --n_splits N        CV splits (default: 3)"
            echo "  --epochs N          Max epochs per trial (default: 100)"
            echo "  --patience N        Early stopping patience (default: 15)"
            echo "  --horizon HORIZON   Forecast horizon: 15min, 30min, 1hour, 3hours, 6hours, 12hours, 24hours (default: 6hours)"
            echo "  --gpu ID            GPU device ID (default: 0)"
            echo "  --output DIR        Output directory (default: auto-generated)"
            echo "  --help, -h          Show this help message"
            echo ""
            echo "Example:"
            echo "  $0 --model lstm --n_trials 100 --gpu 0"
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

python3 -c "import torch" 2>/dev/null || {
    echo "❌ PyTorch not found! Please install PyTorch first."
    exit 1
}

echo "✅ All dependencies satisfied"

# ==================== GPU CHECK ====================
echo ""
echo "=========================================="
echo "GPU Configuration"
echo "=========================================="
python3 -c "
import torch
if torch.cuda.is_available():
    print(f'✅ CUDA available: {torch.cuda.device_count()} GPU(s)')
    for i in range(torch.cuda.device_count()):
        print(f'   GPU {i}: {torch.cuda.get_device_name(i)}')
else:
    print('⚠️  CUDA not available, using CPU')
"

# ==================== DISPLAY CONFIGURATION ====================
echo ""
echo "=========================================="
echo "Optimization Configuration"
echo "=========================================="
echo "Model type:        $MODEL"
echo "Number of trials:  $N_TRIALS"
echo "CV splits:         $N_SPLITS"
echo "Max epochs:        $EPOCHS"
echo "Early stopping:    $PATIENCE"
echo "Forecast horizon:  $HORIZON"
echo "GPU device:        $GPU"
if [ -n "$OUTPUT_DIR" ]; then
    echo "Output directory:  $OUTPUT_DIR"
else
    echo "Output directory:  (auto-generated)"
fi
echo "=========================================="
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
echo "Starting Bayesian Optimization..."
echo "=========================================="
echo "Start time: $(date)"
echo ""

# Build command
CMD="python3 JT_Extreme_event_labeler_DL_optuna.py"
CMD="$CMD --model $MODEL"
CMD="$CMD --n_trials $N_TRIALS"
CMD="$CMD --n_splits $N_SPLITS"
CMD="$CMD --epochs $EPOCHS"
CMD="$CMD --patience $PATIENCE"
CMD="$CMD --horizon $HORIZON"
CMD="$CMD --gpu $GPU"

if [ -n "$OUTPUT_DIR" ]; then
    CMD="$CMD --output $OUTPUT_DIR"
fi

# Run with logging
TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_FILE="optuna_optimization_${MODEL}_${TIMESTAMP}.log"

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
RESULTS_DIR=$(ls -td optuna_results_* 2>/dev/null | head -1)
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
echo "3. Run training with best params using the main DL script"
echo ""

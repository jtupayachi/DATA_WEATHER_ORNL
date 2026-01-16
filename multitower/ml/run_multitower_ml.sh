#!/bin/bash

# Multi-Tower ML Experiment Runner
# This script runs the multi-tower ML approach with logging

echo "============================================================"
echo "MULTI-TOWER ML EXPERIMENT"
echo "============================================================"
echo "Approach: ONE model per event, trained on ALL towers"
echo "Start time: $(date)"
echo "============================================================"
echo ""

# Activate conda environment
echo "Activating conda environment: ml_env2"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate ml_env2

if [ $? -ne 0 ]; then
    echo "❌ Failed to activate conda environment ml_env2"
    exit 1
fi

echo "✓ Environment activated: $CONDA_DEFAULT_ENV"
echo ""

# Create logs directory if it doesn't exist
mkdir -p logs

# Generate timestamp for log files
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")
LOG_FILE="logs/multitower_ml_${TIMESTAMP}.log"
ERROR_FILE="logs/multitower_ml_${TIMESTAMP}.err"

echo "Logging to: $LOG_FILE"
echo ""

# Run the Python script
python JT_Extreme_event_labeler_ML_multitower.py > "$LOG_FILE" 2> "$ERROR_FILE"

# Check exit status
if [ $? -eq 0 ]; then
    echo ""
    echo "============================================================"
    echo "✅ EXPERIMENT COMPLETED SUCCESSFULLY"
    echo "============================================================"
    echo "End time: $(date)"
    echo ""
    echo "Results:"
    echo "  - Output log: $LOG_FILE"
    echo "  - Error log: $ERROR_FILE"
    echo ""
    
    # Find the most recent results directory
    RESULTS_DIR=$(ls -td multitower_results_* 2>/dev/null | head -1)
    
    if [ -n "$RESULTS_DIR" ]; then
        echo "  - Results directory: $RESULTS_DIR"
        echo ""
        echo "Summary report:"
        echo "-----------------------------------------------------------"
        tail -n 50 "$RESULTS_DIR/experiment_report.txt"
        echo "-----------------------------------------------------------"
    fi
else
    echo ""
    echo "============================================================"
    echo "❌ EXPERIMENT FAILED"
    echo "============================================================"
    echo "Check error log: $ERROR_FILE"
    echo ""
    tail -n 20 "$ERROR_FILE"
fi

echo ""
echo "============================================================"

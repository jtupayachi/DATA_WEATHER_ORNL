# Multi-Tower Machine Learning Approach

## Overview

This folder contains a **multi-tower ML approach** where **ONE model** is trained for **ALL towers** simultaneously, rather than training separate models per tower.

## Key Innovation

### Single-Tower Approach (Original):
- Train 18 models: 6 towers × 3 events = 18 separate models
- Each model learns only from its own tower's data
- Problem: Sparse-feature towers (TOWF: 12 features, TOWS: 7 features) struggle
- No knowledge transfer between towers

### Multi-Tower Approach (This Folder):
- Train 3 models: 1 model per event (across all towers)
- Each model learns from ALL towers simultaneously
- Tower identity encoded as categorical feature (one-hot encoded)
- Benefits:
  - **Shared learning**: Data-rich towers (TOWD, TOWY) help sparse towers (TOWF, TOWS)
  - **More training data**: ~900K samples instead of ~150K per tower
  - **Transfer learning**: Patterns learned from one tower benefit others
  - **Spatial context**: Model can learn tower-specific biases

## Files

- `JT_Extreme_event_labeler_ML_multitower.py` - Main training script
- `run_multitower_ml.sh` - Shell script to run experiments
- `README.md` - This file

## Methodology

### Data Preparation
1. **Combine all towers** into single dataset
2. **Add tower_id** as categorical feature (one-hot encoded)
3. **Feature engineering**: Same as single-tower (lags, rolling windows)
4. **Result**: Each sample has:
   - Weather features (lags + rolling)
   - Tower identity (6 binary features: tower_TOWA, tower_TOWB, ...)

### Model Architecture
```
Input Features:
  - Weather lags (e.g., 48 timesteps × N features)
  - Rolling statistics (mean, std over windows)
  - Tower one-hot encoding (6 binary features)
  ↓
XGBoost / LightGBM
  ↓
Output: Event probability [0, 1]
```

### Training Strategy
- **Time Series Split**: 80% train, 20% test
- **5-fold Cross-Validation** on training set
- **Class weights**: Handle imbalanced events
- **Early stopping**: Prevent overfitting

### Evaluation
- Overall metrics (across all towers)
- **Per-tower metrics** on test set
- Compare to single-tower baseline

## Expected Benefits

### Towers Expected to Improve Most:
1. **TOWF** (12 features, currently poor performance)
   - Single-tower E3 F1=0.43 → Expected multi-tower F1=0.55+
   - Single-tower E4 F1=0.15 → Expected multi-tower F1=0.20+

2. **TOWS** (7 features, struggles with E5)
   - Single-tower E5 F1=0.13 → Expected multi-tower F1=0.25+

### Towers Expected to Remain Strong:
- **TOWD** (28 features) - Already performs well, provides knowledge to others
- **TOWY** (18 features) - Strong baseline, helps other towers

## Running the Experiments

### Option 1: Python Script
```bash
cd /home/jose/DATA_WEATHER_ORNL/multitower/ml
python JT_Extreme_event_labeler_ML_multitower.py
```

### Option 2: Shell Script (with logging)
```bash
cd /home/jose/DATA_WEATHER_ORNL/multitower/ml
bash run_multitower_ml.sh
```

## Output

Results saved to: `ml_results_YYYYMMDD_HHMMSS/`

### Directory Structure:
```
ml_results_YYYYMMDD_HHMMSS/
├── config.json                          # Experiment configuration
├── experiment_report.txt                 # Human-readable results
└── models/
    ├── event_E3_LowTemp_lt0_lightgbm_final.pkl
    ├── event_E3_LowTemp_lt0_xgboost_final.pkl
    ├── event_E4_HighWind_Peak_gt25_lightgbm_final.pkl
    ├── event_E4_HighWind_Peak_gt25_xgboost_final.pkl
    ├── event_E5_LowWind_lt2_lightgbm_final.pkl
    └── event_E5_LowWind_lt2_xgboost_final.pkl
```

### Report Contents:
- Cross-validation metrics (5-fold average ± std)
- Final test set metrics (overall)
- **Per-tower test metrics** (to compare with single-tower baseline)

## Comparison to Single-Tower

After running, compare:

### Single-Tower Results:
```
TOWF E3: XGBoost F1=0.4294, MCC=0.2986
TOWS E5: XGBoost F1=0.1282, MCC=0.1204
```

### Multi-Tower Results:
```
Event E3 (XGBoost):
  Overall: F1=0.62, MCC=0.48
  TOWF: F1=0.55, MCC=0.42  ← Expected improvement
  TOWS: F1=0.76, MCC=0.72  ← Should remain strong

Event E5 (XGBoost):
  Overall: F1=0.35, MCC=0.28
  TOWS: F1=0.25, MCC=0.22  ← Expected improvement
```

## Decision Criteria

### If Multi-Tower > Single-Tower by >5%:
✅ **Proceed to multi-tower DL/CNN**
- Skip single-tower DL experiments
- Implement multi-tower LSTM/CNN/GNN with tower embeddings
- Use same pooled-tower approach

### If Multi-Tower ≈ Single-Tower (±2%):
⚠️ **Towers too heterogeneous**
- Stick with single-tower models
- Focus on better feature engineering per tower
- Don't waste time on multi-tower DL

## Next Steps

1. **Run this experiment** (30 minutes on CPU)
2. **Compare per-tower metrics** to single-tower baseline
3. **If successful**: Design multi-tower LSTM/CNN architecture
4. **If not**: Improve single-tower feature engineering

## Notes

- Multi-tower approach works best when towers share similar climatology
- Tower embeddings allow model to learn location-specific patterns
- Requires sufficient data per tower (>10K samples recommended)
- May need to adjust class weights for very imbalanced events

## Contact

For questions or issues, refer to main project documentation.

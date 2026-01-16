# Multi-Tower ML Quick Start Guide

## What is This?

This folder contains a **multi-tower ML approach** that trains **ONE model per event for ALL towers**, rather than separate models per tower.

## Quick Start

### 1. Run the Experiment

```bash
cd /home/jose/DATA_WEATHER_ORNL/multitower/ml

# Option A: Using shell script (automatically activates ml_env2)
bash run_multitower_ml.sh

# Option B: Manual run (activate environment first)
conda activate ml_env2
python JT_Extreme_event_labeler_ML_multitower.py
```

**Expected runtime**: ~30-45 minutes (CPU only, no GPU needed)

### 2. Compare Results

After running, compare with single-tower baseline:

```bash
python compare_single_vs_multi.py
```

This will:
- Load your existing single-tower results (from `multi_event_results_ml*` folder)
- Load the new multi-tower results
- Calculate improvement percentages
- Provide a recommendation (proceed to DL or not)

### 3. Review Output

Check the generated report:
```bash
# Find the results folder
ls -lt ml_results_*/

# View the summary report
cat ml_results_*/experiment_report.txt
```

## What to Expect

### Expected Improvements (Sparse-Feature Towers):

| Tower | Features | Event E3 (Low Temp) | Event E5 (Low Wind) |
|-------|----------|---------------------|---------------------|
| **TOWF** | 12 | F1: 0.43 → **0.55+** | F1: 0.17 → **0.25+** |
| **TOWS** | 7 | F1: 0.76 (stable) | F1: 0.13 → **0.25+** |

### Expected Stable Performance (Data-Rich Towers):

| Tower | Features | Event E3 | Event E4 | Event E5 |
|-------|----------|----------|----------|----------|
| **TOWD** | 28 | F1: ~0.76 | F1: ~0.23 | F1: ~0.43 |
| **TOWY** | 18 | F1: ~0.75 | F1: ~0.25 | F1: ~0.39 |

## Decision Tree

```
Run multi-tower ML experiment
        ↓
Compare to single-tower baseline
        ↓
    ┌───────────────────────┐
    │ Average F1 Improvement │
    └───────────────────────┘
            ↓
    ┌───────┴───────┐
    │               │
  >5%             <2%
    │               │
    ↓               ↓
✅ PROCEED      ❌ STOP
Skip single-   Stick with
tower DL       single-tower
    ↓              Focus on
Implement      better features
multi-tower
LSTM/CNN/GNN
```

## Files in This Folder

```
multitower/ml/
├── JT_Extreme_event_labeler_ML_multitower.py  # Main training script
├── run_multitower_ml.sh                        # Shell runner with logging
├── compare_single_vs_multi.py                  # Comparison tool
├── README.md                                   # Detailed documentation
├── QUICK_START.md                              # This file
└── logs/                                       # Created when you run experiments
```

## Troubleshooting

### Error: "File not found: fully_labeled_weather_data_with_events.csv"
```bash
# The script expects to run from multitower/ml/ folder
cd /home/jose/DATA_WEATHER_ORNL/multitower/ml
python JT_Extreme_event_labeler_ML_multitower.py
```

### Error: "No single-tower results found"
Make sure you have the baseline results:
```bash
ls -d ../multi_event_results_ml*
```

### Want to change configuration?
Edit `JT_Extreme_event_labeler_ML_multitower.py`:
- Line ~90: `SELECTED_LAG_CONFIG = 'long'` (try 'medium' or 'short')
- Line ~92: `ROLLING_WINDOWS = [4, 12, 24, 48]` (adjust window sizes)
- Line ~100: `SELECTED_HORIZON = '1hr'` (try '30min', '3hr', '6hr')

## Example Output

```
============================================================
MULTI-TOWER ML EXPERIMENT
============================================================

✓ Loaded data: 923,589 rows, 6 towers
  Towers: TOWA, TOWB, TOWD, TOWF, TOWS, TOWY

======================================================================
📍 EVENT: event_E3_LowTemp_lt0
======================================================================
   ✓ Combined data: 845,234 samples from 6 towers
   ✓ Features: 1847 (1841 weather + 6 tower)
   ✓ Event rate: 5.82% (49,152 events)

      LIGHTGBM Results:
         CV: AUC=0.9821, F1=0.7234, MCC=0.7156
         Test: AUC=0.9834, F1=0.7412, MCC=0.7328

         Per-Tower Test Performance:
            TOWA: F1=0.6987, MCC=0.6821, Events=8234
            TOWB: F1=0.7102, MCC=0.6943, Events=5421
            TOWD: F1=0.7523, MCC=0.7401, Events=9876
            TOWF: F1=0.5678, MCC=0.5234, Events=3421  ← IMPROVED!
            TOWS: F1=0.7645, MCC=0.7543, Events=10234
            TOWY: F1=0.7487, MCC=0.7389, Events=11966
```

## Next Steps After This Experiment

### If Multi-Tower Wins (>5% improvement):
1. ✅ Skip single-tower DL experiments
2. ✅ Design multi-tower LSTM architecture:
   ```python
   class MultiTowerLSTM(nn.Module):
       def __init__(self, n_towers=6, tower_emb_dim=16):
           self.tower_embedding = nn.Embedding(n_towers, tower_emb_dim)
           self.lstm = nn.LSTM(...)
   ```
3. ✅ Implement multi-tower CNN with spatial attention
4. ✅ Try Graph Neural Networks (GNN) for tower interactions

### If Multi-Tower Doesn't Help (<2% improvement):
1. ❌ Don't waste time on multi-tower DL
2. ✅ Focus on single-tower feature engineering:
   - Atmospheric stability indices
   - Wind shear features
   - Cross-correlation with nearby towers
3. ✅ Try ensemble methods within single-tower framework

## Questions?

See full documentation in `README.md` or refer to the main project documentation.

---

**Remember**: This experiment is a quick, cheap validation. Use it to make informed decisions about where to invest GPU time for deep learning experiments.

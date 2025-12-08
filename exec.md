# 🚀 Hyperparameter Optimization for Weather Event Prediction

This project uses **Optuna (Bayesian Optimization)** to find optimal hyperparameters for deep learning models predicting extreme weather events.

## 📋 Overview

| Script | Models | Description |
|--------|--------|-------------|
| `JT_Extreme_event_labeler_DL_optuna.py` | GRU, LSTM | Recurrent neural networks |
| `JT_Extreme_event_labeler_CNN_optuna.py` | CNN, TCN | Convolutional / Temporal networks |
| `JT_Extreme_event_labeler_ML_optuna.py` | LightGBM, XGBoost | Gradient boosting methods |

## ⚡ Quick Start

### Run All Models in Parallel (Multi-GPU)

```bash
# Terminal 1 - TCN on GPU 0
python JT_Extreme_event_labeler_CNN_optuna.py --gpu 0 --model tcn 

# Terminal 2 - GRU on GPU 1
python JT_Extreme_event_labeler_DL_optuna.py --gpu 1 --model gru 

# Terminal 3 - CNN on GPU 0 (after TCN finishes, or if enough VRAM)
python JT_Extreme_event_labeler_CNN_optuna.py --gpu 0 --model cnn 

# Terminal 4 - LSTM on GPU 1 (after GRU finishes, or if enough VRAM)
python JT_Extreme_event_labeler_DL_optuna.py --gpu 1 --model lstm 
```
```bash
# Teminal 5 - python ; prior replace OPTUNA_BEST_PARAMS_XGBOOST and OPTUNA_BEST_PARAMS_LIGHTGBM  params

JT_Extreme_event_labeler_ML_optuna.py && JT_Extreme_event_labeler_ML.py
```
### Run All Models Sequentially (Single GPU)



## 🎛️ Command Line Arguments

### Deep Learning (GRU/LSTM)

```bash
python JT_Extreme_event_labeler_DL_optuna.py [OPTIONS]

Options:
  --model {gru,lstm,both}    Model type (default: gru)
  --gpu GPU_ID               GPU device ID (default: 0)
  --n_trials N               Optuna trials per experiment (default: 3)
  --n_splits N               Cross-validation folds (default: 3)
  --epochs N                 Max epochs per trial (default: 50)
  --patience N               Early stopping patience (default: 8)
  --horizon HORIZON          Forecast horizon (default: 6hours)
  --parallel N               Parallel experiments (default: auto)
  --output DIR               Output directory
```

### CNN/TCN

```bash
python JT_Extreme_event_labeler_CNN_optuna.py [OPTIONS]

Options:
  --model {cnn,tcn,both}     Model type (default: cnn)
  --gpu GPU_ID               GPU device ID (default: 0)
  --n_trials N               Optuna trials per experiment (default: 3)
  --n_splits N               Cross-validation folds (default: 3)
  --epochs N                 Max epochs per trial (default: 50)
  --patience N               Early stopping patience (default: 8)
  --horizon HORIZON          Forecast horizon (default: 6hours)
  --parallel N               Parallel experiments (default: auto)
  --output DIR               Output directory
```

### Machine Learning (LightGBM/XGBoost)

```bash
python JT_Extreme_event_labeler_ML_optuna.py [OPTIONS]

Options:
  --model {lgbm,xgb,both}    Model type (default: lgbm)
  --n_trials N               Optuna trials per experiment (default: 50)
  --n_splits N               Cross-validation folds (default: 5)
  --horizon HORIZON          Forecast horizon (default: 6hours)
  --output DIR               Output directory
```

## 📊 Forecast Horizons

| Horizon | Time Steps | Use Case |
|---------|------------|----------|
| `15min` | 1 | Near real-time alerts |
| `30min` | 2 | Short-term warnings |
| `1hour` | 4 | Operational planning |
| `3hours` | 12 | Medium-term forecasting |
| `6hours` | 24 | **Default** - Balanced |
| `12hours` | 48 | Long-term planning |
| `24hours` | 96 | Day-ahead forecasting |

## 🎯 Target Events

| Event | Description |
|-------|-------------|
| `event_E3_LowTemp_lt0` | Low temperature (< 0°C) |
| `event_E4_HighWind_Peak_gt25` | High wind peaks (> 25 m/s) |
| `event_E5_LowWind_lt2` | Low wind conditions (< 2 m/s) |
| `event_E6_HighTemp_gt24` | High temperature (> 24°C) |

## 🔧 Performance Features

### Mixed Precision Training (FP16)
- **~2x speedup** on modern GPUs (Volta, Turing, Ampere)
- Automatic via `torch.cuda.amp`
- Uses `BCEWithLogitsLoss` for numerical stability

### Parallel Execution
- Multiple tower-event experiments run simultaneously
- Thread-based parallelism for GPU sharing
- Configurable with `--parallel N`

### Optimized Data Loading
- `num_workers=4` for parallel data loading
- `pin_memory=True` for faster GPU transfers
- `persistent_workers=True` to avoid worker respawn overhead

### Aggressive Pruning
- Median pruner kills bad trials early
- Warmup: 5 epochs before pruning
- Check interval: every 3 epochs

## 📁 Output Structure

```
optuna_results_gru_20251208_160000/
├── best_params.json           # Best hyperparameters per tower-event
├── config.json                # Experiment configuration
├── optimization_results.csv   # All results with metrics
└── trials_TOWER_EVENT_MODEL.csv  # Per-experiment trial history
```

## 📈 Example Usage

### Quick Test (3 trials, fast)
```bash
python JT_Extreme_event_labeler_DL_optuna.py --model gru --n_trials 3 --epochs 20
```

### Full Optimization (50 trials)
```bash
python JT_Extreme_event_labeler_DL_optuna.py \
    --model gru \
    --n_trials 50 \
    --epochs 100 \
    --patience 15 \
    --n_splits 5 \
    --gpu 0 \
    --parallel 4 \
    --output results_gru_full
```

### Multi-GPU Full Run
```bash
# GPU 0: GRU + CNN
python JT_Extreme_event_labeler_DL_optuna.py --gpu 0 --model gru --n_trials 50 &
python JT_Extreme_event_labeler_CNN_optuna.py --gpu 0 --model cnn --n_trials 50 &

# GPU 1: LSTM + TCN
python JT_Extreme_event_labeler_DL_optuna.py --gpu 1 --model lstm --n_trials 50 &
python JT_Extreme_event_labeler_CNN_optuna.py --gpu 1 --model tcn --n_trials 50 &

# Wait for all
wait
echo "All optimizations complete!"
```

## 🖥️ Monitor GPU Usage

```bash
# Watch GPU utilization
watch -n 2 nvidia-smi

# Check available GPUs
nvidia-smi --query-gpu=index,name,memory.free --format=csv
```

## 📊 Optimization Metric

All models are optimized for **MCC (Matthews Correlation Coefficient)**:
- Range: [-1, 1]
- Handles imbalanced classes well
- +1 = perfect prediction
- 0 = random prediction
- -1 = inverse prediction

## ⚠️ Notes

1. **Temporal Data**: `shuffle=False` in DataLoaders to preserve time-series order
2. **TimeSeriesSplit**: Train on past, validate on future (no data leakage)
3. **Memory**: Reduce `--parallel` if GPU OOM errors occur
4. **Typo Warning**: Use `--model lstm` (not `ltsm`)

## 🐛 Troubleshooting

### CUDA Out of Memory
```bash
# Reduce parallelism
python JT_Extreme_event_labeler_DL_optuna.py --parallel 1 --batch_size 32
```

### Slow Training
```bash
# Reduce sequence length and epochs
python JT_Extreme_event_labeler_DL_optuna.py --epochs 20
```

### Check GPU Assignment
```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"Current device: {torch.cuda.current_device()}")
print(f"Device name: {torch.cuda.get_device_name(0)}")
```


# Weather Event Forecasting Pipeline

## 🚀 Quick Start

```bash
conda activate ml_env2
```

---

## 📋 Project Overview

This project implements a **multi-event temporal forecasting pipeline** for predicting extreme weather events from meteorological tower data. The pipeline includes multiple model architectures with both default hyperparameters and Bayesian optimization (Optuna) variants.

---

## 🔄 Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         DATA PROCESSING PIPELINE                            │
├─────────────────────────────────────────────────────────────────────────────┤
│  1. Load CSV: fully_labeled_weather_data_with_events.csv                    │
│  2. Clean columns (remove _min, _meets_duration suffixes)                   │
│  3. Split data by tower (TOWA, TOWB, TOWD, TOWF, TOWS, TOWY)               │
│  4. Filter columns by null percentage threshold                             │
│  5. Prepare features based on model type (see below)                        │
│  6. Train with TimeSeriesSplit cross-validation                             │
│  7. Evaluate with comprehensive metrics (MCC, AUC-ROC, F1, etc.)            │
└─────────────────────────────────────────────────────────────────────────────┘
```

---

## 📁 Script Descriptions

| Script | Model Types | Data Processing | Hyperparameters |
|--------|-------------|-----------------|-----------------|
| `JT_Extreme_event_labeler_ML.py` | LightGBM, XGBoost | **Lag features + Rolling windows** | Fixed/Default |
| `JT_Extreme_event_labeler_ML_optuna.py` | LightGBM, XGBoost | **Lag features + Rolling windows** | Bayesian (Optuna) |
| `JT_Extreme_event_labeler_DL.py` | GRU, LSTM | **Raw sequences (no lags)** | Fixed/Default |
| `JT_Extreme_event_labeler_DL_optuna.py` | GRU, LSTM | **Raw sequences (no lags)** | Bayesian (Optuna) |
| `JT_Extreme_event_labeler_CNN.py` | CNN, TCN | **Raw sequences (no lags)** | Fixed/Default |
| `JT_Extreme_event_labeler_CNN_optuna.py` | CNN, TCN | **Raw sequences (no lags)** | Bayesian (Optuna) |

---

## 🧠 Model-Specific Data Processing

### **ML Models (LightGBM & XGBoost)** — `_ML.py` / `_ML_optuna.py`

- **Method**: Explicit **lag features** and **rolling window statistics**
- **Why?** Tree-based models process each sample independently — they cannot learn temporal patterns from sequences. Explicit lag and rolling features provide historical context.

```
📊 Feature Engineering:
├── Original numeric features
├── Lag features: [1, 2, 4, 8, 16, 24] timesteps back
├── Rolling mean: windows of [4, 12, 24, 96] timesteps
└── Rolling std: windows of [4, 12, 24, 96] timesteps
```

### **Deep Learning Models (GRU & LSTM)** — `_DL.py` / `_DL_optuna.py`

- **Method**: **Raw sequences** fed directly into recurrent networks
- **Why?** GRU/LSTM learn temporal dependencies through their hidden states. The sequence of raw features IS the temporal context — explicit lag features are redundant.

```
📊 Feature Engineering:
├── Original numeric features ONLY (no lag columns)
├── TimeSeriesDataset creates sequences of SEQUENCE_LENGTH
└── Model input shape: (batch, sequence_length, n_features)
```

### **Convolutional Models (CNN & TCN)** — `_CNN.py` / `_CNN_optuna.py`

- **Method**: **Raw sequences** processed via convolutions
- **Why?** CNNs/TCNs learn temporal patterns through convolution operations over the sequence dimension. The kernel slides over time, automatically extracting multi-scale features.

```
📊 Feature Engineering:
├── Original numeric features ONLY (no lag columns)
├── TimeSeriesDataset creates sequences of SEQUENCE_LENGTH
├── CNN: Standard convolutions slide over time
└── TCN: Dilated causal convolutions for long-range patterns
```

---

## ⚙️ Default vs Optuna Scripts

### **Default Scripts** (`_ML.py`, `_DL.py`, `_CNN.py`)
- Fixed, pre-defined hyperparameters
- Quick experiments & baseline results

### **Optuna Scripts** (`_ML_optuna.py`, `_DL_optuna.py`, `_CNN_optuna.py`)
- **Bayesian optimization** via Optuna TPE sampler
- **Optimization metric**: MCC (Matthews Correlation Coefficient)
- **Early pruning** of unpromising trials
- **Per tower-event optimization** — finds best params for each combination
- **ML Optuna also searches**: lag config (`short`/`medium`/`long`), rolling windows

---

## 🎯 Target Events

| Event | Description | Column Name |
|-------|-------------|-------------|
| E3 | Low Temperature < 0°C | `event_E3_LowTemp_lt0` |
| E4 | High Wind Peak > 25 m/s | `event_E4_HighWind_Peak_gt25` |
| E5 | Low Wind < 2 m/s | `event_E5_LowWind_lt2` |
| E6 | High Temperature > 24°C | `event_E6_HighTemp_gt24` |

---

## 📈 Forecast Horizons

| Horizon | Timesteps | Description |
|---------|-----------|-------------|
| 15min | 1 | Next timestep |
| 30min | 2 | 2 timesteps ahead |
| 1hour | 4 | 4 timesteps ahead |
| 3hours | 12 | 12 timesteps ahead |
| **6hours** | **24** | **Default** |
| 12hours | 48 | 48 timesteps ahead |
| 24hours | 96 | 96 timesteps ahead |

---

## 🏃 Running the Scripts for optuna opt

```bash

# ML models (Optuna optimization)
python JT_Extreme_event_labeler_ML_optuna.py --model lightgbm --gpu 0
python JT_Extreme_event_labeler_ML_optuna.py --model xgboost --gpu 1


# Deep Learning (Optuna)
python JT_Extreme_event_labeler_DL_optuna.py --model gru --gpu 0
python JT_Extreme_event_labeler_DL_optuna.py --model lstm --gpu 1


# CNN/TCN (Optuna)
python JT_Extreme_event_labeler_CNN_optuna.py --model tcn --gpu 0
python JT_Extreme_event_labeler_CNN_optuna.py --model cnn --gpu 1
```


## 🏃 Running the Scripts for final results

```bash

# ML models (default)
python JT_Extreme_event_labeler_ML.py

# Deep Learning (default)
python JT_Extreme_event_labeler_DL.py

# CNN/TCN (default)
python JT_Extreme_event_labeler_CNN.py


```

---

## 💡 Summary: Why Different Data Processing?

| Model Type | Learns Temporal Patterns Via | Data Processing |
|------------|------------------------------|-----------------|
| **LightGBM/XGBoost** | Explicit features only | ✅ Lag + Rolling features |
| **GRU/LSTM** | Hidden state memory | ✅ Raw sequences |
| **CNN/TCN** | Convolution kernels | ✅ Raw sequences |
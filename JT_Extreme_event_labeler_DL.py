import pandas as pd
from typing import Tuple, Optional


# Read the CSV file
df = pd.read_csv("fully_labeled_weather_data_with_events.csv")

# 2️⃣ Move 'tower' and 'timestamp' to the first two columns
cols = df.columns.tolist()
priority_cols = ['tower', 'timestamp']
remaining_cols = [c for c in cols if c not in priority_cols]
df = df[priority_cols + remaining_cols]

# Drop columns ending with '_min' or '_meets_duration'
df = df.loc[:, ~df.columns.str.endswith(('_min', '_meets_duration'))]

#remove these columns
df = df.drop(columns=['event_count','active_events','event_durations', 'has_any_event'])


pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)


# Check the first few rows
df.head(20)






import pandas as pd

# List all boolean event columns
event_cols = [
    'event_E3_LowTemp_lt0',
    'event_E4_HighWind_Peak_gt25',
    'event_E5_LowWind_lt2',
    
]

# Ensure the columns are boolean
df[event_cols] = df[event_cols].astype(bool)

# Count how many events are True per row
df["true_event_count"] = df[event_cols].sum(axis=1)

# Filter: rows where more than 2 events are True simultaneously (per tower)
multiple_events_df = df[df["true_event_count"] > 2]

# Optional: group by tower if you want to see which towers are affected
tower_event_summary = multiple_events_df.groupby("tower")["true_event_count"].count().reset_index()

print("✅ Rows with more than two active events:")
print(multiple_events_df[["tower", "timestamp", "true_event_count"]])

print("\n📊 Summary of towers with >2 events:")
print(tower_event_summary)


# Check all unique tower types
print("Unique tower types:", df['tower'].unique())

# Split into separate DataFrames per tower
tower_dfs = {tower: df[df['tower'] == tower].copy() for tower in df['tower'].unique()}

# Example: Access one tower's dataset
print("\nExample – TOWA DataFrame:")
print(tower_dfs['TOWA'].head())



import pandas as pd
import numpy as np

# Ensure timestamp is datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Compute % of nulls per column and tower
null_percentage = df.groupby('tower').apply(lambda x: x.isnull().mean() * 100).round(2)

# Define non-overlapping ranges: 0–10%, 10–20%, ..., 90–100%
bins = np.arange(0, 110, 10)  # 0,10,20,...,100
range_labels = [f"{bins[i]}–{bins[i+1]}% nulls" for i in range(len(bins)-1)]

# Store results in dictionary
ranges = {label: [] for label in range_labels}

for tower in null_percentage.index:
    tower_data = null_percentage.loc[tower]
    for i in range(len(bins)-1):
        low, high = bins[i], bins[i+1]
        cols = list(tower_data[(tower_data >= low) & (tower_data < high)].index)
        ranges[range_labels[i]].append((tower, cols))

# Print summary
for category, values in ranges.items():
    print(f"\n===== {category} =====")
    for tower, cols in values:
        if len(cols) > 0:
            print(f"Tower {tower}: {len(cols)} columns → {cols}")


def filter_dfs_below_threshold(df, ranges, threshold=100):
    """
    Filter all towers, keeping columns whose null % is below a threshold.
    Always keeps 'tower', 'timestamp' and 'true_event_count'.
    """
    filtered_dfs = {}
    
    # Find all ranges below threshold
    selected_ranges = [r for r in ranges.keys() if int(r.split("–")[1].replace("% nulls","")) <= threshold]
    
    for tower in df['tower'].unique():
        cols_to_keep = ['tower', 'timestamp']  # <-- keep true_event_count
        
        # Collect columns from all selected ranges
        for r in selected_ranges:
            for t, cols in ranges[r]:
                if t == tower:
                    cols_to_keep.extend(cols)
        
        # Remove duplicates (some columns may appear in multiple ranges)
        cols_to_keep = list(dict.fromkeys(cols_to_keep))
        
        # Guard: keep only columns that actually exist in the dataframe
        cols_to_keep = [c for c in cols_to_keep if c in df.columns]
        
        filtered_dfs[tower] = df[df['tower'] == tower][cols_to_keep].copy()
        filtered_dfs[tower] = filtered_dfs[tower].drop("true_event_count", axis=1)
    
    return filtered_dfs

filtered_dfs = filter_dfs_below_threshold(df, ranges, threshold=100)


# ==================== MULTI-EVENT TEMPORAL FORECASTING - DEEP LEARNING MODELS ====================
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (classification_report, roc_auc_score, confusion_matrix,
                             precision_recall_curve, f1_score, precision_score, recall_score,
                             accuracy_score, balanced_accuracy_score, matthews_corrcoef, 
                             cohen_kappa_score, average_precision_score)
import torch
import torch.nn as nn
import os
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import warnings
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pickle
import json
from pathlib import Path
import gc

warnings.filterwarnings('ignore')



def clear_gpu_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

def get_gpu_memory_info():
    """Print current GPU memory usage"""
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1024**3
            reserved = torch.cuda.memory_reserved(i) / 1024**3
            print(f"GPU {i}: Allocated: {allocated:.2f}GB, Reserved: {reserved:.2f}GB")


# ==================== PYTORCH DATASET ====================
class TimeSeriesDataset(Dataset):
    """Custom Dataset for time series data"""
    def __init__(self, X: np.ndarray, y: np.ndarray, sequence_length: int = 24):
        """
        Args:
            X: Input features (n_samples, n_features)
            y: Target labels (n_samples,)
            sequence_length: Length of sequence for each sample
        """
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.X) - self.sequence_length + 1
    
    def __getitem__(self, idx):
        # Return sequence of length sequence_length
        X_seq = self.X[idx:idx + self.sequence_length]
        y_target = self.y[idx + self.sequence_length - 1]
        return X_seq, y_target


# ==================== DEEP LEARNING MODELS ====================

class GRUModel(nn.Module):
    """Gated Recurrent Unit model"""
    def __init__(self, input_dim: int, hidden_dim: int = 128, 
                 num_layers: int = 2, dropout: float = 0.3,
                 bidirectional: bool = True):
        super(GRUModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.gru = nn.GRU(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        self.fc = nn.Sequential(
            nn.Linear(fc_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x shape: (batch, sequence_length, features)
        gru_out, _ = self.gru(x)
        # Take last output
        last_output = gru_out[:, -1, :]
        output = self.fc(last_output)
        return output.squeeze(-1)


class LSTMModel(nn.Module):
    """Long Short-Term Memory model"""
    def __init__(self, input_dim: int, hidden_dim: int = 128, 
                 num_layers: int = 2, dropout: float = 0.3,
                 bidirectional: bool = True):
        super(LSTMModel, self).__init__()
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        
        self.lstm = nn.LSTM(
            input_dim, 
            hidden_dim, 
            num_layers, 
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional
        )
        
        fc_input_dim = hidden_dim * 2 if bidirectional else hidden_dim
        
        self.fc = nn.Sequential(
            nn.Linear(fc_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        # x shape: (batch, sequence_length, features)
        lstm_out, _ = self.lstm(x)
        # Take last output
        last_output = lstm_out[:, -1, :]
        output = self.fc(last_output)
        return output.squeeze(-1)


# # ==================== CONFIGURATION ====================
# class DeepLearningConfig:
#     """Configuration for deep learning models"""


#     os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    
#     # Force use of GPU 1
#     DEVICE = 'cuda:1' if torch.cuda.is_available() else 'cpu'  # Changed from 'cuda'
    
    
    
#     # ==================== FORECASTING HORIZON ====================
#     FORECAST_HORIZONS = {
#         '15min': 1, '30min': 2, '1hour': 4, '3hours': 12,
#         '6hours': 24, '12hours': 48, '24hours': 96,
#     }
#     SELECTED_HORIZON = '6hours'
    
#     # ==================== TARGET EVENTS ====================
#     TARGET_EVENTS = [
#         'event_E3_LowTemp_lt0',
#         'event_E4_HighWind_Peak_gt25',
#         'event_E5_LowWind_lt2',
#         'event_E6_HighTemp_gt24'
#     ]
    
#     # ==================== TEMPORAL FEATURES ====================
#     LAG_CONFIGS = {
#         'short': [1, 2, 4, 8],
#         'medium': [1, 2, 4, 8, 16, 24],
#         'long': [1, 4, 8, 16, 24, 48, 96],
#     }
#     SELECTED_LAG_CONFIG = 'medium'
#     ROLLING_WINDOWS = [4, 12, 24, 96]
    
#     # ==================== MODEL PARAMETERS ====================
#     N_SPLITS = 5
#     DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
#     USE_CLASS_WEIGHTS = True
#     MODELS_TO_TRAIN = ['gru', 'lstm']  # Only GRU and LSTM
    
#     # ==================== DEEP LEARNING HYPERPARAMETERS ====================
#     SEQUENCE_LENGTH = 24
#     BATCH_SIZE = 16
#     LEARNING_RATE = 0.001
#     NUM_EPOCHS = 100
#     EARLY_STOPPING_PATIENCE = 15
    
#     # Model-specific parameters
#     GRU_PARAMS = {
#         'hidden_dim': 128,
#         'num_layers': 2,
#         'dropout': 0.3,
#         'bidirectional': True
#     }
    
#     LSTM_PARAMS = {
#         'hidden_dim': 128,
#         'num_layers': 2,
#         'dropout': 0.3,
#         'bidirectional': True
#     }
    
#     # ==================== EVALUATION METRICS ====================
#     CLASSIFICATION_METRICS = [
#         'accuracy',
#         'balanced_accuracy',
#         'precision',
#         'recall',
#         'f1',
#         'auc_roc',
#         'auc_pr',
#         'specificity',
#         'mcc',
#         'cohen_kappa'
#     ]


# ==================== CONFIGURATION ====================
class DeepLearningConfig:
    """Configuration for deep learning models"""

    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # ==================== FORECASTING HORIZON ====================
    FORECAST_HORIZONS = {
        '15min': 1, '30min': 2, '1hour': 4, '3hours': 12,
        '6hours': 24, '12hours': 48, '24hours': 96,
    }
    SELECTED_HORIZON = '6hours'
    
    # ==================== TARGET EVENTS ====================
    TARGET_EVENTS = [
        'event_E3_LowTemp_lt0',
        'event_E4_HighWind_Peak_gt25',
        'event_E5_LowWind_lt2',
        # 'event_E6_HighTemp_gt24'  # Removed - column doesn't exist in dataset
    ]
    
    # ==================== MODEL PARAMETERS ====================
    N_SPLITS = 5
    USE_CLASS_WEIGHTS = True
    MODELS_TO_TRAIN = ['gru', 'lstm']
    
    # ==================== DEEP LEARNING HYPERPARAMETERS ====================
    # ✅ SEQUENCE_LENGTH controls temporal context (replaces manual lag features)
    # 24 steps = 6 hours of history (at 15-min intervals)
    # 48 steps = 12 hours of history
    # 96 steps = 24 hours of history
    SEQUENCE_LENGTH = 48  # Model sees 12 hours of past data as a sequence
    
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    EARLY_STOPPING_PATIENCE = 15
    
    # Model-specific parameters
    GRU_PARAMS = {
        'hidden_dim': 128,
        'num_layers': 2,
        'dropout': 0.3,
        'bidirectional': True
    }
    
    LSTM_PARAMS = {
        'hidden_dim': 128,
        'num_layers': 2,
        'dropout': 0.3,
        'bidirectional': True
    }
    
    # ==================== EVALUATION METRICS ====================
    CLASSIFICATION_METRICS = [
        'accuracy', 'balanced_accuracy', 'precision', 'recall',
        'f1', 'auc_roc', 'auc_pr', 'specificity', 'mcc', 'cohen_kappa'
    ]

# ==================== METRICS CALCULATION ====================
def calculate_all_metrics(y_true, y_pred, y_pred_proba):
    """Calculate comprehensive classification metrics"""
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'auc_roc': roc_auc_score(y_true, y_pred_proba),
        'auc_pr': average_precision_score(y_true, y_pred_proba),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'mcc': matthews_corrcoef(y_true, y_pred),
        'cohen_kappa': cohen_kappa_score(y_true, y_pred),
        'true_positives': int(tp),
        'true_negatives': int(tn),
        'false_positives': int(fp),
        'false_negatives': int(fn)
    }
    
    return metrics


# ==================== MODEL CREATION ====================
def create_model(model_type: str, input_dim: int, config: DeepLearningConfig):
    """Create model based on type"""
    if model_type == 'gru':
        return GRUModel(input_dim, **config.GRU_PARAMS)
    elif model_type == 'lstm':
        return LSTMModel(input_dim, **config.LSTM_PARAMS)
    else:
        raise ValueError(f"Unknown model type: {model_type}")




def train_deep_learning_model(X_train: np.ndarray, y_train: np.ndarray,
                               X_val: np.ndarray, y_val: np.ndarray,
                               model_type: str, config: DeepLearningConfig,
                               class_weights: Optional[torch.Tensor] = None):
    """Train a deep learning model"""
    
    # Create datasets
    train_dataset = TimeSeriesDataset(X_train, y_train, config.SEQUENCE_LENGTH)
    val_dataset = TimeSeriesDataset(X_val, y_val, config.SEQUENCE_LENGTH)
    
    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    # Create model
    input_dim = X_train.shape[1]
    model = create_model(model_type, input_dim, config)
    model = model.to(config.DEVICE)
    
    # Loss function - use reduction='none' for per-sample weighting
    criterion = nn.BCELoss(reduction='none')
    
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LEARNING_RATE)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                            factor=0.5, patience=5)
    
    # Training loop
    best_val_loss = float('inf')
    patience_counter = 0
    train_losses = []
    val_losses = []
    
    for epoch in range(config.NUM_EPOCHS):
        # Training
        model.train()
        train_loss = 0.0
        train_sample_count = 0
        
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch = X_batch.to(config.DEVICE)
            y_batch = y_batch.to(config.DEVICE)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            
            # Compute loss per sample
            loss_per_sample = criterion(outputs, y_batch)
            
            # Apply class weights if provided
            if class_weights is not None:
                # Get the weights for this batch
                start_idx = batch_idx * config.BATCH_SIZE
                end_idx = min(start_idx + len(X_batch), len(train_dataset))
                batch_weights = class_weights[start_idx:end_idx].to(config.DEVICE)
                
                # Weight the losses
                weighted_loss = (loss_per_sample * batch_weights).mean()
            else:
                weighted_loss = loss_per_sample.mean()
            
            weighted_loss.backward()
            optimizer.step()
            
            train_loss += weighted_loss.item() * len(X_batch)
            train_sample_count += len(X_batch)
        
        train_loss /= train_sample_count
        train_losses.append(train_loss)
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_sample_count = 0
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(config.DEVICE)
                y_batch = y_batch.to(config.DEVICE)
                
                outputs = model(X_batch)
                loss_per_sample = criterion(outputs, y_batch)
                loss = loss_per_sample.mean()
                
                val_loss += loss.item() * len(X_batch)
                val_sample_count += len(X_batch)
        
        val_loss /= val_sample_count
        val_losses.append(val_loss)
        
        scheduler.step(val_loss)
        
        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
            
        if patience_counter >= config.EARLY_STOPPING_PATIENCE:
            break
    
    # Load best model
    model.load_state_dict(best_model_state)
    
    return model, {'train_losses': train_losses, 'val_losses': val_losses}



def predict_with_model(model, X: np.ndarray, config: DeepLearningConfig) -> np.ndarray:
    """Make predictions with trained model"""
    dataset = TimeSeriesDataset(X, np.zeros(len(X)), config.SEQUENCE_LENGTH)
    loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=False)
    
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for X_batch, _ in loader:
            X_batch = X_batch.to(config.DEVICE)
            outputs = model(X_batch)
            predictions.extend(outputs.cpu().numpy())
    
    return np.array(predictions)




def train_single_tower_event_models(X: pd.DataFrame, y: pd.Series, 
                                    config: DeepLearningConfig,
                                    tower_name: str,
                                    event_name: str,
                                    metadata: pd.DataFrame = None) -> Dict:
    """Train deep learning models for ONE tower and ONE event"""
    
    tscv = TimeSeriesSplit(n_splits=config.N_SPLITS)
    
    # Convert to numpy
    X_np = X.values
    y_np = y.values
    
    results = {
        'tower': tower_name,
        'event': event_name,
        'models': {}
    }
    
    # Initialize results for each model type
    for model_type in config.MODELS_TO_TRAIN:
        results['models'][model_type] = {
            'trained_models': [],
            'fold_metadata': [],
            'training_history': []
        }
        # Initialize all metrics
        for metric in config.CLASSIFICATION_METRICS:
            results['models'][model_type][metric] = []
        # Initialize confusion matrix components
        for cm_comp in ['true_positives', 'true_negatives', 'false_positives', 'false_negatives']:
            results['models'][model_type][cm_comp] = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_np), 1):
        print(f"         Fold {fold}/{config.N_SPLITS}", end=' ')
        
        # ✅ Clear GPU memory before each fold
        clear_gpu_memory()
        
        X_train, X_val = X_np[train_idx], X_np[val_idx]
        y_train, y_val = y_np[train_idx], y_np[val_idx]
        
        # Calculate class weights
        class_weights = None
        if config.USE_CLASS_WEIGHTS:
            n_samples = len(y_train)
            n_pos = y_train.sum()
            n_neg = n_samples - n_pos
            
            if n_pos > 0 and n_neg > 0:
                weight_pos = n_samples / (2 * n_pos)
                weight_neg = n_samples / (2 * n_neg)
                
                # Create weight tensor for each sample
                weights = np.where(y_train == 1, weight_pos, weight_neg)
                class_weights = torch.FloatTensor(weights)
        
        # Train each model type
        for model_type in config.MODELS_TO_TRAIN:
            print(f"[{model_type}]", end=' ')
            
            model, history = train_deep_learning_model(
                X_train, y_train, X_val, y_val,
                model_type, config, class_weights
            )
            
            # Predict on validation set
            y_pred_proba = predict_with_model(model, X_val, config)
            
            # Need to adjust indices for sequence length
            valid_indices = slice(config.SEQUENCE_LENGTH - 1, len(y_val))
            y_val_adjusted = y_val[valid_indices]
            
            # Optimal threshold
            precisions, recalls, thresholds = precision_recall_curve(y_val_adjusted, y_pred_proba)
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
            optimal_idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
            
            y_pred = (y_pred_proba >= optimal_threshold).astype(int)
            
            # Calculate all metrics
            metrics = calculate_all_metrics(y_val_adjusted, y_pred, y_pred_proba)
            
            # Store metrics
            for metric_name, metric_value in metrics.items():
                if metric_name in results['models'][model_type]:
                    results['models'][model_type][metric_name].append(metric_value)
            
            # ✅ Move model to CPU before storing to save GPU memory
            model = model.cpu()
            results['models'][model_type]['trained_models'].append(model)
            results['models'][model_type]['training_history'].append(history)
            
            # Store fold metadata
            if metadata is not None:
                fold_meta = metadata.iloc[val_idx[valid_indices]].copy()
                fold_meta['y_true'] = y_val_adjusted
                fold_meta['y_pred_proba'] = y_pred_proba
                fold_meta['y_pred'] = y_pred
                fold_meta['fold'] = fold
                fold_meta['model_type'] = model_type
                results['models'][model_type]['fold_metadata'].append(fold_meta)
            
            # ✅ Clear GPU memory after each model
            del model
            clear_gpu_memory()
        
        print()  # New line after fold
    
    return results


# # ==================== FEATURE PREPARATION ====================
# def prepare_temporal_features(df: pd.DataFrame, config: DeepLearningConfig, 
#                               target_col: str) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
#     """Create temporal features (lags, rolling stats) for forecasting"""
    
#     df = df.copy()
#     df = df.sort_index()
    
#     # Get numeric columns (exclude event columns, metadata, and non-numeric)
#     numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
#     # Exclude patterns
#     exclude_patterns = ['event_', 'tower', 'timestamp', 'date', 'year_month', 
#                        'week_of_month', 'hour_of_day', 'day_name', 'day_numeric',
#                        'date_numeric']
    
#     numeric_cols = [c for c in numeric_cols 
#                    if not any(pattern in str(c) for pattern in exclude_patterns)]
    
#     if target_col in numeric_cols:
#         numeric_cols.remove(target_col)
    
#     for col in numeric_cols:
#         if df[col].dtype == 'bool':
#             df[col] = df[col].astype(int)
    
#     feature_dfs = []
#     feature_names = []
    
#     # Original features
#     feature_dfs.append(df[numeric_cols])
#     feature_names.extend(numeric_cols)
    
#     # Lag features
#     lags = config.LAG_CONFIGS[config.SELECTED_LAG_CONFIG]
#     for col in numeric_cols:
#         for lag in lags:
#             lag_col = f'{col}_lag{lag}'
#             feature_dfs.append(df[col].shift(lag).to_frame(lag_col))
#             feature_names.append(lag_col)
    
#     # Rolling window features
#     for col in numeric_cols:
#         for window in config.ROLLING_WINDOWS:
#             roll_mean_col = f'{col}_roll{window}_mean'
#             feature_dfs.append(df[col].rolling(window=window).mean().to_frame(roll_mean_col))
#             feature_names.append(roll_mean_col)
            
#             roll_std_col = f'{col}_roll{window}_std'
#             feature_dfs.append(df[col].rolling(window=window).std().to_frame(roll_std_col))
#             feature_names.append(roll_std_col)
    
#     X = pd.concat(feature_dfs, axis=1)
    
#     horizon_steps = config.FORECAST_HORIZONS[config.SELECTED_HORIZON]
    
#     if df[target_col].dtype == 'object':
#         df[target_col] = df[target_col].astype(bool).astype(int)
#     elif df[target_col].dtype == 'bool':
#         df[target_col] = df[target_col].astype(int)
    
#     y = df[target_col].shift(-horizon_steps)
    
#     valid_idx = X.notna().all(axis=1) & y.notna()
#     X = X[valid_idx]
#     y = y[valid_idx]
    
#     for col in X.columns:
#         if X[col].dtype == 'object':
#             X[col] = pd.to_numeric(X[col], errors='coerce')
    
#     X = X.select_dtypes(include=[np.number])
#     feature_names = list(X.columns)
    
#     return X, y, feature_names


# ==================== FEATURE PREPARATION (SEQUENCES HANDLE TEMPORAL PATTERNS) ====================
def prepare_temporal_features(df: pd.DataFrame, config: DeepLearningConfig, 
                              target_col: str) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """
    Prepare features for LSTM/GRU models.
    
    NO explicit lag columns or rolling window columns needed!
    The TimeSeriesDataset creates sequences of length SEQUENCE_LENGTH,
    so the model sees [t-23, t-22, ..., t-1, t] for each prediction.
    This IS the lag information - learned natively by the recurrent layers.
    """
    
    df = df.copy()
    df = df.sort_index()
    
    # Get numeric columns (exclude event columns, metadata, and non-numeric)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # Exclude patterns
    exclude_patterns = ['event_', 'tower', 'timestamp', 'date', 'year_month', 
                       'week_of_month', 'hour_of_day', 'day_name', 'day_numeric',
                       'date_numeric']
    
    numeric_cols = [c for c in numeric_cols 
                   if not any(pattern in str(c) for pattern in exclude_patterns)]
    
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    # Convert boolean columns to int
    for col in numeric_cols:
        if df[col].dtype == 'bool':
            df[col] = df[col].astype(int)
    
    # ✅ Use ONLY original features - NO lag columns, NO rolling columns
    # The SEQUENCE_LENGTH in TimeSeriesDataset handles temporal context:
    # - SEQUENCE_LENGTH=24 means model sees 24 timesteps (6 hours of 15-min data)
    # - SEQUENCE_LENGTH=96 means model sees 96 timesteps (24 hours of 15-min data)
    X = df[numeric_cols].copy()
    feature_names = list(numeric_cols)
    
    # Create target with forecast horizon shift
    horizon_steps = config.FORECAST_HORIZONS[config.SELECTED_HORIZON]
    
    if df[target_col].dtype == 'object':
        df[target_col] = df[target_col].astype(bool).astype(int)
    elif df[target_col].dtype == 'bool':
        df[target_col] = df[target_col].astype(int)
    
    y = df[target_col].shift(-horizon_steps)
    
    # Remove rows with NaN values
    valid_idx = X.notna().all(axis=1) & y.notna()
    X = X[valid_idx]
    y = y[valid_idx]
    
    # Ensure all columns are numeric
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = pd.to_numeric(X[col], errors='coerce')
    
    X = X.select_dtypes(include=[np.number])
    feature_names = list(X.columns)
    
    return X, y, feature_names

# ==================== RUN EXPERIMENTS ====================
def run_multi_event_experiments(filtered_dfs: Dict[str, pd.DataFrame], 
                                config: DeepLearningConfig) -> Dict:
    """Run experiments for all towers and all events"""
    
    all_results = {}
    
    print("\n" + "="*80)
    print("RUNNING MULTI-EVENT DEEP LEARNING EXPERIMENTS")
    print("="*80)
    print(f"Device: {config.DEVICE}")
    print(f"Models: {', '.join(config.MODELS_TO_TRAIN)}")
    print(f"Sequence Length: {config.SEQUENCE_LENGTH}")
    print(f"Batch Size: {config.BATCH_SIZE}")
    print("="*80)
    
    for tower_name, tower_df in filtered_dfs.items():
        print(f"\n{'='*70}")
        print(f"🏗️  TOWER: {tower_name}")
        print(f"{'='*70}")
        
        all_results[tower_name] = {}
        
        for event_col in config.TARGET_EVENTS:
            
            if event_col not in tower_df.columns:
                print(f"   ⚠️  {event_col} not found in {tower_name}, skipping...")
                all_results[tower_name][event_col] = {'error': 'Event column not found'}
                continue
            
            print(f"\n   📍 Event: {event_col}")
            
            try:
                X, y, feature_cols = prepare_temporal_features(tower_df, config, event_col)
                
                if len(y) < 100:
                    print(f"      ⚠️  Insufficient data ({len(y)} samples), skipping...")
                    all_results[tower_name][event_col] = {'error': 'Insufficient data'}
                    continue
                
                if y.sum() < 10:
                    print(f"      ⚠️  Too few positive events ({y.sum()}), skipping...")
                    all_results[tower_name][event_col] = {'error': 'Too few positive events'}
                    continue
                
                print(f"      ✓ Samples: {len(y):,} | Events: {y.sum():,} ({y.mean()*100:.2f}%)")
                print(f"      ✓ Features: {len(feature_cols)}")
                
                metadata = tower_df.loc[X.index, ['timestamp']].copy() if 'timestamp' in tower_df.columns else None
                
                results = train_single_tower_event_models(
                    X, y, config, tower_name, event_col, metadata
                )
                
                all_results[tower_name][event_col] = {
                    'results': results,
                    'importance': None,  # Deep learning models don't have feature importance
                    'n_samples': len(y),
                    'event_rate': float(y.mean())
                }
                
                # Print summary
                print(f"\n      📊 Results Summary:")
                for model_type in config.MODELS_TO_TRAIN:
                    model_results = results['models'][model_type]
                    mean_auc = np.mean(model_results['auc_roc'])
                    mean_f1 = np.mean(model_results['f1'])
                    print(f"         {model_type:10s}: AUC={mean_auc:.4f}, F1={mean_f1:.4f}")
                
            except Exception as e:
                print(f"      ❌ Error: {str(e)}")
                import traceback
                traceback.print_exc()
                all_results[tower_name][event_col] = {'error': str(e)}
    
    print("\n" + "="*80)
    print("✅ ALL TOWER-EVENT COMBINATIONS PROCESSED")
    print("="*80)
    
    return all_results


# ==================== SAVE RESULTS ====================
def save_all_results(all_results: Dict, config: DeepLearningConfig):
    """Save all experiment results with comprehensive summaries"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    output_dir = Path(f"deep_learning_results_{timestamp}")
    output_dir.mkdir(exist_ok=True)
    
    models_dir = output_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    # Save configuration
    config_dict = {
        'experiment': {
            'timestamp': timestamp,
            'description': 'Multi-event temporal forecasting with deep learning (GRU & LSTM)'
        },
        'events': {
            'target_events': config.TARGET_EVENTS,
            'horizon': config.SELECTED_HORIZON,
            'horizon_steps': config.FORECAST_HORIZONS[config.SELECTED_HORIZON]
        },
        'features': {
            'lag_config': config.SELECTED_LAG_CONFIG,
            'lags': config.LAG_CONFIGS[config.SELECTED_LAG_CONFIG],
            'rolling_windows': config.ROLLING_WINDOWS
        },
        'models': {
            'models_trained': config.MODELS_TO_TRAIN,
            'n_splits': config.N_SPLITS,
            'use_class_weights': config.USE_CLASS_WEIGHTS,
            'device': config.DEVICE,
            'sequence_length': config.SEQUENCE_LENGTH,
            'batch_size': config.BATCH_SIZE,
            'learning_rate': config.LEARNING_RATE,
            'num_epochs': config.NUM_EPOCHS
        },
        'gru_params': config.GRU_PARAMS,
        'lstm_params': config.LSTM_PARAMS,
        'metrics': config.CLASSIFICATION_METRICS
    }
    
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"\n✅ Configuration saved to: {output_dir / 'config.json'}")
    
    # Create summary
    summary_data = []
    for tower, tower_results in all_results.items():
        for event, event_results in tower_results.items():
            if 'error' not in event_results:
                row = {
                    'tower': tower,
                    'event': event,
                    'n_samples': event_results['n_samples'],
                    'event_rate': event_results['event_rate']
                }
                
                for model_type in config.MODELS_TO_TRAIN:
                    model_results = event_results['results']['models'][model_type]
                    
                    for metric in config.CLASSIFICATION_METRICS:
                        if metric in model_results:
                            values = model_results[metric]
                            row[f'{model_type}_{metric}_mean'] = np.mean(values)
                            row[f'{model_type}_{metric}_std'] = np.std(values)
                    
                    for cm_comp in ['true_positives', 'true_negatives', 'false_positives', 'false_negatives']:
                        if cm_comp in model_results:
                            row[f'{model_type}_{cm_comp}_mean'] = np.mean(model_results[cm_comp])
                
                summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_dir / 'summary_results.csv', index=False)
    print(f"✅ Summary results saved to: {output_dir / 'summary_results.csv'}")
    
    # ==================== PER-EVENT SUMMARY ====================
    print("\n📊 Creating per-event best models summary...")
    per_event_summary = []
    
    for event in config.TARGET_EVENTS:
        event_subset = summary_df[summary_df['event'] == event]
        
        if len(event_subset) == 0:
            continue
        
        event_summary = {
            'event': event,
            'n_towers': event_subset['tower'].nunique(),
            'total_samples': event_subset['n_samples'].sum(),
            'avg_event_rate': event_subset['event_rate'].mean()
        }
        
        for model_type in config.MODELS_TO_TRAIN:
            for metric in config.CLASSIFICATION_METRICS:
                metric_col = f'{model_type}_{metric}_mean'
                
                if metric_col in event_subset.columns:
                    best_idx = event_subset[metric_col].idxmax()
                    best_row = event_subset.loc[best_idx]
                    
                    event_summary[f'{model_type}_best_{metric}'] = best_row[metric_col]
                    event_summary[f'{model_type}_best_{metric}_tower'] = best_row['tower']
            
            for metric in config.CLASSIFICATION_METRICS:
                metric_col = f'{model_type}_{metric}_mean'
                if metric_col in event_subset.columns:
                    event_summary[f'{model_type}_avg_{metric}'] = event_subset[metric_col].mean()
                    event_summary[f'{model_type}_std_{metric}'] = event_subset[metric_col].std()
        
        per_event_summary.append(event_summary)
    
    per_event_df = pd.DataFrame(per_event_summary)
    per_event_df.to_csv(output_dir / 'best_models_per_event.csv', index=False)
    print(f"✅ Per-event best models saved to: {output_dir / 'best_models_per_event.csv'}")
    
    # ==================== PER-TOWER SUMMARY ====================
    print("\n📊 Creating per-tower best models summary...")
    per_tower_summary = []
    
    for tower in summary_df['tower'].unique():
        tower_subset = summary_df[summary_df['tower'] == tower]
        
        tower_summary = {
            'tower': tower,
            'n_events': tower_subset['event'].nunique(),
            'total_samples': tower_subset['n_samples'].sum(),
            'avg_event_rate': tower_subset['event_rate'].mean()
        }
        
        for model_type in config.MODELS_TO_TRAIN:
            for metric in config.CLASSIFICATION_METRICS:
                metric_col = f'{model_type}_{metric}_mean'
                
                if metric_col in tower_subset.columns:
                    best_idx = tower_subset[metric_col].idxmax()
                    best_row = tower_subset.loc[best_idx]
                    
                    tower_summary[f'{model_type}_best_{metric}'] = best_row[metric_col]
                    tower_summary[f'{model_type}_best_{metric}_event'] = best_row['event']
            
            for metric in config.CLASSIFICATION_METRICS:
                metric_col = f'{model_type}_{metric}_mean'
                if metric_col in tower_subset.columns:
                    tower_summary[f'{model_type}_avg_{metric}'] = tower_subset[metric_col].mean()
                    tower_summary[f'{model_type}_std_{metric}'] = tower_subset[metric_col].std()
        
        per_tower_summary.append(tower_summary)
    
    per_tower_df = pd.DataFrame(per_tower_summary)
    per_tower_df.to_csv(output_dir / 'best_models_per_tower.csv', index=False)
    print(f"✅ Per-tower best models saved to: {output_dir / 'best_models_per_tower.csv'}")
    
    # ==================== PER-TOWER-EVENT SUMMARY ====================
    print("\n📊 Creating per-tower-event best models summary...")
    per_tower_event_summary = []
    
    for tower in summary_df['tower'].unique():
        for event in config.TARGET_EVENTS:
            subset = summary_df[(summary_df['tower'] == tower) & (summary_df['event'] == event)]
            
            if len(subset) == 0:
                continue
            
            row = subset.iloc[0]
            
            tower_event_summary = {
                'tower': tower,
                'event': event,
                'n_samples': row['n_samples'],
                'event_rate': row['event_rate']
            }
            
            for model_type in config.MODELS_TO_TRAIN:
                for metric in config.CLASSIFICATION_METRICS:
                    metric_col = f'{model_type}_{metric}_mean'
                    if metric_col in row.index:
                        tower_event_summary[f'{model_type}_{metric}'] = row[metric_col]
            
            # Determine which model is better
            gru_auc = row.get('gru_auc_roc_mean', 0)
            lstm_auc = row.get('lstm_auc_roc_mean', 0)
            
            tower_event_summary['best_model'] = 'gru' if gru_auc >= lstm_auc else 'lstm'
            tower_event_summary['best_auc'] = max(gru_auc, lstm_auc)
            
            per_tower_event_summary.append(tower_event_summary)
    
    per_tower_event_df = pd.DataFrame(per_tower_event_summary)
    per_tower_event_df.to_csv(output_dir / 'best_models_per_tower_event.csv', index=False)
    print(f"✅ Per-tower-event best models saved to: {output_dir / 'best_models_per_tower_event.csv'}")
    
    # ==================== DETAILED TEXT REPORT ====================
    report_file = output_dir / 'experiment_report.txt'
    
    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MULTI-EVENT TEMPORAL FORECASTING - DEEP LEARNING REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Models: {', '.join(config.MODELS_TO_TRAIN)}\n")
        f.write(f"Forecast Horizon: {config.SELECTED_HORIZON} ({config.FORECAST_HORIZONS[config.SELECTED_HORIZON]} steps)\n")
        f.write(f"Sequence Length: {config.SEQUENCE_LENGTH}\n")
        f.write(f"Cross-validation folds: {config.N_SPLITS}\n")
        f.write(f"Total experiments: {len(summary_df)}\n\n")
        
        f.write("="*80 + "\n")
        f.write("OVERALL PERFORMANCE SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        for model_type in config.MODELS_TO_TRAIN:
            f.write(f"\n{model_type.upper()} MODEL\n")
            f.write("-"*80 + "\n")
            
            for metric in config.CLASSIFICATION_METRICS:
                mean_col = f'{model_type}_{metric}_mean'
                
                if mean_col in summary_df.columns:
                    mean_val = summary_df[mean_col].mean()
                    std_val = summary_df[mean_col].std()
                    max_val = summary_df[mean_col].max()
                    min_val = summary_df[mean_col].min()
                    
                    f.write(f"  {metric:20s}: {mean_val:.4f} ± {std_val:.4f}  "
                           f"[{min_val:.4f}, {max_val:.4f}]\n")
            
            f.write("\n")
        
        # Rankings
        f.write("\n" + "="*80 + "\n")
        f.write("ALL CONFIGURATIONS RANKED BY AUC-ROC\n")
        f.write("="*80 + "\n\n")
        
        for model_type in config.MODELS_TO_TRAIN:
            f.write(f"\n{model_type.upper()}\n")
            f.write("-"*80 + "\n")
            
            metric_col = f'{model_type}_auc_roc_mean'
            if metric_col in summary_df.columns:
                all_configs = summary_df.sort_values(metric_col, ascending=False)
                
                f.write(f"{'Rank':<6} {'Tower':<8} {'Event':<35} {'AUC':<8} {'F1':<8} {'Event%':<8}\n")
                f.write("-"*80 + "\n")
                
                for rank, (idx, row) in enumerate(all_configs.iterrows(), 1):
                    event_display = row['event'].replace('event_', '')
                    f.write(f"{rank:<6} {row['tower']:<8} {event_display:<35} "
                           f"{row[metric_col]:<8.4f} "
                           f"{row[f'{model_type}_f1_mean']:<8.4f} "
                           f"{row['event_rate']*100:<8.1f}\n")
                
                f.write("\n")
    
    print(f"✅ Experiment report saved to: {output_dir / 'experiment_report.txt'}")
    
    # Save models
    for tower, tower_results in all_results.items():
        for event, event_results in tower_results.items():
            if 'error' in event_results:
                continue
            
            for model_type in config.MODELS_TO_TRAIN:
                model_data = event_results['results']['models'][model_type]
                
                for fold_idx, model in enumerate(model_data['trained_models'], 1):
                    model_filename = f"{tower}_{event}_{model_type}_fold{fold_idx}.pt"
                    model_path = models_dir / model_filename
                    
                    torch.save({
                        'model_state_dict': model.state_dict(),
                        'model_type': model_type,
                        'tower': tower,
                        'event': event,
                        'fold': fold_idx
                    }, model_path)
    
    print(f"✅ Models saved to: {models_dir}/ ({len(list(models_dir.glob('*.pt')))} files)")
    
    print(f"\n{'='*80}")
    print(f"✅ ALL RESULTS SAVED TO: {output_dir}")
    print(f"{'='*80}\n")
    
    return output_dir


# ==================== MAIN EXECUTION ====================
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'  # ✅ Force only GPU 1 to be visible

config = DeepLearningConfig()

print("="*70)
print("MULTI-EVENT TEMPORAL FORECASTING - DEEP LEARNING")
print("="*70)
print(f"Device: {config.DEVICE}")
print(f"Batch size: {config.BATCH_SIZE}")
print("="*70)

# Clear GPU memory before starting
clear_gpu_memory()
get_gpu_memory_info()

# Run experiments
all_results = run_multi_event_experiments(filtered_dfs, config)


# Save results
output_dir = save_all_results(all_results, config)

print("\n" + "="*80)
print("✅ DEEP LEARNING EXPERIMENTS COMPLETE!")
print("="*80)
#!/usr/bin/env python3
"""
Hyperparameter Optimization for Deep Learning Models using Optuna (Bayesian Optimization)
Optimizes for MCC (Matthews Correlation Coefficient)
With Mixed Precision Training and Parallel Execution
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler  # Mixed precision
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (confusion_matrix, precision_recall_curve, f1_score, 
                             precision_score, recall_score, accuracy_score, 
                             balanced_accuracy_score, matthews_corrcoef, 
                             cohen_kappa_score, average_precision_score, roc_auc_score)
import optuna
from optuna.samplers import TPESampler
from optuna.pruners import MedianPruner
import warnings
import gc
import os
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import argparse
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
import multiprocessing as mp
import threading

warnings.filterwarnings('ignore')
optuna.logging.set_verbosity(optuna.logging.WARNING)  # Reduce optuna verbosity for parallel

# ==================== GPU SETUP ====================
def setup_gpu(gpu_id: int = 0):
    """Setup GPU device"""
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    return device

def clear_gpu_memory():
    """Clear GPU memory cache"""
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()

# ==================== PYTORCH DATASET ====================
class TimeSeriesDataset(Dataset):
    """Custom Dataset for time series data"""
    def __init__(self, X: np.ndarray, y: np.ndarray, sequence_length: int = 24):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y)
        self.sequence_length = sequence_length
        
    def __len__(self):
        return len(self.X) - self.sequence_length + 1
    
    def __getitem__(self, idx):
        X_seq = self.X[idx:idx + self.sequence_length]
        y_target = self.y[idx + self.sequence_length - 1]
        return X_seq, y_target

# ==================== DEEP LEARNING MODELS ====================
class GRUModel(nn.Module):
    """Gated Recurrent Unit model (no sigmoid - use with BCEWithLogitsLoss)"""
    def __init__(self, input_dim: int, hidden_dim: int = 128, 
                 num_layers: int = 2, dropout: float = 0.3,
                 bidirectional: bool = True):
        super(GRUModel, self).__init__()
        
        self.gru = nn.GRU(
            input_dim, hidden_dim, num_layers, 
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
            nn.Linear(32, 1)
            # No Sigmoid - using BCEWithLogitsLoss for mixed precision stability
        )
        
    def forward(self, x):
        gru_out, _ = self.gru(x)
        last_output = gru_out[:, -1, :]
        return self.fc(last_output).squeeze(-1)


class LSTMModel(nn.Module):
    """Long Short-Term Memory model (no sigmoid - use with BCEWithLogitsLoss)"""
    def __init__(self, input_dim: int, hidden_dim: int = 128, 
                 num_layers: int = 2, dropout: float = 0.3,
                 bidirectional: bool = True):
        super(LSTMModel, self).__init__()
        
        self.lstm = nn.LSTM(
            input_dim, hidden_dim, num_layers, 
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
            nn.Linear(32, 1)
            # No Sigmoid - using BCEWithLogitsLoss for mixed precision stability
        )
        
    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]
        return self.fc(last_output).squeeze(-1)


# ==================== HYPERPARAMETER SEARCH SPACE ====================
SEARCH_SPACE = {
    # Sequence and batch parameters - larger batch = faster training
    'sequence_length': [24, 48],  # Reduced: skip 96 (too slow)
    'batch_size': [32, 64, 128],  # Larger batches for GPU efficiency
    
    # Learning parameters
    'learning_rate': (1e-4, 1e-2),  # log-uniform
    'weight_decay': (1e-6, 1e-3),   # log-uniform
    
    # Model architecture - reduced options
    'hidden_dim': [64, 128],  # Skip 256 (slower, often overfits)
    'num_layers': [1, 2],     # Skip 3 (diminishing returns)
    'dropout': (0.1, 0.4),
    'bidirectional': [True, False],
    
    # Training parameters - reduced options
    'optimizer': ['adamw'],  # AdamW generally better, skip adam
    'scheduler': ['cosine', 'none'],  # Skip plateau (slower convergence)
}

# ==================== FORECAST HORIZONS ====================
FORECAST_HORIZONS = {
    '15min': 1, '30min': 2, '1hour': 4, '3hours': 12,
    '6hours': 24, '12hours': 48, '24hours': 96,
}

# ==================== TARGET EVENTS ====================
TARGET_EVENTS = [
    'event_E3_LowTemp_lt0',
    'event_E4_HighWind_Peak_gt25',
    'event_E5_LowWind_lt2',
    
]


def create_model(model_type: str, input_dim: int, params: dict, device: str):
    """Create model based on type and hyperparameters"""
    model_params = {
        'input_dim': input_dim,
        'hidden_dim': params['hidden_dim'],
        'num_layers': params['num_layers'],
        'dropout': params['dropout'],
        'bidirectional': params['bidirectional']
    }
    
    if model_type == 'gru':
        model = GRUModel(**model_params)
    elif model_type == 'lstm':
        model = LSTMModel(**model_params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model.to(device)


def train_and_evaluate(X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray, y_val: np.ndarray,
                       model_type: str, params: dict, device: str,
                       num_epochs: int = 100, early_stopping_patience: int = 15,
                       trial: optuna.Trial = None) -> Tuple[float, dict]:
    """Train model and return MCC score with mixed precision training"""
    
    # Create datasets
    train_dataset = TimeSeriesDataset(X_train, y_train, params['sequence_length'])
    val_dataset = TimeSeriesDataset(X_val, y_val, params['sequence_length'])
    
    if len(train_dataset) < params['batch_size'] or len(val_dataset) < params['batch_size']:
        return -1.0, {}  # Not enough data
    
    # Note: shuffle=False to preserve temporal order in time-series data
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=False,
                              num_workers=4, pin_memory=True, persistent_workers=True)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False,
                            num_workers=4, pin_memory=True, persistent_workers=True)
    
    # Create model
    input_dim = X_train.shape[1]
    model = create_model(model_type, input_dim, params, device)
    
    # Loss function - use BCEWithLogitsLoss for mixed precision stability
    criterion = nn.BCEWithLogitsLoss(reduction='none')
    
    # Mixed precision scaler
    scaler = GradScaler() if device == 'cuda' else None
    use_amp = device == 'cuda'
    
    # Optimizer
    if params['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'], 
                                     weight_decay=params['weight_decay'])
    else:  # adamw
        optimizer = torch.optim.AdamW(model.parameters(), lr=params['learning_rate'], 
                                      weight_decay=params['weight_decay'])
    
    # Scheduler
    if params['scheduler'] == 'plateau':
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', 
                                                                factor=0.5, patience=5)
    elif params['scheduler'] == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)
    else:
        scheduler = None
    
    # Class weights
    n_samples = len(y_train)
    n_pos = y_train.sum()
    n_neg = n_samples - n_pos
    
    if n_pos > 0 and n_neg > 0:
        weight_pos = n_samples / (2 * n_pos)
        weight_neg = n_samples / (2 * n_neg)
        weights = np.where(y_train == 1, weight_pos, weight_neg)
        class_weights = torch.FloatTensor(weights)
    else:
        class_weights = None
    
    # Training loop
    best_val_loss = float('inf')
    best_mcc = -1.0
    patience_counter = 0
    best_model_state = None
    
    for epoch in range(num_epochs):
        # Training with mixed precision
        model.train()
        train_loss = 0.0
        train_sample_count = 0
        
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch = X_batch.to(device, non_blocking=True)
            y_batch = y_batch.to(device, non_blocking=True)
            
            optimizer.zero_grad(set_to_none=True)  # Faster than zero_grad()
            
            # Mixed precision forward pass
            with autocast(enabled=use_amp):
                outputs = model(X_batch)
                loss_per_sample = criterion(outputs, y_batch)
                
                if class_weights is not None:
                    start_idx = batch_idx * params['batch_size']
                    end_idx = min(start_idx + len(X_batch), len(train_dataset))
                    batch_weights = class_weights[start_idx:end_idx].to(device, non_blocking=True)
                    weighted_loss = (loss_per_sample * batch_weights).mean()
                else:
                    weighted_loss = loss_per_sample.mean()
            
            # Mixed precision backward pass
            if scaler is not None:
                scaler.scale(weighted_loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                weighted_loss.backward()
                optimizer.step()
            
            train_loss += weighted_loss.item() * len(X_batch)
            train_sample_count += len(X_batch)
        
        train_loss /= train_sample_count
        
        # Validation with mixed precision
        model.eval()
        val_loss = 0.0
        val_sample_count = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device, non_blocking=True)
                y_batch = y_batch.to(device, non_blocking=True)
                
                with autocast(enabled=use_amp):
                    outputs = model(X_batch)
                    loss = criterion(outputs, y_batch).mean()
                
                val_loss += loss.item() * len(X_batch)
                val_sample_count += len(X_batch)
                
                # Apply sigmoid for BCEWithLogitsLoss
                all_preds.extend(torch.sigmoid(outputs).cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())
        
        val_loss /= val_sample_count
        
        # Calculate MCC
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
        # Find optimal threshold
        precisions, recalls, thresholds = precision_recall_curve(all_targets, all_preds)
        f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
        optimal_idx = np.argmax(f1_scores)
        optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
        
        y_pred_binary = (all_preds >= optimal_threshold).astype(int)
        current_mcc = matthews_corrcoef(all_targets, y_pred_binary)
        
        # Update scheduler
        if scheduler is not None:
            if params['scheduler'] == 'plateau':
                scheduler.step(val_loss)
            else:
                scheduler.step()
        
        # Early stopping based on MCC
        if current_mcc > best_mcc:
            best_mcc = current_mcc
            best_val_loss = val_loss
            patience_counter = 0
            best_model_state = model.state_dict().copy()
        else:
            patience_counter += 1
        
        # Optuna pruning
        if trial is not None:
            trial.report(current_mcc, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()
        
        if patience_counter >= early_stopping_patience:
            break
    
    # Load best model and calculate all metrics
    if best_model_state is not None:
        model.load_state_dict(best_model_state)
    
    model.eval()
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        for X_batch, y_batch in val_loader:
            X_batch = X_batch.to(device, non_blocking=True)
            with autocast(enabled=use_amp):
                outputs = model(X_batch)
            # Apply sigmoid for BCEWithLogitsLoss
            all_preds.extend(torch.sigmoid(outputs).cpu().numpy())
            all_targets.extend(y_batch.cpu().numpy())
    
    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    
    # Calculate all metrics
    precisions, recalls, thresholds = precision_recall_curve(all_targets, all_preds)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    
    y_pred_binary = (all_preds >= optimal_threshold).astype(int)
    
    try:
        tn, fp, fn, tp = confusion_matrix(all_targets, y_pred_binary).ravel()
    except:
        tn, fp, fn, tp = 0, 0, 0, 0
    
    metrics = {
        'mcc': matthews_corrcoef(all_targets, y_pred_binary),
        'accuracy': accuracy_score(all_targets, y_pred_binary),
        'balanced_accuracy': balanced_accuracy_score(all_targets, y_pred_binary),
        'precision': precision_score(all_targets, y_pred_binary, zero_division=0),
        'recall': recall_score(all_targets, y_pred_binary, zero_division=0),
        'f1': f1_score(all_targets, y_pred_binary, zero_division=0),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'cohen_kappa': cohen_kappa_score(all_targets, y_pred_binary),
        'optimal_threshold': optimal_threshold,
        'tp': int(tp), 'tn': int(tn), 'fp': int(fp), 'fn': int(fn)
    }
    
    try:
        metrics['auc_roc'] = roc_auc_score(all_targets, all_preds)
        metrics['auc_pr'] = average_precision_score(all_targets, all_preds)
    except:
        metrics['auc_roc'] = 0.0
        metrics['auc_pr'] = 0.0
    
    # Cleanup
    del model
    clear_gpu_memory()
    
    return best_mcc, metrics


def prepare_data(df: pd.DataFrame, target_col: str, forecast_horizon: str = '6hours'):
    """Prepare features for deep learning models"""
    df = df.copy()
    df = df.sort_index()
    
    # Get numeric columns
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
    
    X = df[numeric_cols].copy()
    
    # Drop columns that are entirely NaN (can't be filled)
    X = X.dropna(axis=1, how='all')
    
    # Handle remaining NaN values using forward-fill then backward-fill (appropriate for time series)
    X = X.ffill().bfill()
    
    # Create target with forecast horizon shift
    horizon_steps = FORECAST_HORIZONS[forecast_horizon]
    
    if df[target_col].dtype == 'object':
        df[target_col] = df[target_col].astype(bool).astype(int)
    elif df[target_col].dtype == 'bool':
        df[target_col] = df[target_col].astype(int)
    
    y = df[target_col].shift(-horizon_steps)
    
    # Remove rows with NaN values (only target NaNs from shifting now)
    valid_idx = X.notna().all(axis=1) & y.notna()
    X = X[valid_idx]
    y = y[valid_idx]
    
    # Ensure all columns are numeric
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = pd.to_numeric(X[col], errors='coerce')
    
    X = X.select_dtypes(include=[np.number])
    
    return X.values, y.values


class OptunaObjective:
    """Optuna objective function for hyperparameter optimization"""
    
    def __init__(self, X: np.ndarray, y: np.ndarray, model_type: str, 
                 device: str, n_splits: int = 3, num_epochs: int = 100,
                 early_stopping_patience: int = 15):
        self.X = X
        self.y = y
        self.model_type = model_type
        self.device = device
        self.n_splits = n_splits
        self.num_epochs = num_epochs
        self.early_stopping_patience = early_stopping_patience
    
    def __call__(self, trial: optuna.Trial) -> float:
        # Sample hyperparameters
        params = {
            'sequence_length': trial.suggest_categorical('sequence_length', SEARCH_SPACE['sequence_length']),
            'batch_size': trial.suggest_categorical('batch_size', SEARCH_SPACE['batch_size']),
            'learning_rate': trial.suggest_float('learning_rate', *SEARCH_SPACE['learning_rate'], log=True),
            'weight_decay': trial.suggest_float('weight_decay', *SEARCH_SPACE['weight_decay'], log=True),
            'hidden_dim': trial.suggest_categorical('hidden_dim', SEARCH_SPACE['hidden_dim']),
            'num_layers': trial.suggest_categorical('num_layers', SEARCH_SPACE['num_layers']),
            'dropout': trial.suggest_float('dropout', *SEARCH_SPACE['dropout']),
            'bidirectional': trial.suggest_categorical('bidirectional', SEARCH_SPACE['bidirectional']),
            'optimizer': trial.suggest_categorical('optimizer', SEARCH_SPACE['optimizer']),
            'scheduler': trial.suggest_categorical('scheduler', SEARCH_SPACE['scheduler']),
        }
        
        # Cross-validation
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        mcc_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(self.X)):
            X_train, X_val = self.X[train_idx], self.X[val_idx]
            y_train, y_val = self.y[train_idx], self.y[val_idx]
            
            mcc, _ = train_and_evaluate(
                X_train, y_train, X_val, y_val,
                self.model_type, params, self.device,
                self.num_epochs, self.early_stopping_patience, trial
            )
            
            if mcc == -1.0:  # Not enough data
                return -1.0
            
            mcc_scores.append(mcc)
            clear_gpu_memory()
        
        return np.mean(mcc_scores)


def save_best_params_incremental(output_dir: Path, best_params_per_event: dict, tower: str, event: str):
    """Save best parameters incrementally as they improve (thread-safe)"""
    best_params_file = output_dir / 'best_params.json'
    lock_file = output_dir / '.best_params.lock'
    
    # Simple file-based locking for thread safety
    import fcntl
    with open(lock_file, 'w') as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
        try:
            # Load existing params if file exists
            if best_params_file.exists():
                with open(best_params_file, 'r') as f:
                    existing_params = json.load(f)
            else:
                existing_params = {}
            
            # Update with new best params
            key = f"{tower}_{event}"
            existing_params[key] = best_params_per_event[key]
            
            # Save updated params
            with open(best_params_file, 'w') as f:
                json.dump(existing_params, f, indent=2)
        finally:
            fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
    
    print(f"      💾 Best params saved to {best_params_file}")


def optimize_single_experiment(args):
    """Run optimization for a single tower-event combination (for parallel execution)"""
    tower_name, tower_df, event_col, model_type, device, n_splits, num_epochs, \
        early_stopping_patience, n_trials, forecast_horizon, output_dir = args
    
    result = None
    best_params_entry = None
    
    try:
        # Prepare data
        X, y = prepare_data(tower_df, event_col, forecast_horizon)
        
        if len(X) < 500:
            print(f"   [{tower_name}/{event_col}] ⚠️  Not enough data ({len(X)} samples), skipping...")
            return None, None
        
        event_rate = y.mean()
        print(f"   [{tower_name}/{event_col}] Starting... Samples: {len(X)}, Event rate: {event_rate:.2%}")
        
        # Create study with aggressive pruning for speed
        study_name = f"{tower_name}_{event_col}_{model_type}"
        study = optuna.create_study(
            study_name=study_name,
            direction='maximize',
            sampler=TPESampler(seed=42, n_startup_trials=3),
            pruner=MedianPruner(n_startup_trials=2, n_warmup_steps=5, interval_steps=3)
        )
        
        # Optimize - use n_jobs for parallel trials within this experiment
        objective = OptunaObjective(
            X, y, model_type, device, n_splits, 
            num_epochs, early_stopping_patience
        )
        
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False, n_jobs=1)
        
        # Get best parameters
        best_params = study.best_params
        best_mcc = study.best_value
        
        print(f"   [{tower_name}/{event_col}] ✅ Best MCC: {best_mcc:.4f}")
        
        # Train final model with best params and get all metrics
        tscv = TimeSeriesSplit(n_splits=n_splits)
        final_metrics_list = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            _, metrics = train_and_evaluate(
                X_train, y_train, X_val, y_val,
                model_type, best_params, device,
                num_epochs, early_stopping_patience
            )
            final_metrics_list.append(metrics)
            clear_gpu_memory()
        
        # Average metrics
        avg_metrics = {}
        for key in final_metrics_list[0].keys():
            if key not in ['tp', 'tn', 'fp', 'fn', 'optimal_threshold']:
                avg_metrics[key] = np.mean([m[key] for m in final_metrics_list])
                avg_metrics[f'{key}_std'] = np.std([m[key] for m in final_metrics_list])
        
        # Store results
        result = {
            'tower': tower_name,
            'event': event_col,
            'model_type': model_type,
            'best_mcc': best_mcc,
            'n_samples': len(X),
            'event_rate': event_rate,
            **best_params,
            **avg_metrics
        }
        
        # Store best params entry
        best_params_entry = {
            'key': f"{tower_name}_{event_col}",
            'data': {
                'params': best_params,
                'mcc': best_mcc,
                'metrics': avg_metrics
            }
        }
        
        # Save study results
        study_df = study.trials_dataframe()
        study_df.to_csv(output_dir / f"trials_{study_name}.csv", index=False)
        
    except Exception as e:
        print(f"   [{tower_name}/{event_col}] ❌ Error: {str(e)}")
        return None, None
    
    return result, best_params_entry


def run_optimization(filtered_dfs: Dict[str, pd.DataFrame], 
                     model_type: str = 'gru',
                     n_trials: int = 3,
                     n_splits: int = 3,
                     num_epochs: int = 100,
                     early_stopping_patience: int = 15,
                     forecast_horizon: str = '6hours',
                     gpu_id: int = 0,
                     output_dir: str = None,
                     n_parallel: int = None):
    """Run Bayesian hyperparameter optimization with parallel execution
    
    Args:
        n_trials: Number of configurations to try per experiment
        n_parallel: Number of parallel experiments (default: number of CPU cores / 4)
    """
    
    device = setup_gpu(gpu_id)
    
    # Determine parallelism - use threads for GPU sharing
    if n_parallel is None:
        n_parallel = max(1, mp.cpu_count() // 4)  # Conservative for GPU memory
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        output_dir = Path(f"optuna_results_{model_type}_{timestamp}")
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    print("\n" + "="*80)
    print(f"🚀 PARALLEL HYPERPARAMETER OPTIMIZATION - {model_type.upper()}")
    print("="*80)
    print(f"Optimization metric: MCC (Matthews Correlation Coefficient)")
    print(f"Number of trials per experiment: {n_trials}")
    print(f"Parallel experiments: {n_parallel}")
    print(f"CV folds: {n_splits}")
    print(f"Forecast horizon: {forecast_horizon}")
    print(f"Device: {device}")
    print(f"Mixed precision: ENABLED")
    print(f"Output: {output_dir}")
    print("="*80)
    
    # Build list of all experiments
    experiments = []
    for tower_name, tower_df in filtered_dfs.items():
        for event_col in TARGET_EVENTS:
            if event_col in tower_df.columns:
                experiments.append((
                    tower_name, tower_df, event_col, model_type, device, n_splits,
                    num_epochs, early_stopping_patience, n_trials, forecast_horizon, output_dir
                ))
    
    print(f"\n📋 Total experiments to run: {len(experiments)}")
    print(f"   Running {n_parallel} experiments in parallel...\n")
    
    all_results = []
    best_params_per_event = {}
    
    # Run experiments in parallel using ThreadPoolExecutor (shares GPU)
    with ThreadPoolExecutor(max_workers=n_parallel) as executor:
        futures = {executor.submit(optimize_single_experiment, exp): exp for exp in experiments}
        
        for future in as_completed(futures):
            result, best_params_entry = future.result()
            
            if result is not None:
                all_results.append(result)
                
                if best_params_entry is not None:
                    best_params_per_event[best_params_entry['key']] = best_params_entry['data']
                    # Save incrementally
                    save_best_params_incremental(
                        output_dir, best_params_per_event,
                        result['tower'], result['event']
                    )
                
                # Save running results
                pd.DataFrame(all_results).to_csv(output_dir / 'optimization_results.csv', index=False)
    
    # Save final results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_dir / 'optimization_results.csv', index=False)
    
    # Final save of best parameters
    with open(output_dir / 'best_params.json', 'w') as f:
        json.dump(best_params_per_event, f, indent=2)
    
    # Create summary
    summary = {
        'timestamp': timestamp,
        'model_type': model_type,
        'n_trials': n_trials,
        'n_splits': n_splits,
        'num_epochs': num_epochs,
        'early_stopping_patience': early_stopping_patience,
        'forecast_horizon': forecast_horizon,
        'search_space': {k: str(v) for k, v in SEARCH_SPACE.items()},
        'total_experiments': len(all_results)
    }
    
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(summary, f, indent=2)
    
    print("\n" + "="*80)
    print("✅ OPTIMIZATION COMPLETE!")
    print("="*80)
    print(f"Results saved to: {output_dir}")
    
    # Print best results per event
    if len(results_df) > 0:
        print("\n📊 Best MCC per Event (averaged across towers):")
        event_summary = results_df.groupby('event')['best_mcc'].agg(['mean', 'std', 'max']).round(4)
        print(event_summary)
    
    return results_df, best_params_per_event


def load_and_prepare_data():
    """Load and prepare the weather data"""
    print("Loading data...")
    
    # Read the CSV file
    df = pd.read_csv("fully_labeled_weather_data_with_events.csv")
    
    # Move 'tower' and 'timestamp' to the first two columns
    cols = df.columns.tolist()
    priority_cols = ['tower', 'timestamp']
    remaining_cols = [c for c in cols if c not in priority_cols]
    df = df[priority_cols + remaining_cols]
    
    # Drop columns ending with '_min' or '_meets_duration'
    df = df.loc[:, ~df.columns.str.endswith(('_min', '_meets_duration'))]
    
    # Remove these columns
    cols_to_drop = ['event_count', 'active_events', 'event_durations', 'has_any_event']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    # Ensure timestamp is datetime
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Split into separate DataFrames per tower
    filtered_dfs = {tower: df[df['tower'] == tower].copy() for tower in df['tower'].unique()}
    
    print(f"Loaded data for {len(filtered_dfs)} towers")
    for tower, tower_df in filtered_dfs.items():
        print(f"  {tower}: {len(tower_df)} samples")
    
    return filtered_dfs


def main():
    parser = argparse.ArgumentParser(description='Parallel Hyperparameter Optimization for DL Models with Mixed Precision')
    parser.add_argument('--model', type=str, default='gru', choices=['gru', 'lstm', 'both'],
                        help='Model type to optimize')
    parser.add_argument('--n_trials', type=int, default=3,
                        help='Number of Optuna trials per experiment')
    parser.add_argument('--n_splits', type=int, default=3,
                        help='Number of CV splits')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Max epochs per trial (reduced for speed)')
    parser.add_argument('--patience', type=int, default=8,
                        help='Early stopping patience (reduced for speed)')
    parser.add_argument('--horizon', type=str, default='6hours',
                        choices=list(FORECAST_HORIZONS.keys()),
                        help='Forecast horizon')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory')
    parser.add_argument('--parallel', type=int, default=None,
                        help='Number of parallel experiments (default: auto based on CPU cores)')
    
    args = parser.parse_args()
    
    # Load data
    filtered_dfs = load_and_prepare_data()
    
    # Run optimization
    if args.model == 'both':
        for model_type in ['gru', 'lstm']:
            run_optimization(
                filtered_dfs,
                model_type=model_type,
                n_trials=args.n_trials,
                n_splits=args.n_splits,
                num_epochs=args.epochs,
                early_stopping_patience=args.patience,
                forecast_horizon=args.horizon,
                gpu_id=args.gpu,
                output_dir=args.output,
                n_parallel=args.parallel
            )
    else:
        run_optimization(
            filtered_dfs,
            model_type=args.model,
            n_trials=args.n_trials,
            n_splits=args.n_splits,
            num_epochs=args.epochs,
            early_stopping_patience=args.patience,
            forecast_horizon=args.horizon,
            gpu_id=args.gpu,
            output_dir=args.output,
            n_parallel=args.parallel
        )


if __name__ == "__main__":
    main()

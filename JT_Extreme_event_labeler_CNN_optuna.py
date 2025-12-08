#!/usr/bin/env python3
"""
Hyperparameter Optimization for CNN/TCN Models using Optuna (Bayesian Optimization)
Optimizes for MCC (Matthews Correlation Coefficient)
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
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

warnings.filterwarnings('ignore')

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

class TemporalBlock(nn.Module):
    """A single temporal block for TCN with causal convolutions"""
    def __init__(self, in_channels, out_channels, kernel_size, stride, dilation, padding, dropout):
        super(TemporalBlock, self).__init__()
        self.padding = padding
        
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size,
                               stride=stride, padding=padding, dilation=dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
        self.relu = nn.ReLU()

    def forward(self, x):
        out = self.conv1(x)
        out = out[:, :, :-self.padding] if self.padding > 0 else out
        out = self.dropout1(self.relu1(self.bn1(out)))
        
        out = self.conv2(out)
        out = out[:, :, :-self.padding] if self.padding > 0 else out
        out = self.dropout2(self.relu2(self.bn2(out)))
        
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TCNModel(nn.Module):
    """Temporal Convolutional Network - Pure PyTorch Implementation"""
    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 num_layers: int = 4, dropout: float = 0.3,
                 kernel_size: int = 3):
        super(TCNModel, self).__init__()
        
        layers = []
        num_channels = [hidden_dim] * num_layers
        
        for i in range(num_layers):
            dilation = 2 ** i
            in_channels = input_dim if i == 0 else num_channels[i-1]
            out_channels = num_channels[i]
            padding = (kernel_size - 1) * dilation
            
            layers.append(TemporalBlock(in_channels, out_channels, kernel_size,
                                        stride=1, dilation=dilation,
                                        padding=padding, dropout=dropout))
        
        self.tcn = nn.Sequential(*layers)
        
        self.fc = nn.Sequential(
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        tcn_out = self.tcn(x)
        last_output = tcn_out[:, :, -1]
        output = self.fc(last_output)
        return output.squeeze(-1)


class CNNModel(nn.Module):
    """1D Convolutional Neural Network for time series classification"""
    def __init__(self, input_dim: int, hidden_dim: int = 128,
                 num_layers: int = 3, dropout: float = 0.3,
                 kernel_size: int = 3):
        super(CNNModel, self).__init__()
        
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        
        layers = []
        in_channels = input_dim
        
        for i in range(num_layers):
            out_channels = hidden_dim * (2 ** min(i, 2))
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.MaxPool1d(kernel_size=2, stride=1, padding=1)
            ])
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*layers)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        self.fc = nn.Sequential(
            nn.Linear(in_channels, 64),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
    def forward(self, x):
        x = x.permute(0, 2, 1)
        x = self.conv_layers(x)
        x = self.global_pool(x).squeeze(-1)
        output = self.fc(x)
        return output.squeeze(-1)


# ==================== HYPERPARAMETER SEARCH SPACE ====================
SEARCH_SPACE = {
    # Sequence and batch parameters
    'sequence_length': [24, 48, 96],  # 6h, 12h, 24h of history
    'batch_size': [16, 32, 64],
    
    # Learning parameters
    'learning_rate': (1e-4, 1e-2),  # log-uniform
    'weight_decay': (1e-6, 1e-3),   # log-uniform
    
    # Model architecture
    'hidden_dim': [64, 128, 256],
    'num_layers': [2, 3, 4, 5],
    'dropout': (0.1, 0.5),
    'kernel_size': [3, 5, 7],
    
    # Training parameters
    'optimizer': ['adam', 'adamw'],
    'scheduler': ['plateau', 'cosine', 'none'],
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
        'kernel_size': params['kernel_size']
    }
    
    if model_type == 'cnn':
        model = CNNModel(**model_params)
    elif model_type == 'tcn':
        model = TCNModel(**model_params)
    else:
        raise ValueError(f"Unknown model type: {model_type}")
    
    return model.to(device)


def train_and_evaluate(X_train: np.ndarray, y_train: np.ndarray,
                       X_val: np.ndarray, y_val: np.ndarray,
                       model_type: str, params: dict, device: str,
                       num_epochs: int = 100, early_stopping_patience: int = 15,
                       trial: optuna.Trial = None) -> Tuple[float, dict]:
    """Train model and return MCC score"""
    
    # Create datasets
    train_dataset = TimeSeriesDataset(X_train, y_train, params['sequence_length'])
    val_dataset = TimeSeriesDataset(X_val, y_val, params['sequence_length'])
    
    if len(train_dataset) < params['batch_size'] or len(val_dataset) < params['batch_size']:
        return -1.0, {}
    
    train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=params['batch_size'], shuffle=False)
    
    # Create model
    input_dim = X_train.shape[1]
    model = create_model(model_type, input_dim, params, device)
    
    # Loss function
    criterion = nn.BCELoss(reduction='none')
    
    # Optimizer
    if params['optimizer'] == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'], 
                                     weight_decay=params['weight_decay'])
    else:
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
        # Training
        model.train()
        train_loss = 0.0
        train_sample_count = 0
        
        for batch_idx, (X_batch, y_batch) in enumerate(train_loader):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            
            optimizer.zero_grad()
            outputs = model(X_batch)
            loss_per_sample = criterion(outputs, y_batch)
            
            if class_weights is not None:
                start_idx = batch_idx * params['batch_size']
                end_idx = min(start_idx + len(X_batch), len(train_dataset))
                batch_weights = class_weights[start_idx:end_idx].to(device)
                weighted_loss = (loss_per_sample * batch_weights).mean()
            else:
                weighted_loss = loss_per_sample.mean()
            
            weighted_loss.backward()
            optimizer.step()
            
            train_loss += weighted_loss.item() * len(X_batch)
            train_sample_count += len(X_batch)
        
        train_loss /= train_sample_count
        
        # Validation
        model.eval()
        val_loss = 0.0
        val_sample_count = 0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device)
                
                outputs = model(X_batch)
                loss = criterion(outputs, y_batch).mean()
                
                val_loss += loss.item() * len(X_batch)
                val_sample_count += len(X_batch)
                
                all_preds.extend(outputs.cpu().numpy())
                all_targets.extend(y_batch.cpu().numpy())
        
        val_loss /= val_sample_count
        
        # Calculate MCC
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
        
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
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            all_preds.extend(outputs.cpu().numpy())
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
    """Prepare features for CNN/TCN models"""
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
    
    # Remove rows with NaN values
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
            'kernel_size': trial.suggest_categorical('kernel_size', SEARCH_SPACE['kernel_size']),
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
            
            if mcc == -1.0:
                return -1.0
            
            mcc_scores.append(mcc)
            clear_gpu_memory()
        
        return np.mean(mcc_scores)


def save_best_params_incremental(output_dir: Path, best_params_per_event: dict, tower: str, event: str):
    """Save best parameters incrementally as they improve"""
    best_params_file = output_dir / 'best_params.json'
    
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
    
    print(f"      💾 Best params saved to {best_params_file}")


def run_optimization(filtered_dfs: Dict[str, pd.DataFrame], 
                     model_type: str = 'cnn',
                     n_trials: int = 3,  # Reduced to 3 trials for faster execution
                     n_splits: int = 3,
                     num_epochs: int = 100,
                     early_stopping_patience: int = 15,
                     forecast_horizon: str = '6hours',
                     gpu_id: int = 0,
                     output_dir: str = None):
    """Run Bayesian hyperparameter optimization with limited trials
    
    Args:
        n_trials: Number of configurations to try (default: 3 for quick runs)
    """
    
    device = setup_gpu(gpu_id)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        output_dir = Path(f"optuna_cnn_results_{model_type}_{timestamp}")
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    all_results = []
    best_params_per_event = {}
    
    print("\n" + "="*80)
    print(f"FAST HYPERPARAMETER OPTIMIZATION - {model_type.upper()}")
    print("="*80)
    print(f"Optimization metric: MCC (Matthews Correlation Coefficient)")
    print(f"Number of trials: {n_trials} (quick mode)")
    print(f"CV folds: {n_splits}")
    print(f"Forecast horizon: {forecast_horizon}")
    print(f"Device: {device}")
    print(f"Best params saved incrementally to: {output_dir / 'best_params.json'}")
    print("="*80)
    
    for tower_name, tower_df in filtered_dfs.items():
        print(f"\n{'='*70}")
        print(f"🏗️  TOWER: {tower_name}")
        print(f"{'='*70}")
        
        for event_col in TARGET_EVENTS:
            if event_col not in tower_df.columns:
                print(f"   ⚠️  {event_col} not found, skipping...")
                continue
            
            print(f"\n   📍 Event: {event_col}")
            print(f"      Optimizing {model_type.upper()}...")
            
            try:
                # Prepare data
                X, y = prepare_data(tower_df, event_col, forecast_horizon)
                
                if len(X) < 500:
                    print(f"      ⚠️  Not enough data ({len(X)} samples), skipping...")
                    continue
                
                event_rate = y.mean()
                print(f"      Samples: {len(X)}, Event rate: {event_rate:.2%}")
                
                # Create study
                study_name = f"{tower_name}_{event_col}_{model_type}"
                study = optuna.create_study(
                    study_name=study_name,
                    direction='maximize',
                    sampler=TPESampler(seed=42),
                    pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=10)
                )
                
                # Optimize
                objective = OptunaObjective(
                    X, y, model_type, device, n_splits, 
                    num_epochs, early_stopping_patience
                )
                
                study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
                
                # Get best parameters
                best_params = study.best_params
                best_mcc = study.best_value
                
                print(f"      ✅ Best MCC: {best_mcc:.4f}")
                print(f"      Best params: {best_params}")
                
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
                all_results.append(result)
                
                # Store best params
                key = f"{tower_name}_{event_col}"
                best_params_per_event[key] = {
                    'params': best_params,
                    'mcc': best_mcc,
                    'metrics': avg_metrics
                }
                
                # Save best params incrementally (so we don't lose progress)
                save_best_params_incremental(output_dir, best_params_per_event, tower_name, event_col)
                
                # Save study results
                study_df = study.trials_dataframe()
                study_df.to_csv(output_dir / f"trials_{study_name}.csv", index=False)
                
                # Also save running results CSV
                pd.DataFrame(all_results).to_csv(output_dir / 'optimization_results.csv', index=False)
                
            except Exception as e:
                print(f"      ❌ Error: {str(e)}")
                continue
    
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
    
    if len(results_df) > 0:
        print("\n📊 Best MCC per Event (averaged across towers):")
        event_summary = results_df.groupby('event')['best_mcc'].agg(['mean', 'std', 'max']).round(4)
        print(event_summary)
    
    return results_df, best_params_per_event


def load_and_prepare_data():
    """Load and prepare the weather data"""
    print("Loading data...")
    
    df = pd.read_csv("fully_labeled_weather_data_with_events.csv")
    
    cols = df.columns.tolist()
    priority_cols = ['tower', 'timestamp']
    remaining_cols = [c for c in cols if c not in priority_cols]
    df = df[priority_cols + remaining_cols]
    
    df = df.loc[:, ~df.columns.str.endswith(('_min', '_meets_duration'))]
    
    cols_to_drop = ['event_count', 'active_events', 'event_durations', 'has_any_event']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])
    
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    filtered_dfs = {tower: df[df['tower'] == tower].copy() for tower in df['tower'].unique()}
    
    print(f"Loaded data for {len(filtered_dfs)} towers")
    for tower, tower_df in filtered_dfs.items():
        print(f"  {tower}: {len(tower_df)} samples")
    
    return filtered_dfs


def main():
    parser = argparse.ArgumentParser(description='Fast Hyperparameter Optimization for CNN/TCN Models')
    parser.add_argument('--model', type=str, default='cnn', choices=['cnn', 'tcn', 'both'],
                        help='Model type to optimize')
    parser.add_argument('--n_trials', type=int, default=3,
                        help='Number of Optuna trials (default: 3 for quick runs)')
    parser.add_argument('--n_splits', type=int, default=3,
                        help='Number of CV splits')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Max epochs per trial')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')
    parser.add_argument('--horizon', type=str, default='6hours',
                        choices=list(FORECAST_HORIZONS.keys()),
                        help='Forecast horizon')
    parser.add_argument('--gpu', type=int, default=0,
                        help='GPU device ID')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory')
    
    args = parser.parse_args()
    
    # Load data
    filtered_dfs = load_and_prepare_data()
    
    # Run optimization
    if args.model == 'both':
        for model_type in ['cnn', 'tcn']:
            run_optimization(
                filtered_dfs,
                model_type=model_type,
                n_trials=args.n_trials,
                n_splits=args.n_splits,
                num_epochs=args.epochs,
                early_stopping_patience=args.patience,
                forecast_horizon=args.horizon,
                gpu_id=args.gpu,
                output_dir=args.output
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
            output_dir=args.output
        )


if __name__ == "__main__":
    main()

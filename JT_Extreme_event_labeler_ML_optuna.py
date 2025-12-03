#!/usr/bin/env python3
"""
Hyperparameter Optimization for ML Models (LightGBM & XGBoost) using Optuna (Bayesian Optimization)
Optimizes for MCC (Matthews Correlation Coefficient)
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (confusion_matrix, precision_recall_curve, f1_score, 
                             precision_score, recall_score, accuracy_score, 
                             balanced_accuracy_score, matthews_corrcoef, 
                             cohen_kappa_score, average_precision_score, roc_auc_score)
import lightgbm as lgb
import xgboost as xgb
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

# ==================== HYPERPARAMETER SEARCH SPACE ====================
LGBM_SEARCH_SPACE = {
    # Tree structure
    'num_leaves': (16, 128),
    'max_depth': (3, 12),
    'min_child_samples': (5, 100),
    'min_child_weight': (1e-4, 1e-1),
    
    # Regularization
    'reg_alpha': (1e-8, 10.0),  # L1
    'reg_lambda': (1e-8, 10.0), # L2
    'min_split_gain': (0.0, 1.0),
    
    # Sampling
    'feature_fraction': (0.5, 1.0),
    'bagging_fraction': (0.5, 1.0),
    'bagging_freq': [1, 3, 5, 7],
    
    # Learning
    'learning_rate': (0.01, 0.3),
    'n_estimators': [100, 300, 500, 1000, 1500],
}

XGB_SEARCH_SPACE = {
    # Tree structure
    'max_depth': (3, 12),
    'min_child_weight': (1, 10),
    'max_delta_step': (0, 10),
    
    # Regularization
    'reg_alpha': (1e-8, 10.0),  # L1
    'reg_lambda': (1e-8, 10.0), # L2
    'gamma': (0.0, 1.0),
    
    # Sampling
    'subsample': (0.5, 1.0),
    'colsample_bytree': (0.5, 1.0),
    'colsample_bylevel': (0.5, 1.0),
    'colsample_bynode': (0.5, 1.0),
    
    # Learning
    'learning_rate': (0.01, 0.3),
    'n_estimators': [100, 300, 500, 1000, 1500],
}

# Common feature engineering search space
FEATURE_SEARCH_SPACE = {
    'lag_config': ['short', 'medium', 'long'],
    'use_rolling': [True, False],
    'rolling_windows': [[4, 12], [4, 12, 24], [4, 12, 24, 96]],
}

LAG_CONFIGS = {
    'short': [1, 2, 4, 8],
    'medium': [1, 2, 4, 8, 16, 24],
    'long': [1, 4, 8, 16, 24, 48, 96],
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
    'event_E6_HighTemp_gt24'
]


def prepare_temporal_features(df: pd.DataFrame, target_col: str, 
                              forecast_horizon: str = '6hours',
                              lag_config: str = 'medium',
                              use_rolling: bool = True,
                              rolling_windows: List[int] = [4, 12, 24, 96]) -> Tuple[np.ndarray, np.ndarray]:
    """Create temporal features (lags, rolling stats) for forecasting"""
    
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
    
    # Drop columns that are entirely NaN (can't be filled)
    valid_cols = [c for c in numeric_cols if not df[c].isna().all()]
    numeric_cols = valid_cols
    
    # Handle NaN values using forward-fill then backward-fill (appropriate for time series)
    for col in numeric_cols:
        df[col] = df[col].ffill().bfill()
    
    feature_dfs = []
    
    # Original features
    feature_dfs.append(df[numeric_cols])
    
    # Lag features
    lags = LAG_CONFIGS[lag_config]
    for col in numeric_cols:
        for lag in lags:
            lag_col = f'{col}_lag{lag}'
            feature_dfs.append(df[col].shift(lag).to_frame(lag_col))
    
    # Rolling window features
    if use_rolling:
        for col in numeric_cols:
            for window in rolling_windows:
                roll_mean_col = f'{col}_roll{window}_mean'
                feature_dfs.append(df[col].rolling(window=window).mean().to_frame(roll_mean_col))
                
                roll_std_col = f'{col}_roll{window}_std'
                feature_dfs.append(df[col].rolling(window=window).std().to_frame(roll_std_col))
    
    # Combine all features
    X = pd.concat(feature_dfs, axis=1)
    
    # Target: shift forward by forecast horizon
    horizon_steps = FORECAST_HORIZONS[forecast_horizon]
    
    if df[target_col].dtype == 'object':
        df[target_col] = df[target_col].astype(bool).astype(int)
    elif df[target_col].dtype == 'bool':
        df[target_col] = df[target_col].astype(int)
    
    y = df[target_col].shift(-horizon_steps)
    
    # Drop rows with NaN
    valid_idx = X.notna().all(axis=1) & y.notna()
    X = X[valid_idx]
    y = y[valid_idx]
    
    # Ensure all columns are numeric
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = pd.to_numeric(X[col], errors='coerce')
    
    X = X.select_dtypes(include=[np.number])
    
    return X.values, y.values


def train_and_evaluate_lgbm(X_train: np.ndarray, y_train: np.ndarray,
                            X_val: np.ndarray, y_val: np.ndarray,
                            params: dict, use_gpu: bool = False) -> Tuple[float, dict]:
    """Train LightGBM model and return MCC score"""
    
    # Calculate sample weights
    n_samples = len(y_train)
    n_pos = y_train.sum()
    n_neg = n_samples - n_pos
    
    if n_pos > 0 and n_neg > 0:
        weight_pos = n_samples / (2 * n_pos)
        weight_neg = n_samples / (2 * n_neg)
        sample_weights = np.where(y_train == 1, weight_pos, weight_neg)
    else:
        sample_weights = None
    
    train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weights)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    lgb_params = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'is_unbalance': True,
        'device': 'gpu' if use_gpu else 'cpu',
        'verbose': -1,
        'random_state': 42,
        **{k: v for k, v in params.items() if k not in ['n_estimators', 'lag_config', 'use_rolling', 'rolling_windows']}
    }
    
    n_estimators = params.get('n_estimators', 1000)
    
    model = lgb.train(
        lgb_params,
        train_data,
        num_boost_round=n_estimators,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=[lgb.early_stopping(stopping_rounds=50, verbose=False)]
    )
    
    # Predict
    y_pred_proba = model.predict(X_val)
    
    # Find optimal threshold
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_pred_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    
    y_pred = (y_pred_proba >= optimal_threshold).astype(int)
    
    # Calculate metrics
    try:
        tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
    except:
        tn, fp, fn, tp = 0, 0, 0, 0
    
    metrics = {
        'mcc': matthews_corrcoef(y_val, y_pred),
        'accuracy': accuracy_score(y_val, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred, zero_division=0),
        'recall': recall_score(y_val, y_pred, zero_division=0),
        'f1': f1_score(y_val, y_pred, zero_division=0),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'cohen_kappa': cohen_kappa_score(y_val, y_pred),
        'optimal_threshold': optimal_threshold,
    }
    
    try:
        metrics['auc_roc'] = roc_auc_score(y_val, y_pred_proba)
        metrics['auc_pr'] = average_precision_score(y_val, y_pred_proba)
    except:
        metrics['auc_roc'] = 0.0
        metrics['auc_pr'] = 0.0
    
    return metrics['mcc'], metrics


def train_and_evaluate_xgb(X_train: np.ndarray, y_train: np.ndarray,
                           X_val: np.ndarray, y_val: np.ndarray,
                           params: dict, use_gpu: bool = False) -> Tuple[float, dict]:
    """Train XGBoost model and return MCC score"""
    
    # Calculate sample weights
    n_samples = len(y_train)
    n_pos = y_train.sum()
    n_neg = n_samples - n_pos
    
    if n_pos > 0 and n_neg > 0:
        weight_pos = n_samples / (2 * n_pos)
        weight_neg = n_samples / (2 * n_neg)
        sample_weights = np.where(y_train == 1, weight_pos, weight_neg)
    else:
        sample_weights = None
    
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weights)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    xgb_params = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'tree_method': 'gpu_hist' if use_gpu else 'hist',
        'verbosity': 0,
        'random_state': 42,
        **{k: v for k, v in params.items() if k not in ['n_estimators', 'lag_config', 'use_rolling', 'rolling_windows']}
    }
    
    n_estimators = params.get('n_estimators', 1000)
    
    evals = [(dtrain, 'train'), (dval, 'val')]
    
    model = xgb.train(
        xgb_params,
        dtrain,
        num_boost_round=n_estimators,
        evals=evals,
        early_stopping_rounds=50,
        verbose_eval=False
    )
    
    # Predict
    y_pred_proba = model.predict(dval)
    
    # Find optimal threshold
    precisions, recalls, thresholds = precision_recall_curve(y_val, y_pred_proba)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    optimal_idx = np.argmax(f1_scores)
    optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
    
    y_pred = (y_pred_proba >= optimal_threshold).astype(int)
    
    # Calculate metrics
    try:
        tn, fp, fn, tp = confusion_matrix(y_val, y_pred).ravel()
    except:
        tn, fp, fn, tp = 0, 0, 0, 0
    
    metrics = {
        'mcc': matthews_corrcoef(y_val, y_pred),
        'accuracy': accuracy_score(y_val, y_pred),
        'balanced_accuracy': balanced_accuracy_score(y_val, y_pred),
        'precision': precision_score(y_val, y_pred, zero_division=0),
        'recall': recall_score(y_val, y_pred, zero_division=0),
        'f1': f1_score(y_val, y_pred, zero_division=0),
        'specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'cohen_kappa': cohen_kappa_score(y_val, y_pred),
        'optimal_threshold': optimal_threshold,
    }
    
    try:
        metrics['auc_roc'] = roc_auc_score(y_val, y_pred_proba)
        metrics['auc_pr'] = average_precision_score(y_val, y_pred_proba)
    except:
        metrics['auc_roc'] = 0.0
        metrics['auc_pr'] = 0.0
    
    return metrics['mcc'], metrics


class LGBMObjective:
    """Optuna objective for LightGBM"""
    
    def __init__(self, df: pd.DataFrame, target_col: str, forecast_horizon: str,
                 n_splits: int = 3, use_gpu: bool = False):
        self.df = df
        self.target_col = target_col
        self.forecast_horizon = forecast_horizon
        self.n_splits = n_splits
        self.use_gpu = use_gpu
    
    def __call__(self, trial: optuna.Trial) -> float:
        # Sample hyperparameters
        params = {
            # Feature engineering
            'lag_config': trial.suggest_categorical('lag_config', FEATURE_SEARCH_SPACE['lag_config']),
            'use_rolling': trial.suggest_categorical('use_rolling', FEATURE_SEARCH_SPACE['use_rolling']),
            'rolling_windows': trial.suggest_categorical('rolling_windows', 
                                                         [str(w) for w in FEATURE_SEARCH_SPACE['rolling_windows']]),
            
            # LightGBM params
            'num_leaves': trial.suggest_int('num_leaves', *LGBM_SEARCH_SPACE['num_leaves']),
            'max_depth': trial.suggest_int('max_depth', *LGBM_SEARCH_SPACE['max_depth']),
            'min_child_samples': trial.suggest_int('min_child_samples', *LGBM_SEARCH_SPACE['min_child_samples']),
            'min_child_weight': trial.suggest_float('min_child_weight', *LGBM_SEARCH_SPACE['min_child_weight'], log=True),
            'reg_alpha': trial.suggest_float('reg_alpha', *LGBM_SEARCH_SPACE['reg_alpha'], log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', *LGBM_SEARCH_SPACE['reg_lambda'], log=True),
            'min_split_gain': trial.suggest_float('min_split_gain', *LGBM_SEARCH_SPACE['min_split_gain']),
            'feature_fraction': trial.suggest_float('feature_fraction', *LGBM_SEARCH_SPACE['feature_fraction']),
            'bagging_fraction': trial.suggest_float('bagging_fraction', *LGBM_SEARCH_SPACE['bagging_fraction']),
            'bagging_freq': trial.suggest_categorical('bagging_freq', LGBM_SEARCH_SPACE['bagging_freq']),
            'learning_rate': trial.suggest_float('learning_rate', *LGBM_SEARCH_SPACE['learning_rate'], log=True),
            'n_estimators': trial.suggest_categorical('n_estimators', LGBM_SEARCH_SPACE['n_estimators']),
        }
        
        # Parse rolling windows
        rolling_windows = eval(params['rolling_windows'])
        
        # Prepare data with sampled feature engineering params
        X, y = prepare_temporal_features(
            self.df, self.target_col, self.forecast_horizon,
            params['lag_config'], params['use_rolling'], rolling_windows
        )
        
        if len(X) < 500:
            return -1.0
        
        # Cross-validation
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        mcc_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            mcc, _ = train_and_evaluate_lgbm(X_train, y_train, X_val, y_val, params, self.use_gpu)
            mcc_scores.append(mcc)
        
        return np.mean(mcc_scores)


class XGBObjective:
    """Optuna objective for XGBoost"""
    
    def __init__(self, df: pd.DataFrame, target_col: str, forecast_horizon: str,
                 n_splits: int = 3, use_gpu: bool = False):
        self.df = df
        self.target_col = target_col
        self.forecast_horizon = forecast_horizon
        self.n_splits = n_splits
        self.use_gpu = use_gpu
    
    def __call__(self, trial: optuna.Trial) -> float:
        # Sample hyperparameters
        params = {
            # Feature engineering
            'lag_config': trial.suggest_categorical('lag_config', FEATURE_SEARCH_SPACE['lag_config']),
            'use_rolling': trial.suggest_categorical('use_rolling', FEATURE_SEARCH_SPACE['use_rolling']),
            'rolling_windows': trial.suggest_categorical('rolling_windows', 
                                                         [str(w) for w in FEATURE_SEARCH_SPACE['rolling_windows']]),
            
            # XGBoost params
            'max_depth': trial.suggest_int('max_depth', *XGB_SEARCH_SPACE['max_depth']),
            'min_child_weight': trial.suggest_int('min_child_weight', *XGB_SEARCH_SPACE['min_child_weight']),
            'max_delta_step': trial.suggest_int('max_delta_step', *XGB_SEARCH_SPACE['max_delta_step']),
            'reg_alpha': trial.suggest_float('reg_alpha', *XGB_SEARCH_SPACE['reg_alpha'], log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', *XGB_SEARCH_SPACE['reg_lambda'], log=True),
            'gamma': trial.suggest_float('gamma', *XGB_SEARCH_SPACE['gamma']),
            'subsample': trial.suggest_float('subsample', *XGB_SEARCH_SPACE['subsample']),
            'colsample_bytree': trial.suggest_float('colsample_bytree', *XGB_SEARCH_SPACE['colsample_bytree']),
            'colsample_bylevel': trial.suggest_float('colsample_bylevel', *XGB_SEARCH_SPACE['colsample_bylevel']),
            'colsample_bynode': trial.suggest_float('colsample_bynode', *XGB_SEARCH_SPACE['colsample_bynode']),
            'learning_rate': trial.suggest_float('learning_rate', *XGB_SEARCH_SPACE['learning_rate'], log=True),
            'n_estimators': trial.suggest_categorical('n_estimators', XGB_SEARCH_SPACE['n_estimators']),
        }
        
        # Parse rolling windows
        rolling_windows = eval(params['rolling_windows'])
        
        # Prepare data with sampled feature engineering params
        X, y = prepare_temporal_features(
            self.df, self.target_col, self.forecast_horizon,
            params['lag_config'], params['use_rolling'], rolling_windows
        )
        
        if len(X) < 500:
            return -1.0
        
        # Cross-validation
        tscv = TimeSeriesSplit(n_splits=self.n_splits)
        mcc_scores = []
        
        for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
            X_train, X_val = X[train_idx], X[val_idx]
            y_train, y_val = y[train_idx], y[val_idx]
            
            mcc, _ = train_and_evaluate_xgb(X_train, y_train, X_val, y_val, params, self.use_gpu)
            mcc_scores.append(mcc)
        
        return np.mean(mcc_scores)


def run_optimization(filtered_dfs: Dict[str, pd.DataFrame], 
                     model_type: str = 'lightgbm',
                     n_trials: int = 50,
                     n_splits: int = 3,
                     forecast_horizon: str = '6hours',
                     use_gpu: bool = False,
                     output_dir: str = None):
    """Run Bayesian hyperparameter optimization for ML models"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if output_dir is None:
        output_dir = Path(f"optuna_ml_results_{model_type}_{timestamp}")
    else:
        output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    
    all_results = []
    best_params_per_event = {}
    
    print("\n" + "="*80)
    print(f"BAYESIAN HYPERPARAMETER OPTIMIZATION - {model_type.upper()}")
    print("="*80)
    print(f"Optimization metric: MCC (Matthews Correlation Coefficient)")
    print(f"Number of trials: {n_trials}")
    print(f"CV folds: {n_splits}")
    print(f"Forecast horizon: {forecast_horizon}")
    print(f"Use GPU: {use_gpu}")
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
                # Quick data check
                temp_X, temp_y = prepare_temporal_features(tower_df, event_col, forecast_horizon)
                if len(temp_X) < 500:
                    print(f"      ⚠️  Not enough data ({len(temp_X)} samples), skipping...")
                    continue
                
                event_rate = temp_y.mean()
                print(f"      Samples: ~{len(temp_X)}, Event rate: {event_rate:.2%}")
                
                # Create study
                study_name = f"{tower_name}_{event_col}_{model_type}"
                study = optuna.create_study(
                    study_name=study_name,
                    direction='maximize',
                    sampler=TPESampler(seed=42),
                    pruner=MedianPruner(n_startup_trials=5, n_warmup_steps=5)
                )
                
                # Create objective
                if model_type == 'lightgbm':
                    objective = LGBMObjective(tower_df, event_col, forecast_horizon, n_splits, use_gpu)
                else:  # xgboost
                    objective = XGBObjective(tower_df, event_col, forecast_horizon, n_splits, use_gpu)
                
                # Optimize
                study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
                
                # Get best parameters
                best_params = study.best_params
                best_mcc = study.best_value
                
                print(f"      ✅ Best MCC: {best_mcc:.4f}")
                print(f"      Best params: {best_params}")
                
                # Train final model with best params and get all metrics
                rolling_windows = eval(best_params['rolling_windows'])
                X, y = prepare_temporal_features(
                    tower_df, event_col, forecast_horizon,
                    best_params['lag_config'], best_params['use_rolling'], rolling_windows
                )
                
                tscv = TimeSeriesSplit(n_splits=n_splits)
                final_metrics_list = []
                
                for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
                    X_train, X_val = X[train_idx], X[val_idx]
                    y_train, y_val = y[train_idx], y[val_idx]
                    
                    if model_type == 'lightgbm':
                        _, metrics = train_and_evaluate_lgbm(X_train, y_train, X_val, y_val, best_params, use_gpu)
                    else:
                        _, metrics = train_and_evaluate_xgb(X_train, y_train, X_val, y_val, best_params, use_gpu)
                    
                    final_metrics_list.append(metrics)
                
                # Average metrics
                avg_metrics = {}
                for key in final_metrics_list[0].keys():
                    if key != 'optimal_threshold':
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
                
                # Save study results
                study_df = study.trials_dataframe()
                study_df.to_csv(output_dir / f"trials_{study_name}.csv", index=False)
                
            except Exception as e:
                print(f"      ❌ Error: {str(e)}")
                import traceback
                traceback.print_exc()
                continue
    
    # Save all results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv(output_dir / 'optimization_results.csv', index=False)
    
    # Save best parameters
    with open(output_dir / 'best_params.json', 'w') as f:
        json.dump(best_params_per_event, f, indent=2, default=str)
    
    # Create summary
    summary = {
        'timestamp': timestamp,
        'model_type': model_type,
        'n_trials': n_trials,
        'n_splits': n_splits,
        'forecast_horizon': forecast_horizon,
        'use_gpu': use_gpu,
        'lgbm_search_space': {k: str(v) for k, v in LGBM_SEARCH_SPACE.items()} if model_type == 'lightgbm' else None,
        'xgb_search_space': {k: str(v) for k, v in XGB_SEARCH_SPACE.items()} if model_type == 'xgboost' else None,
        'feature_search_space': {k: str(v) for k, v in FEATURE_SEARCH_SPACE.items()},
        'total_experiments': len(all_results)
    }
    
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)
    
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
    parser = argparse.ArgumentParser(description='Bayesian Hyperparameter Optimization for ML Models')
    parser.add_argument('--model', type=str, default='lightgbm', choices=['lightgbm', 'xgboost', 'both'],
                        help='Model type to optimize')
    parser.add_argument('--n_trials', type=int, default=50,
                        help='Number of Optuna trials')
    parser.add_argument('--n_splits', type=int, default=3,
                        help='Number of CV splits')
    parser.add_argument('--horizon', type=str, default='6hours',
                        choices=list(FORECAST_HORIZONS.keys()),
                        help='Forecast horizon')
    parser.add_argument('--gpu', action='store_true',
                        help='Use GPU for training')
    parser.add_argument('--output', type=str, default=None,
                        help='Output directory')
    
    args = parser.parse_args()
    
    # Load data
    filtered_dfs = load_and_prepare_data()
    
    # Run optimization
    if args.model == 'both':
        for model_type in ['lightgbm', 'xgboost']:
            run_optimization(
                filtered_dfs,
                model_type=model_type,
                n_trials=args.n_trials,
                n_splits=args.n_splits,
                forecast_horizon=args.horizon,
                use_gpu=args.gpu,
                output_dir=args.output
            )
    else:
        run_optimization(
            filtered_dfs,
            model_type=args.model,
            n_trials=args.n_trials,
            n_splits=args.n_splits,
            forecast_horizon=args.horizon,
            use_gpu=args.gpu,
            output_dir=args.output
        )


if __name__ == "__main__":
    main()

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
    'event_E6_HighTemp_gt24'
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


{tower: list(df.columns) for tower, df in filtered_dfs.items()}



# ==================== MULTI-EVENT TEMPORAL FORECASTING - PER-TOWER EXPERIMENTS ====================
import pandas as pd
import numpy as np
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (classification_report, roc_auc_score, confusion_matrix,
                             precision_recall_curve, f1_score, precision_score, recall_score)
import lightgbm as lgb
import xgboost as xgb
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import warnings
import optuna
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import pickle
import json
from scipy import stats
from pathlib import Path

warnings.filterwarnings('ignore')

# ==================== CONFIGURATION ====================
# ==================== ENHANCED CONFIGURATION WITH MORE HYPERPARAMETERS ====================
class EventForecastConfig:
    """Enhanced configuration for temporal event forecasting"""
    
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
        'event_E6_HighTemp_gt24'
    ]
    
    # ==================== TEMPORAL FEATURES ====================
    LAG_CONFIGS = {
        'short': [1, 2, 4, 8],
        'medium': [1, 2, 4, 8, 16, 24],
        'long': [1, 4, 8, 16, 24, 48, 96],
    }
    SELECTED_LAG_CONFIG = 'medium'
    ROLLING_WINDOWS = [4, 12, 24, 96]
    
    # ==================== MODEL PARAMETERS ====================
    N_SPLITS = 5
    USE_GPU = False
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    USE_CLASS_WEIGHTS = True
    MODELS_TO_TRAIN = ['lightgbm', 'xgboost']
    
    # ==================== LIGHTGBM HYPERPARAMETERS ====================
    LGBM_PARAMS = {
        'objective': 'binary',
        'metric': 'auc',
        'boosting_type': 'gbdt',
        'is_unbalance': True,
        'device': 'gpu' if USE_GPU else 'cpu',
        'num_leaves': 31,
        'max_depth': -1,
        'learning_rate': 0.05,
        'n_estimators': 1000,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'min_child_samples': 20,
        'min_child_weight': 0.001,
        'min_split_gain': 0.0,
        'reg_alpha': 0.0,
        'reg_lambda': 0.0,
        'subsample': 0.8,
        'subsample_freq': 1,
        'colsample_bytree': 0.9,
        'verbose': -1,
        'random_state': 42,
        'early_stopping_rounds': 50
    }
    
    # ==================== XGBOOST HYPERPARAMETERS ====================
    XGB_PARAMS = {
        'objective': 'binary:logistic',
        'eval_metric': 'auc',
        'max_depth': 6,
        'min_child_weight': 1,
        'learning_rate': 0.05,
        'n_estimators': 1000,
        'subsample': 0.8,
        'colsample_bytree': 0.9,
        'colsample_bylevel': 1.0,
        'colsample_bynode': 1.0,
        'gamma': 0,
        'reg_alpha': 0,
        'reg_lambda': 1,
        'scale_pos_weight': 1,
        'max_delta_step': 0,
        'random_state': 42,
        'tree_method': 'gpu_hist' if USE_GPU else 'hist',
        'verbosity': 0,
        'early_stopping_rounds': 50
    }
    
    # ==================== EVALUATION METRICS ====================
    CLASSIFICATION_METRICS = [
        'accuracy',
        'balanced_accuracy',
        'precision',
        'recall',
        'f1',
        'auc_roc',
        'auc_pr',
        'specificity',
        'mcc',
        'cohen_kappa'
    ]

# ==================== ENHANCED METRICS CALCULATION ====================
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, 
    matthews_corrcoef, cohen_kappa_score,
    average_precision_score
)

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

# ==================== UPDATE MODEL TRAINING TO USE ENHANCED METRICS ====================
def train_single_tower_event_models(X: pd.DataFrame, y: pd.Series, 
                                    config: EventForecastConfig,
                                    tower_name: str,
                                    event_name: str,
                                    metadata: pd.DataFrame = None) -> Dict:
    """Train models for ONE tower and ONE event with enhanced metrics"""
    
    tscv = TimeSeriesSplit(n_splits=config.N_SPLITS)
    
    results = {
        'tower': tower_name,
        'event': event_name,
        'models': {}
    }
    
    # Initialize results for each model type
    for model_type in config.MODELS_TO_TRAIN:
        results['models'][model_type] = {
            'trained_models': [],
            'fold_metadata': []
        }
        # Initialize all metrics
        for metric in config.CLASSIFICATION_METRICS:
            results['models'][model_type][metric] = []
        # Initialize confusion matrix components
        for cm_comp in ['true_positives', 'true_negatives', 'false_positives', 'false_negatives']:
            results['models'][model_type][cm_comp] = []
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # Calculate class weights
        if config.USE_CLASS_WEIGHTS:
            n_samples = len(y_train)
            n_pos = y_train.sum()
            n_neg = n_samples - n_pos
            
            weight_pos = n_samples / (2 * n_pos) if n_pos > 0 else 1.0
            weight_neg = n_samples / (2 * n_neg) if n_neg > 0 else 1.0
            
            sample_weights = np.where(y_train == 1, weight_pos, weight_neg)
        else:
            sample_weights = None
        
        # Train each model type
        for model_type in config.MODELS_TO_TRAIN:
            
            if model_type == 'lightgbm':
                model = train_lightgbm_model(X_train, y_train, X_val, y_val, 
                                            config.LGBM_PARAMS, sample_weights)
                y_pred_proba = model.predict(X_val)
                
            elif model_type == 'xgboost':
                model = train_xgboost_model(X_train, y_train, X_val, y_val,
                                           config.XGB_PARAMS, sample_weights)
                dval = xgb.DMatrix(X_val)
                y_pred_proba = model.predict(dval)
            
            # Optimal threshold
            precisions, recalls, thresholds = precision_recall_curve(y_val, y_pred_proba)
            f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
            optimal_idx = np.argmax(f1_scores)
            optimal_threshold = thresholds[optimal_idx] if optimal_idx < len(thresholds) else 0.5
            
            y_pred = (y_pred_proba >= optimal_threshold).astype(int)
            
            # Calculate all metrics
            metrics = calculate_all_metrics(y_val, y_pred, y_pred_proba)
            
            # Store metrics
            for metric_name, metric_value in metrics.items():
                if metric_name in results['models'][model_type]:
                    results['models'][model_type][metric_name].append(metric_value)
            
            # Store model
            results['models'][model_type]['trained_models'].append(model)
            
            # Store fold metadata
            if metadata is not None:
                fold_meta = metadata.iloc[val_idx].copy()
                fold_meta['y_true'] = y_val.values
                fold_meta['y_pred_proba'] = y_pred_proba
                fold_meta['y_pred'] = y_pred
                fold_meta['fold'] = fold
                fold_meta['model_type'] = model_type
                results['models'][model_type]['fold_metadata'].append(fold_meta)
    
    return results

# ==================== ENHANCED SAVE RESULTS WITH COMPREHENSIVE SUMMARY ====================
def save_all_results(all_results: Dict, config: EventForecastConfig):
    """Save all experiment results with comprehensive summaries"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create output directory
    output_dir = Path(f"multi_event_results_{timestamp}")
    output_dir.mkdir(exist_ok=True)
    
    models_dir = output_dir / "models"
    models_dir.mkdir(exist_ok=True)
    
    # ==================== SAVE ENHANCED CONFIGURATION ====================
    config_dict = {
        'experiment': {
            'timestamp': timestamp,
            'description': 'Multi-event temporal forecasting with enhanced metrics'
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
            'device': config.DEVICE
        },
        'lightgbm_params': config.LGBM_PARAMS,
        'xgboost_params': config.XGB_PARAMS,
        'metrics': config.CLASSIFICATION_METRICS
    }
    
    with open(output_dir / 'config.json', 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    print(f"\n✅ Configuration saved to: {output_dir / 'config.json'}")
    
    # ==================== CREATE COMPREHENSIVE SUMMARY ====================
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
                
                # Add all metrics for each model type
                for model_type in config.MODELS_TO_TRAIN:
                    model_results = event_results['results']['models'][model_type]
                    
                    for metric in config.CLASSIFICATION_METRICS:
                        if metric in model_results:
                            values = model_results[metric]
                            row[f'{model_type}_{metric}_mean'] = np.mean(values)
                            row[f'{model_type}_{metric}_std'] = np.std(values)
                    
                    # Add confusion matrix components
                    for cm_comp in ['true_positives', 'true_negatives', 'false_positives', 'false_negatives']:
                        if cm_comp in model_results:
                            row[f'{model_type}_{cm_comp}_mean'] = np.mean(model_results[cm_comp])
                
                summary_data.append(row)
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(output_dir / 'summary_results.csv', index=False)
    print(f"✅ Summary results saved to: {output_dir / 'summary_results.csv'}")
    

    # Add this section RIGHT AFTER creating best_models_summary.csv in save_all_results()

    # ==================== CREATE PER-EVENT BEST MODELS SUMMARY ====================
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
        
        # For each model type
        for model_type in config.MODELS_TO_TRAIN:
            # For each metric, find best tower for this event
            for metric in config.CLASSIFICATION_METRICS:
                metric_col = f'{model_type}_{metric}_mean'
                
                if metric_col in event_subset.columns:
                    best_idx = event_subset[metric_col].idxmax()
                    best_row = event_subset.loc[best_idx]
                    
                    event_summary[f'{model_type}_best_{metric}'] = best_row[metric_col]
                    event_summary[f'{model_type}_best_{metric}_tower'] = best_row['tower']
            
            # Average performance across all towers for this event
            for metric in config.CLASSIFICATION_METRICS:
                metric_col = f'{model_type}_{metric}_mean'
                if metric_col in event_subset.columns:
                    event_summary[f'{model_type}_avg_{metric}'] = event_subset[metric_col].mean()
                    event_summary[f'{model_type}_std_{metric}'] = event_subset[metric_col].std()
        
        per_event_summary.append(event_summary)

    per_event_df = pd.DataFrame(per_event_summary)
    per_event_df.to_csv(output_dir / 'best_models_per_event.csv', index=False)
    print(f"✅ Per-event best models saved to: {output_dir / 'best_models_per_event.csv'}")

    # ==================== CREATE PER-TOWER BEST MODELS SUMMARY ====================
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
        
        # For each model type
        for model_type in config.MODELS_TO_TRAIN:
            # For each metric, find best event for this tower
            for metric in config.CLASSIFICATION_METRICS:
                metric_col = f'{model_type}_{metric}_mean'
                
                if metric_col in tower_subset.columns:
                    best_idx = tower_subset[metric_col].idxmax()
                    best_row = tower_subset.loc[best_idx]
                    
                    tower_summary[f'{model_type}_best_{metric}'] = best_row[metric_col]
                    tower_summary[f'{model_type}_best_{metric}_event'] = best_row['event']
            
            # Average performance across all events for this tower
            for metric in config.CLASSIFICATION_METRICS:
                metric_col = f'{model_type}_{metric}_mean'
                if metric_col in tower_subset.columns:
                    tower_summary[f'{model_type}_avg_{metric}'] = tower_subset[metric_col].mean()
                    tower_summary[f'{model_type}_std_{metric}'] = tower_subset[metric_col].std()
        
        per_tower_summary.append(tower_summary)

    per_tower_df = pd.DataFrame(per_tower_summary)
    per_tower_df.to_csv(output_dir / 'best_models_per_tower.csv', index=False)
    print(f"✅ Per-tower best models saved to: {output_dir / 'best_models_per_tower.csv'}")

    # ==================== CREATE PER-TOWER-EVENT BEST MODELS SUMMARY ====================
    print("\n📊 Creating per-tower-event best models summary...")

    per_tower_event_summary = []

    for tower in summary_df['tower'].unique():
        for event in config.TARGET_EVENTS:
            subset = summary_df[(summary_df['tower'] == tower) & (summary_df['event'] == event)]
            
            if len(subset) == 0:
                continue
            
            row = subset.iloc[0]  # Should only be one row per tower-event pair
            
            tower_event_summary = {
                'tower': tower,
                'event': event,
                'n_samples': row['n_samples'],
                'event_rate': row['event_rate']
            }
            
            # For each model type, add all metrics
            for model_type in config.MODELS_TO_TRAIN:
                for metric in config.CLASSIFICATION_METRICS:
                    metric_col = f'{model_type}_{metric}_mean'
                    if metric_col in row.index:
                        tower_event_summary[f'{model_type}_{metric}'] = row[metric_col]
            
            # Determine which model is better for this specific combination
            lightgbm_auc = row.get('lightgbm_auc_roc_mean', 0)
            xgboost_auc = row.get('xgboost_auc_roc_mean', 0)
            
            tower_event_summary['best_model'] = 'lightgbm' if lightgbm_auc >= xgboost_auc else 'xgboost'
            tower_event_summary['best_auc'] = max(lightgbm_auc, xgboost_auc)
            
            per_tower_event_summary.append(tower_event_summary)

    per_tower_event_df = pd.DataFrame(per_tower_event_summary)
    per_tower_event_df.to_csv(output_dir / 'best_models_per_tower_event.csv', index=False)
    print(f"✅ Per-tower-event best models saved to: {output_dir / 'best_models_per_tower_event.csv'}")


    # ==================== CREATE DETAILED TEXT REPORT ====================
    report_file = output_dir / 'experiment_report.txt'

    with open(report_file, 'w') as f:
        f.write("="*80 + "\n")
        f.write("MULTI-EVENT TEMPORAL FORECASTING - EXPERIMENT REPORT\n")
        f.write("="*80 + "\n\n")
        
        f.write(f"Timestamp: {timestamp}\n")
        f.write(f"Forecast Horizon: {config.SELECTED_HORIZON} ({config.FORECAST_HORIZONS[config.SELECTED_HORIZON]} steps)\n")
        f.write(f"Lag Configuration: {config.SELECTED_LAG_CONFIG}\n")
        f.write(f"Models: {', '.join(config.MODELS_TO_TRAIN)}\n")
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
                std_col = f'{model_type}_{metric}_std'
                
                if mean_col in summary_df.columns:
                    mean_val = summary_df[mean_col].mean()
                    std_val = summary_df[mean_col].std()
                    max_val = summary_df[mean_col].max()
                    min_val = summary_df[mean_col].min()
                    
                    f.write(f"  {metric:20s}: {mean_val:.4f} ± {std_val:.4f}  "
                        f"[{min_val:.4f}, {max_val:.4f}]\n")
            
            f.write("\n")
        
        # ==================== ALL EVENTS RANKED BY AUC-ROC ====================
        f.write("\n" + "="*80 + "\n")
        f.write("ALL EVENT CONFIGURATIONS RANKED BY AUC-ROC\n")
        f.write("="*80 + "\n\n")
        
        for model_type in config.MODELS_TO_TRAIN:
            f.write(f"\n{model_type.upper()}\n")
            f.write("-"*80 + "\n")
            
            metric_col = f'{model_type}_auc_roc_mean'
            if metric_col in summary_df.columns:
                # Sort by AUC-ROC and include ALL configurations
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
        
        # ==================== PER-EVENT SUMMARY (ALL EVENTS) ====================
        f.write("\n" + "="*80 + "\n")
        f.write("PER-EVENT PERFORMANCE SUMMARY (ALL EVENTS)\n")
        f.write("="*80 + "\n\n")
        
        # Group by event and show statistics
        for event in config.TARGET_EVENTS:
            event_subset = summary_df[summary_df['event'] == event]
            
            if len(event_subset) == 0:
                f.write(f"\nEvent: {event.replace('event_', '')}\n")
                f.write("  No data available\n")
                continue
            
            f.write(f"\nEvent: {event.replace('event_', '')}\n")
            f.write("-"*70 + "\n")
            f.write(f"  Towers tested: {event_subset['tower'].nunique()}\n")
            f.write(f"  Total samples: {event_subset['n_samples'].sum():,}\n")
            f.write(f"  Avg event rate: {event_subset['event_rate'].mean()*100:.2f}%\n\n")
            
            for model_type in config.MODELS_TO_TRAIN:
                f.write(f"  {model_type.upper()}:\n")
                
                # Show performance across all metrics
                for metric in ['auc_roc', 'f1', 'precision', 'recall']:
                    metric_col = f'{model_type}_{metric}_mean'
                    if metric_col in event_subset.columns:
                        mean_val = event_subset[metric_col].mean()
                        std_val = event_subset[metric_col].std()
                        best_val = event_subset[metric_col].max()
                        best_tower = event_subset.loc[event_subset[metric_col].idxmax(), 'tower']
                        
                        f.write(f"    {metric.upper():12s}: {mean_val:.4f} ± {std_val:.4f}  "
                            f"(best: {best_val:.4f} @ {best_tower})\n")
                f.write("\n")
        
        # ==================== TOWER-BY-TOWER SUMMARY ====================
        f.write("\n" + "="*80 + "\n")
        f.write("TOWER-BY-TOWER SUMMARY\n")
        f.write("="*80 + "\n\n")
        
        for tower in sorted(summary_df['tower'].unique()):
            tower_subset = summary_df[summary_df['tower'] == tower]
            
            f.write(f"\nTower: {tower}\n")
            f.write("-"*70 + "\n")
            f.write(f"  Events tested: {tower_subset['event'].nunique()}\n")
            f.write(f"  Total samples: {tower_subset['n_samples'].sum():,}\n\n")
            
            # Show best performing event for each model
            for model_type in config.MODELS_TO_TRAIN:
                metric_col = f'{model_type}_auc_roc_mean'
                if metric_col in tower_subset.columns:
                    best_idx = tower_subset[metric_col].idxmax()
                    best_row = tower_subset.loc[best_idx]
                    
                    f.write(f"  {model_type.upper()} Best Event:\n")
                    f.write(f"    Event: {best_row['event'].replace('event_', '')}\n")
                    f.write(f"    AUC: {best_row[metric_col]:.4f}\n")
                    f.write(f"    F1: {best_row[f'{model_type}_f1_mean']:.4f}\n")
                    f.write(f"    Precision: {best_row[f'{model_type}_precision_mean']:.4f}\n")
                    f.write(f"    Recall: {best_row[f'{model_type}_recall_mean']:.4f}\n\n")

    print(f"✅ Experiment report saved to: {output_dir / 'experiment_report.txt'}")


    # ==================== SAVE MODELS AND FEATURE IMPORTANCE ====================
    for tower, tower_results in all_results.items():
        for event, event_results in tower_results.items():
            if 'error' in event_results:
                continue
            
            # Save models
            for model_type in config.MODELS_TO_TRAIN:
                model_data = event_results['results']['models'][model_type]
                
                for fold_idx, model in enumerate(model_data['trained_models'], 1):
                    model_filename = f"{tower}_{event}_{model_type}_fold{fold_idx}.pkl"
                    model_path = models_dir / model_filename
                    
                    with open(model_path, 'wb') as f:
                        pickle.dump(model, f)
            
            # Save feature importance
            if event_results['importance'] is not None:
                importance_file = output_dir / f"importance_{tower}_{event}.csv"
                event_results['importance'].to_csv(importance_file, index=False)
    
    print(f"✅ Models saved to: {models_dir}/ ({len(list(models_dir.glob('*.pkl')))} files)")
    
    print(f"\n{'='*80}")
    print(f"✅ ALL RESULTS SAVED TO: {output_dir}")
    print(f"{'='*80}\n")
    
    return output_dir


# ==================== MISSING FUNCTION IMPLEMENTATIONS ====================

def train_lightgbm_model(X_train, y_train, X_val, y_val, params, sample_weights=None):
    """Train LightGBM model with early stopping"""
    train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weights)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # Remove early_stopping_rounds from params for lgb.train
    params_copy = params.copy()
    early_stopping = params_copy.pop('early_stopping_rounds', 50)
    
    model = lgb.train(
        params_copy,
        train_data,
        valid_sets=[train_data, val_data],
        valid_names=['train', 'val'],
        callbacks=[lgb.early_stopping(stopping_rounds=early_stopping, verbose=False)]
    )
    
    return model


def train_xgboost_model(X_train, y_train, X_val, y_val, params, sample_weights=None):
    """Train XGBoost model with early stopping"""
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weights)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # Remove early_stopping_rounds from params for xgb.train
    params_copy = params.copy()
    early_stopping = params_copy.pop('early_stopping_rounds', 50)
    
    evals = [(dtrain, 'train'), (dval, 'val')]
    
    model = xgb.train(
        params_copy,
        dtrain,
        num_boost_round=params.get('n_estimators', 1000),
        evals=evals,
        early_stopping_rounds=early_stopping,
        verbose_eval=False
    )
    
    return model


def prepare_temporal_features(df: pd.DataFrame, config: EventForecastConfig, 
                              target_col: str) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    """Create temporal features (lags, rolling stats) for forecasting"""
    
    df = df.copy()
    df = df.sort_index()
    
    # Get numeric columns (exclude event columns, metadata, and non-numeric)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    # IMPORTANT: Exclude ALL event columns and metadata columns
    exclude_patterns = ['event_', 'tower', 'timestamp', 'date', 'year_month', 
                       'week_of_month', 'hour_of_day', 'day_name', 'day_numeric',
                       'date_numeric']
    
    numeric_cols = [c for c in numeric_cols 
                   if not any(pattern in str(c) for pattern in exclude_patterns)]
    
    # Also ensure the target column itself is excluded from features
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    # Additional safety: convert any remaining boolean columns to int
    for col in numeric_cols:
        if df[col].dtype == 'bool':
            df[col] = df[col].astype(int)
    
    feature_dfs = []
    feature_names = []
    
    # Original features
    feature_dfs.append(df[numeric_cols])
    feature_names.extend(numeric_cols)
    
    # Lag features
    lags = config.LAG_CONFIGS[config.SELECTED_LAG_CONFIG]
    for col in numeric_cols:
        for lag in lags:
            lag_col = f'{col}_lag{lag}'
            feature_dfs.append(df[col].shift(lag).to_frame(lag_col))
            feature_names.append(lag_col)
    
    # Rolling window features
    for col in numeric_cols:
        for window in config.ROLLING_WINDOWS:
            # Mean
            roll_mean_col = f'{col}_roll{window}_mean'
            feature_dfs.append(df[col].rolling(window=window).mean().to_frame(roll_mean_col))
            feature_names.append(roll_mean_col)
            
            # Std
            roll_std_col = f'{col}_roll{window}_std'
            feature_dfs.append(df[col].rolling(window=window).std().to_frame(roll_std_col))
            feature_names.append(roll_std_col)
    
    # Combine all features
    X = pd.concat(feature_dfs, axis=1)
    
    # Target: shift forward by forecast horizon
    horizon_steps = config.FORECAST_HORIZONS[config.SELECTED_HORIZON]
    
    # Ensure target is boolean/int
    if df[target_col].dtype == 'object':
        df[target_col] = df[target_col].astype(bool).astype(int)
    elif df[target_col].dtype == 'bool':
        df[target_col] = df[target_col].astype(int)
    
    y = df[target_col].shift(-horizon_steps)
    
    # Drop rows with NaN (from lags/rolling/future target)
    valid_idx = X.notna().all(axis=1) & y.notna()
    X = X[valid_idx]
    y = y[valid_idx]
    
    # Final safety check: ensure all X columns are numeric
    for col in X.columns:
        if X[col].dtype == 'object':
            X[col] = pd.to_numeric(X[col], errors='coerce')
    
    # Drop any columns that are still non-numeric
    X = X.select_dtypes(include=[np.number])
    feature_names = list(X.columns)
    
    return X, y, feature_names


def run_multi_event_experiments(filtered_dfs: Dict[str, pd.DataFrame], 
                                config: EventForecastConfig) -> Dict:
    """Run experiments for all towers and all events"""
    
    all_results = {}
    
    print("\n" + "="*80)
    print("RUNNING MULTI-EVENT EXPERIMENTS")
    print("="*80)
    
    # Iterate through towers
    for tower_name, tower_df in filtered_dfs.items():
        print(f"\n{'='*70}")
        print(f"🏗️  TOWER: {tower_name}")
        print(f"{'='*70}")
        
        all_results[tower_name] = {}
        
        # Iterate through target events
        for event_col in config.TARGET_EVENTS:
            
            if event_col not in tower_df.columns:
                print(f"   ⚠️  {event_col} not found in {tower_name}, skipping...")
                all_results[tower_name][event_col] = {'error': 'Event column not found'}
                continue
            
            print(f"\n   📍 Event: {event_col}")
            
            try:
                # Prepare data
                X, y, feature_cols = prepare_temporal_features(tower_df, config, event_col)
                
                # Check if sufficient data
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
                
                # Create metadata (timestamp info for analysis)
                metadata = tower_df.loc[X.index, ['timestamp']].copy() if 'timestamp' in tower_df.columns else None
                
                # Train models
                results = train_single_tower_event_models(
                    X, y, config, tower_name, event_col, metadata
                )
                
                # Calculate feature importance (from first fold's first model)
                importance_df = analyze_feature_importance(
                    results['models']['lightgbm']['trained_models'][0],
                    feature_cols,
                    top_n=20
                )
                
                # Store results
                all_results[tower_name][event_col] = {
                    'results': results,
                    'importance': importance_df,
                    'n_samples': len(y),
                    'event_rate': float(y.mean())
                }
                
                # Print summary
                for model_type in config.MODELS_TO_TRAIN:
                    model_results = results['models'][model_type]
                    mean_auc = np.mean(model_results['auc_roc'])
                    mean_f1 = np.mean(model_results['f1'])
                    print(f"      ✓ {model_type:10s}: AUC={mean_auc:.4f}, F1={mean_f1:.4f}")
                
            except Exception as e:
                print(f"      ❌ Error: {str(e)}")
                all_results[tower_name][event_col] = {'error': str(e)}
    
    print("\n" + "="*80)
    print("✅ ALL TOWER-EVENT COMBINATIONS PROCESSED")
    print("="*80)
    
    return all_results


def analyze_feature_importance(model, feature_names: List[str], top_n: int = 20) -> pd.DataFrame:
    """Extract and rank feature importance from trained model"""
    
    if isinstance(model, lgb.Booster):
        importance = model.feature_importance(importance_type='gain')
    elif isinstance(model, xgb.Booster):
        importance_dict = model.get_score(importance_type='gain')
        # XGBoost uses f0, f1, ... need to map to actual names
        importance = np.zeros(len(feature_names))
        for i, fname in enumerate(feature_names):
            importance[i] = importance_dict.get(f'f{i}', 0)
    else:
        return None
    
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    })
    
    importance_df = importance_df.sort_values('importance', ascending=False)
    importance_df['rank'] = range(1, len(importance_df) + 1)
    importance_df['importance_pct'] = 100 * importance_df['importance'] / importance_df['importance'].sum()
    
    return importance_df.head(top_n)


# ==================== INSTANTIATE CONFIG ====================
config = EventForecastConfig()


# ==================== NOW RUN EXECUTION ====================
print("="*70)
print("MULTI-EVENT TEMPORAL FORECASTING")
print("="*70)
print(f"Device: {config.DEVICE}")
print(f"Forecast horizon: {config.SELECTED_HORIZON}")
print(f"Target events: {', '.join(config.TARGET_EVENTS)}")
print(f"Models: {', '.join(config.MODELS_TO_TRAIN)}")
print(f"Lag configuration: {config.SELECTED_LAG_CONFIG}")
print("="*70)

# Run experiments
all_results = run_multi_event_experiments(filtered_dfs, config)

# Save results
output_dir = save_all_results(all_results, config)

print("\n" + "="*80)
print("✅ MULTI-EVENT EXPERIMENTS COMPLETE!")
print("="*80)
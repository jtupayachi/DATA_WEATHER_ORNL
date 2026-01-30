"""
Multi-Tower Machine Learning Approach for Weather Event Prediction
==================================================================
This script trains a SINGLE model for ALL towers simultaneously, using tower_id
as a categorical feature to enable transfer learning across towers.

Key differences from single-tower approach:
1. Pools data from all towers into one dataset
2. Adds tower_id as a feature (one-hot encoded)
3. Trains ONE model per event (not per tower-event combination)
4. Shares learned patterns across towers
5. Benefits sparse-feature towers (TOWF, TOWS) with data-rich towers (TOWD, TOWY)
"""

import pandas as pd
import numpy as np
from typing import Tuple, Optional, Dict, List
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import (classification_report, roc_auc_score, confusion_matrix,
                             precision_recall_curve, f1_score, precision_score, 
                             recall_score, accuracy_score, balanced_accuracy_score, 
                             matthews_corrcoef, cohen_kappa_score, average_precision_score)
import lightgbm as lgb
import xgboost as xgb
import shap
import warnings
from datetime import datetime
import pickle
import json
from pathlib import Path
import os
import glob

warnings.filterwarnings('ignore')

# ==================== LOAD AND PREPARE DATA ====================
print("="*80)
print("MULTI-TOWER ML APPROACH: ONE MODEL FOR ALL TOWERS")
print("="*80)

# Read the CSV file
df = pd.read_csv("../../fully_labeled_weather_data_with_events.csv")

# FILTER TO LAST 2 YEARS (2021-2023) for faster training
print(f"\n{'='*80}", flush=True)
print("FILTERING DATA TO LAST 2 YEARS (2021-2023)", flush=True)
print(f"{'='*80}", flush=True)
original_len = len(df)
df['timestamp'] = pd.to_datetime(df['timestamp'])
df = df[(df['timestamp'] >= '2021-01-01') & (df['timestamp'] < '2023-01-01')].copy()
print(f"Original data: {original_len:,} rows", flush=True)
print(f"Filtered data: {len(df):,} rows ({len(df)/original_len*100:.1f}%)", flush=True)
print(f"Date range: {df['timestamp'].min()} to {df['timestamp'].max()}", flush=True)
print(f"{'='*80}\n", flush=True)

# Move 'tower' and 'timestamp' to the first two columns
cols = df.columns.tolist()
priority_cols = ['tower', 'timestamp']
remaining_cols = [c for c in cols if c not in priority_cols]
df = df[priority_cols + remaining_cols]

# Drop unnecessary columns
df = df.loc[:, ~df.columns.str.endswith(('_min', '_meets_duration'))]
df = df.drop(columns=['event_count','active_events','event_durations', 'has_any_event'], errors='ignore')

# Ensure timestamp is datetime
df['timestamp'] = pd.to_datetime(df['timestamp'])

# List all event columns
event_cols = [
    'event_E3_LowTemp_lt0',
    'event_E4_HighWind_Peak_gt25',
    'event_E5_LowWind_lt2',
]

# Ensure the columns are boolean
df[event_cols] = df[event_cols].astype(bool)

# ==================== TOWER SUBSET FILTERING (for ablation studies) ====================
# Set TOWER_SUBSET to None to use all towers, or specify list like ['TOWA', 'TOWB']
TOWER_SUBSET = None  # Default: use all towers
# TOWER_SUBSET = ['TOWF', 'TOWS']  # Example: high elevation only
# TOWER_SUBSET = ['TOWA', 'TOWB', 'TOWD']  # Example: low elevation only

if TOWER_SUBSET is not None:
    original_count = len(df)
    df = df[df['tower'].isin(TOWER_SUBSET)].copy()
    print(f"\n⚠️  TOWER SUBSET FILTER ACTIVE", flush=True)
    print(f"  Keeping only: {', '.join(TOWER_SUBSET)}", flush=True)
    print(f"  Rows: {original_count:,} → {len(df):,}", flush=True)

print(f"\n✓ Loaded data (long format): {len(df):,} rows, {len(df['tower'].unique())} towers", flush=True)
print(f"  Towers: {', '.join(sorted(df['tower'].unique()))}", flush=True)
print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}", flush=True)

# ==================== PIVOT TO WIDE FORMAT ====================
print("\n" + "="*80, flush=True)
print("PIVOTING DATA: ONE ROW PER TIMESTAMP", flush=True)
print("="*80, flush=True)
print("Converting from long format (6 rows/timestamp) to wide format (1 row/timestamp)...", flush=True)

# Separate event columns from feature columns
feature_cols = [c for c in df.columns if c not in ['tower', 'timestamp'] + event_cols]

# Pivot feature columns: create {tower}_{feature} columns
print(f"  Pivoting {len(feature_cols)} feature columns for 6 towers...", flush=True)
df_pivot = df.pivot_table(
    index='timestamp',
    columns='tower',
    values=feature_cols,
    aggfunc='first'  # Take first value (should be only one per tower-timestamp)
)

# Flatten multi-level columns: (feature, tower) -> tower_feature
df_pivot.columns = [f"{tower}_{col}" for col, tower in df_pivot.columns]
df_pivot = df_pivot.reset_index()

# ==================== APPLY EVENT LABELING LOGIC ====================
print(f"  Applying event detection logic to pivoted data...", flush=True)

colmap_per_tower = {
    "TOWA": {"TempC": "TempC_030m", "PkWSpdMph": "PkWSpdMph_030m"},
    "TOWB": {"TempC": "TempC_030m", "PkWSpdMph": "PkWSpdMph_030m"},
    "TOWD": {"TempC": "TempC_035m", "PkWSpdMph": "PkWSpdMph_035m"},
    "TOWF": {"TempC": "TempC_010m", "PkWSpdMph": "PkWSpdMph_010m"},
    "TOWS": {"TempC": "TempC_025m", "PkWSpdMph": "PkWSpdMph_025m"},
    "TOWY": {"TempC": "TempC_033m", "PkWSpdMph": "PkWSpdMph_033m"},
}

# 🎯 FIXED EVENT THRESHOLDS - Using absolute values, not percentiles
print(f"\n  Applying fixed event thresholds...", flush=True)

EVENT_SPECS = {}

# E3: Low Temperature - FIXED THRESHOLD: <3°C, Duration: 2.0 hours
# Name stays E3_LowTemp_lt0 for consistency
print(f"    E3_LowTemp_lt0: Threshold = 3.0°C (FIXED), Duration = 2.0 hours", flush=True)
EVENT_SPECS["E3_LowTemp_lt0"] = (["TempC"], lambda df, col: df[col["TempC"]] < 3.0, 2.0)

# E4: High Wind - FIXED THRESHOLD: >19 mph, Duration: 0.5 hours
# Name stays E4_HighWind_Peak_gt25 for consistency
print(f"    E4_HighWind_Peak_gt25: Threshold = 19.0 mph (FIXED), Duration = 0.5 hours", flush=True)
EVENT_SPECS["E4_HighWind_Peak_gt25"] = (["PkWSpdMph"], lambda df, col: df[col["PkWSpdMph"]] > 19.0, 0.5)

# E5: Low Wind - FIXED THRESHOLD: <4 mph, Duration: 1.0 hours
# Name stays E5_LowWind_lt2 for consistency
print(f"    E5_LowWind_lt2: Threshold = 4.0 mph (FIXED), Duration = 1.0 hours", flush=True)
EVENT_SPECS["E5_LowWind_lt2"] = (["PkWSpdMph"], lambda df, col: df[col["PkWSpdMph"]] < 4.0, 1.0)

print(f"\n  Applying thresholds with duration requirements...", flush=True)

# For each event, check conditions across all towers with DURATION requirements
for event_name, (needed_keys, cond_fn, min_hours) in EVENT_SPECS.items():
    event_col = f"event_{event_name}"
    print(f"\n  Processing {event_col} (duration: {min_hours} hours)...", flush=True)
    
    # Calculate required consecutive timesteps (15-min intervals)
    min_steps = int(min_hours * 4)  # 4 intervals per hour (15 min each)
    print(f"    Requires {min_steps} consecutive 15-min intervals", flush=True)
    
    # Create boolean mask for each tower
    tower_masks = []
    for tower in ['TOWA', 'TOWB', 'TOWD', 'TOWF', 'TOWS', 'TOWY']:
        cmap = colmap_per_tower[tower]
        
        # Build pivoted column names: TOWER_column
        pivoted_cols = {}
        has_all_cols = True
        for k in needed_keys:
            col_name = cmap.get(k)
            if col_name is None:
                has_all_cols = False
                break
            pivoted_col = f"{tower}_{col_name}"
            if pivoted_col not in df_pivot.columns:
                has_all_cols = False
                break
            pivoted_cols[k] = pivoted_col
        
        if not has_all_cols:
            # Tower doesn't have required columns, skip
            continue
        
        # Apply condition function (instant threshold check)
        instant_mask = cond_fn(df_pivot, pivoted_cols)
        
        # Set False where any required column is NaN
        for col in pivoted_cols.values():
            instant_mask = instant_mask & df_pivot[col].notna()
        
        # Apply DURATION requirement: rolling window check
        # True only if condition is met for min_steps consecutive intervals
        duration_mask = instant_mask.rolling(window=min_steps, min_periods=min_steps).sum() >= min_steps
        
        tower_masks.append(duration_mask)
        print(f"    {tower}: {instant_mask.sum():,} instant → {duration_mask.sum():,} with duration", flush=True)
    
    # UNION: True if ANY tower has the event (with duration requirement)
    if tower_masks:
        combined_mask = tower_masks[0]
        for mask in tower_masks[1:]:
            combined_mask = combined_mask | mask
        df_pivot[event_col] = combined_mask.fillna(False).astype(bool)
    else:
        df_pivot[event_col] = False
    
    n_events = df_pivot[event_col].sum()
    rate = (df_pivot[event_col].mean() * 100)
    print(f"    ✓ UNION result: {n_events:,} events ({rate:.2f}%)", flush=True)

print(f"\n✓ Pivoted data: {len(df_pivot):,} rows (timestamps), {len(df_pivot.columns)-1:,} columns", flush=True)
print(f"  Unique timestamps: {df_pivot['timestamp'].nunique()}", flush=True)
print(f"  Columns per tower: ~{len(feature_cols)}", flush=True)

# ==================== EVENT-SPECIFIC RESAMPLING ====================
# Resample to match event duration for better temporal alignment
print("\n" + "="*80, flush=True)
print("EVENT-SPECIFIC RESAMPLING", flush=True)
print("="*80, flush=True)

# Define resampling rules based on event duration
EVENT_RESAMPLE_RULES = {
    "event_E3_LowTemp_lt0": "2H",        # 2-hour duration → 2-hour intervals
    "event_E4_HighWind_Peak_gt25": "30T",  # 0.5-hour duration → 30-min intervals  
    "event_E5_LowWind_lt2": "30T",       # 0.5-hour duration → 30-min intervals
}

print("Resampling strategy:", flush=True)
for event, rule in EVENT_RESAMPLE_RULES.items():
    print(f"  {event}: {rule}", flush=True)

# Store original 15-min data for reference
df_pivot_15min = df_pivot.copy()

# Show sample of pivoted data
print("\n" + "-"*80, flush=True)
print("PIVOTED DATA STRUCTURE (first 5 rows, first 20 columns):", flush=True)
print("-"*80, flush=True)
print(df_pivot.iloc[:5, :20].to_string(), flush=True)

print("\n" + "-"*80, flush=True)
print("EVENT DISTRIBUTION (across all timestamps - 15min resolution):", flush=True)
print("-"*80, flush=True)
for event_col in event_cols:
    n_events = df_pivot[event_col].sum()
    rate = (df_pivot[event_col].mean() * 100)
    print(f"{event_col}: {n_events} events ({rate:.2f}%)", flush=True)

# ==================== RESAMPLE FUNCTION ====================
def resample_for_event(df: pd.DataFrame, event_col: str, resample_rule: str, event_cols: List[str]) -> pd.DataFrame:
    """
    Resample dataframe to match event duration.
    
    Args:
        df: Dataframe with timestamp index
        event_col: Target event column
        resample_rule: Pandas resample rule (e.g., '30T', '2H')
        event_cols: List of all event columns
    
    Returns:
        Resampled dataframe
    """
    print(f"\n  Resampling for {event_col} to {resample_rule}...", flush=True)
    print(f"    Before: {len(df):,} rows", flush=True)
    
    # Set timestamp as index
    df_temp = df.set_index('timestamp')
    
    # Separate features and events
    feature_cols = [c for c in df_temp.columns if c not in event_cols]
    
    # Resample features (mean)
    df_features = df_temp[feature_cols].resample(resample_rule).mean()
    
    # Resample events (max - if ANY timestep has event, resampled period has event)
    df_events = df_temp[event_cols].resample(resample_rule).max()
    
    # Fill NaN values in feature columns (ffill → bfill → 0)
    df_features = df_features.ffill().bfill().fillna(0)
    
    # Fill NaN values in event columns with False and ensure boolean type
    df_events = df_events.fillna(False).astype(bool)
    
    # Combine
    df_resampled = pd.concat([df_features, df_events], axis=1).reset_index()
    
    print(f"    After: {len(df_resampled):,} rows", flush=True)
    print(f"    Event rate: {df_resampled[event_col].mean()*100:.2f}%", flush=True)
    
    # 🌦️ LOAD AND MERGE NSRD DATA
    print(f"\n  Loading NSRD data for {event_col}...", flush=True)
    df_nsrd = load_nsrd_data_for_event(event_col)
    
    if df_nsrd is not None:
        df_resampled = merge_nsrd_with_tower_data(df_resampled, df_nsrd)
    else:
        print(f"  ⚠️  Continuing without NSRD data", flush=True)
    
    return df_resampled

# Replace df with pivoted version (will be resampled per-event in training loop)
df = df_pivot
print("\n" + "="*80, flush=True)

# ==================== LOAD AND MERGE NSRD DATA ====================
def load_nsrd_data_for_event(event_name: str, base_path: str = "../../data") -> Optional[pd.DataFrame]:
    """
    Load NSRD data resampled for specific event.
    
    Parameters:
    -----------
    event_name : str
        Event name (e.g., 'event_E3_LowTemp_lt0')
    base_path : str
        Path to data directory
        
    Returns:
    --------
    pd.DataFrame or None
        NSRD data with timestamp and NSRD_* columns, or None if file not found
    """
    # Extract event identifier (remove 'event_' prefix)
    event_id = event_name.replace('event_', '')
    
    # Construct filename
    nsrd_file = Path(base_path) / f"NSRD_BackgroundMet2022_{event_id}_resampled.csv"
    
    if not nsrd_file.exists():
        print(f"  ⚠️  NSRD file not found: {nsrd_file}", flush=True)
        return None
    
    # Load NSRD data
    df_nsrd = pd.read_csv(nsrd_file)
    df_nsrd['timestamp'] = pd.to_datetime(df_nsrd['timestamp'])
    
    print(f"  ✓ Loaded NSRD data: {len(df_nsrd):,} rows, {len(df_nsrd.columns)-1} features", flush=True)
    print(f"    Date range: {df_nsrd['timestamp'].min()} to {df_nsrd['timestamp'].max()}", flush=True)
    
    return df_nsrd


def merge_nsrd_with_tower_data(df_tower: pd.DataFrame, df_nsrd: pd.DataFrame) -> pd.DataFrame:
    """
    Merge NSRD data with tower data on timestamp.
    
    Parameters:
    -----------
    df_tower : pd.DataFrame
        Tower data with timestamp column
    df_nsrd : pd.DataFrame
        NSRD data with timestamp and NSRD_* columns
        
    Returns:
    --------
    pd.DataFrame
        Merged dataframe with both tower and NSRD features
    """
    print(f"  Merging NSRD data with tower data...", flush=True)
    print(f"    Tower data: {len(df_tower):,} rows", flush=True)
    print(f"    NSRD data: {len(df_nsrd):,} rows", flush=True)
    
    # Merge on timestamp (left join to keep all tower timestamps)
    df_merged = df_tower.merge(df_nsrd, on='timestamp', how='left')
    
    # Fill NaN values in NSRD columns (forward fill → backward fill → 0)
    nsrd_cols = [c for c in df_nsrd.columns if c.startswith('NSRD_')]
    for col in nsrd_cols:
        if col in df_merged.columns:
            df_merged[col] = df_merged[col].ffill().bfill().fillna(0)
    
    print(f"    Merged data: {len(df_merged):,} rows, {len(df_merged.columns)} columns", flush=True)
    print(f"    Added {len(nsrd_cols)} NSRD features", flush=True)
    
    return df_merged


# ==================== FILTER COLUMNS BY NULL PERCENTAGE ====================
def filter_columns_by_nulls(df, threshold=80):
    """
    Keep columns with null percentage below threshold.
    For pivoted data: columns are already tower-specific (e.g., TOWA_TempC_015m)
    """
    print("\n" + "="*80, flush=True)
    print("FILTERING COLUMNS BY NULL PERCENTAGE", flush=True)
    print("="*80, flush=True)
    
    # Calculate null percentage for all columns
    feature_cols = [c for c in df.columns if c not in ['timestamp'] + event_cols]
    null_pct = df[feature_cols].isnull().mean() * 100
    
    # Categorize columns
    valid_cols = null_pct[null_pct <= threshold].index.tolist()
    dropped_cols = null_pct[null_pct > threshold].index.tolist()
    
    print(f"Threshold: <={threshold}% nulls", flush=True)
    print(f"Total feature columns: {len(feature_cols)}", flush=True)
    print(f"Valid columns: {len(valid_cols)}", flush=True)
    print(f"Dropped columns: {len(dropped_cols)}", flush=True)
    
    # Show distribution
    n_cols_0 = (null_pct == 0).sum()
    n_cols_0_20 = ((null_pct > 0) & (null_pct <= 20)).sum()
    n_cols_20_50 = ((null_pct > 20) & (null_pct <= 50)).sum()
    n_cols_50_80 = ((null_pct > 50) & (null_pct <= 80)).sum()
    n_cols_80_100 = (null_pct > 80).sum()
    
    print(f"\nNull distribution:", flush=True)
    print(f"  0% nulls:      {n_cols_0:4d} columns", flush=True)
    print(f"  0-20% nulls:   {n_cols_0_20:4d} columns", flush=True)
    print(f"  20-50% nulls:  {n_cols_20_50:4d} columns", flush=True)
    print(f"  50-80% nulls:  {n_cols_50_80:4d} columns", flush=True)
    print(f"  >80% nulls:    {n_cols_80_100:4d} columns (dropped)", flush=True)
    
    # Fill missing values (forward fill → backward fill → 0)
    print(f"\nFilling missing values in valid columns (ffill → bfill → 0)...", flush=True)
    df_filtered = df[['timestamp'] + valid_cols + event_cols].copy()
    
    for col in valid_cols:
        if df_filtered[col].dtype in [np.float64, np.int64]:
            df_filtered[col] = df_filtered[col].ffill().bfill().fillna(0)
    
    print(f"\n✓ Final: {len(df_filtered.columns)} columns (1 timestamp + {len(valid_cols)} features + {len(event_cols)} events)", flush=True)
    print("="*80, flush=True)
    
    return df_filtered
    
    # Any remaining NaNs, fill with 0 (rare case)
    numeric_cols = df_filtered.select_dtypes(include=[np.number]).columns
    df_filtered[numeric_cols] = df_filtered[numeric_cols].fillna(0)
    
    print(f"\n✓ Final: {len(valid_cols)} columns kept (threshold: <={threshold}% nulls in ANY tower)", flush=True)
    print("="*80, flush=True)
    
    return df_filtered

df_filtered = filter_columns_by_nulls(df, threshold=100)

# ==================== CONFIGURATION ====================
class MultiTowerConfig:
    """Configuration for multi-tower experiments"""
    
    # Target events - PRIORITIZED BY DIFFICULTY (most imbalanced first)
    TARGET_EVENTS = [
        'event_E5_LowWind_lt2',          # 1.52% - HARDEST (severe imbalance)
        'event_E4_HighWind_Peak_gt25',   # 1.67% - HARD
        'event_E3_LowTemp_lt0',          # 5.86% - EASIER
    ]
    
    # Models to train
    MODELS_TO_TRAIN = ['lightgbm', 'xgboost']
    
    # Time series cross-validation (3-fold for better sample size per fold)
    N_SPLITS = 3
    
    # Feature engineering
    LAG_CONFIGS = {
        'short': list(range(1, 13)),      # 1-12 timesteps (6 hours)
        'medium': list(range(1, 25)),     # 1-24 timesteps (12 hours)
        'long': list(range(1, 49)),       # 1-48 timesteps (24 hours)
    }
    SELECTED_LAG_CONFIG = 'long'
    
    # 🎯 EVENT-SPECIFIC PARAMETERS
    EVENT_RESAMPLE_RULES = {
        "event_E3_LowTemp_lt0": "2H",           # 2-hour duration → 2-hour intervals
        "event_E4_HighWind_Peak_gt25": "30T",   # 0.5-hour duration → 30-min intervals  
        "event_E5_LowWind_lt2": "30T",          # 1.0-hour duration → 30-min intervals
    }
    
    # Rolling windows per event (adapted to resampled frequency)
    ROLLING_WINDOWS_PER_EVENT = {
        "event_E3_LowTemp_lt0": [2, 6, 12, 24],       # 4h, 12h, 24h, 48h (2h intervals)
        "event_E4_HighWind_Peak_gt25": [2, 6, 12, 24],  # 1h, 3h, 6h, 12h (30min intervals)
        "event_E5_LowWind_lt2": [2, 6, 12, 24],       # 1h, 3h, 6h, 12h (30min intervals)
    }
    
    # Lag configs per event (adapted to resampled frequency)
    LAG_CONFIGS_PER_EVENT = {
        "event_E3_LowTemp_lt0": list(range(1, 25)),      # 1-24 steps (2-48 hours)
        "event_E4_HighWind_Peak_gt25": list(range(1, 25)),  # 1-24 steps (0.5-12 hours)
        "event_E5_LowWind_lt2": list(range(1, 25)),      # 1-24 steps (0.5-12 hours)
    }
    
    # Forecast horizons (EVENT-SPECIFIC INTERVALS)
    # E3 uses 2-hour intervals, E4/E5 use 30-min intervals
    FORECAST_HORIZONS_PER_EVENT = {
        "event_E3_LowTemp_lt0": {
            '2hr': 1,    # 1 × 2hr = 2 hours
            '4hr': 2,    # 2 × 2hr = 4 hours
            '6hr': 3,    # 3 × 2hr = 6 hours
            '12hr': 6,   # 6 × 2hr = 12 hours
            '24hr': 12,  # 12 × 2hr = 24 hours
            '48hr': 24,  # 24 × 2hr = 48 hours
        },
        "event_E4_HighWind_Peak_gt25": {
            '1hr': 2,    # 2 × 30min = 1 hour
            '3hr': 6,    # 6 × 30min = 3 hours
            '6hr': 12,   # 12 × 30min = 6 hours
            '12hr': 24,  # 24 × 30min = 12 hours
            '24hr': 48,  # 48 × 30min = 24 hours
        },
        "event_E5_LowWind_lt2": {
            '1hr': 2,    # 2 × 30min = 1 hour
            '3hr': 6,    # 6 × 30min = 3 hours
            '6hr': 12,   # 12 × 30min = 6 hours
            '12hr': 24,  # 24 × 30min = 12 hours
            '24hr': 48,  # 48 × 30min = 24 hours
        },
    }
    
    # Selected horizons per event
    SELECTED_HORIZONS_PER_EVENT = {
        "event_E3_LowTemp_lt0": ['2hr', '4hr', '6hr', '12hr', '24hr', '48hr'],
        "event_E4_HighWind_Peak_gt25": ['1hr', '3hr', '6hr', '12hr', '24hr'],
        "event_E5_LowWind_lt2": ['1hr', '3hr', '6hr', '12hr', '24hr'],
    }
    
    # Model parameters (will be updated with Optuna params if available)
    USE_CLASS_WEIGHTS = True
    
    LGBM_PARAMS = {
        'objective': 'binary',
        'metric': 'binary_logloss',
        'boosting_type': 'gbdt',
        'num_leaves': 31,
        'learning_rate': 0.05,
        'feature_fraction': 0.9,
        'bagging_fraction': 0.8,
        'bagging_freq': 5,
        'verbose': -1,
        'n_estimators': 1000,
        'min_child_samples': 20,
        'reg_alpha': 0.0,
        'reg_lambda': 0.0,
        # 🎯 Focal loss parameters for minority class emphasis
        'is_unbalance': False,  # We use sample_weights instead
        'boost_from_average': False,  # Don't average, focus on minority
    }
    
    XGB_PARAMS = {
        'objective': 'binary:logistic',
        'eval_metric': 'logloss',
        'tree_method': 'hist',
        'max_depth': 6,
        'learning_rate': 0.05,
        'subsample': 0.8,
        'colsample_bytree': 0.9,
        'min_child_weight': 1,
        'reg_alpha': 0,
        'reg_lambda': 1,
        'n_estimators': 1000,
        'random_state': 42,
    }

config = MultiTowerConfig()

# ==================== PREPARE MULTI-TOWER FEATURES ====================
def prepare_multitower_features(df: pd.DataFrame, 
                                config: MultiTowerConfig,
                                target_col: str,
                                horizon_name: str,
                                lag_config: str = 'long',
                                use_rolling: bool = True,
                                rolling_windows: List[int] = None) -> Tuple[pd.DataFrame, pd.Series, List[str], pd.DataFrame]:
    """
    Prepare features for multi-tower modeling with PIVOTED data.
    Data is already in wide format with one row per timestamp.
    Uses event-specific lags, rolling windows, and forecast horizons.
    
    Returns:
        X: Feature dataframe
        y: Target series
        feature_names: List of feature names
        metadata: Metadata (timestamp) for each sample
    """
    
    # Use event-specific parameters
    if rolling_windows is None:
        rolling_windows = config.ROLLING_WINDOWS_PER_EVENT.get(target_col, [2, 6, 12, 24])
    
    lags = config.LAG_CONFIGS_PER_EVENT.get(target_col, list(range(1, 25)))
    
    # Sort by timestamp
    df = df.sort_values('timestamp').reset_index(drop=True)
    
    # Get feature columns (exclude timestamp, events)
    exclude_cols = ['timestamp'] + config.TARGET_EVENTS
    numeric_cols = [c for c in df.columns 
                   if c not in exclude_cols 
                   and pd.api.types.is_numeric_dtype(df[c])]
    
    print(f"\nCreating lag/rolling features for {len(numeric_cols)} base columns...", flush=True)
    print(f"  Base data shape: {df.shape}", flush=True)
    print(f"  Base data date range: {df['timestamp'].min()} to {df['timestamp'].max()}", flush=True)
    print(f"  Event '{target_col}' distribution: {df[target_col].sum()} events ({df[target_col].mean()*100:.2f}%)", flush=True)
    
    feature_dfs = []
    feature_names = []
    
    # 1. Lag features
    print(f"  Creating lag features: {lags}", flush=True)
    for col in numeric_cols:
        for lag in lags:
            lag_col = f'{col}_lag{lag}'
            feature_dfs.append(df[col].shift(lag).to_frame(lag_col))
            feature_names.append(lag_col)
    
    # 2. Rolling features (ENHANCED: mean, std, min, max)
    if use_rolling:
        print(f"  Creating rolling features: windows={rolling_windows}", flush=True)
        for col in numeric_cols:
            for window in rolling_windows:
                # Mean
                roll_mean_col = f'{col}_roll{window}_mean'
                feature_dfs.append(df[col].rolling(window=window).mean().to_frame(roll_mean_col))
                feature_names.append(roll_mean_col)
                
                # Std (volatility)
                roll_std_col = f'{col}_roll{window}_std'
                feature_dfs.append(df[col].rolling(window=window).std().to_frame(roll_std_col))
                feature_names.append(roll_std_col)
                
                # Min (extreme low)
                roll_min_col = f'{col}_roll{window}_min'
                feature_dfs.append(df[col].rolling(window=window).min().to_frame(roll_min_col))
                feature_names.append(roll_min_col)
                
                # Max (extreme high)
                roll_max_col = f'{col}_roll{window}_max'
                feature_dfs.append(df[col].rolling(window=window).max().to_frame(roll_max_col))
                feature_names.append(roll_max_col)
    
    # 3. 🔥 RATE OF CHANGE (1st & 2nd derivatives) - Critical for extreme events!
    print(f"  Creating rate of change features (derivatives)...", flush=True)
    for col in numeric_cols:
        # 1st derivative (velocity) - rate of change
        diff1_col = f'{col}_diff1'
        feature_dfs.append(df[col].diff(1).to_frame(diff1_col))
        feature_names.append(diff1_col)
        
        # 2nd derivative (acceleration) - rate of rate of change
        diff2_col = f'{col}_diff2'
        feature_dfs.append(df[col].diff(1).diff(1).to_frame(diff2_col))
        feature_names.append(diff2_col)
    
    # 4. 🔥 INTER-TOWER SPATIAL GRADIENTS - Extreme events show spatial patterns!
    print(f"  Creating inter-tower spatial gradient features...", flush=True)
    towers = ['TOWA', 'TOWB', 'TOWD', 'TOWF', 'TOWS', 'TOWY']
    
    # Key variables for spatial analysis
    spatial_vars = ['TempC_030m', 'PkWSpdMph_030m', 'RH_030m']
    
    for var in spatial_vars:
        # Find towers that have this variable
        tower_cols = [f'{t}_{var}' for t in towers if f'{t}_{var}' in df.columns]
        
        if len(tower_cols) >= 2:
            # Mean across towers
            mean_col = f'{var}_tower_mean'
            feature_dfs.append(df[tower_cols].mean(axis=1).to_frame(mean_col))
            feature_names.append(mean_col)
            
            # Std across towers (spatial variability)
            std_col = f'{var}_tower_std'
            feature_dfs.append(df[tower_cols].std(axis=1).to_frame(std_col))
            feature_names.append(std_col)
            
            # Range across towers (max-min spatial gradient)
            range_col = f'{var}_tower_range'
            feature_dfs.append((df[tower_cols].max(axis=1) - df[tower_cols].min(axis=1)).to_frame(range_col))
            feature_names.append(range_col)
            
            # Deviation from tower mean (anomaly detection)
            for tower_col in tower_cols:
                dev_col = f'{tower_col}_dev_from_mean'
                tower_mean = df[tower_cols].mean(axis=1)
                feature_dfs.append((df[tower_col] - tower_mean).to_frame(dev_col))
                feature_names.append(dev_col)
    
    # 5. 🔥 TEMPORAL CONTEXT - Extreme events have time-of-day/seasonal patterns!
    print(f"  Creating temporal context features...", flush=True)
    hour = df['timestamp'].dt.hour
    day_of_week = df['timestamp'].dt.dayofweek
    month = df['timestamp'].dt.month
    
    # Cyclical encoding for temporal features
    feature_dfs.append(np.sin(2 * np.pi * hour / 24).to_frame('hour_sin'))
    feature_names.append('hour_sin')
    feature_dfs.append(np.cos(2 * np.pi * hour / 24).to_frame('hour_cos'))
    feature_names.append('hour_cos')
    
    feature_dfs.append(np.sin(2 * np.pi * day_of_week / 7).to_frame('dow_sin'))
    feature_names.append('dow_sin')
    feature_dfs.append(np.cos(2 * np.pi * day_of_week / 7).to_frame('dow_cos'))
    feature_names.append('dow_cos')
    
    feature_dfs.append(np.sin(2 * np.pi * month / 12).to_frame('month_sin'))
    feature_names.append('month_sin')
    feature_dfs.append(np.cos(2 * np.pi * month / 12).to_frame('month_cos'))
    feature_names.append('month_cos')
    
    # 6. 🔥 THRESHOLD PROXIMITY - How close to danger zone!
    print(f"  Creating threshold proximity features...", flush=True)
    
    if 'LowTemp' in target_col:
        # For low temp events: distance to 3°C
        for tower in towers:
            temp_col = f'{tower}_TempC_030m'
            if temp_col in df.columns:
                prox_col = f'{temp_col}_dist_to_3C'
                feature_dfs.append((df[temp_col] - 3.0).to_frame(prox_col))
                feature_names.append(prox_col)
    
    elif 'HighWind' in target_col:
        # For high wind events: distance to 20 mph
        for tower in towers:
            wind_col = f'{tower}_PkWSpdMph_030m'
            if wind_col in df.columns:
                prox_col = f'{wind_col}_dist_to_20mph'
                feature_dfs.append((df[wind_col] - 20.0).to_frame(prox_col))
                feature_names.append(prox_col)
    
    elif 'LowWind' in target_col:
        # For low wind events: distance to 4 mph
        for tower in towers:
            wind_col = f'{tower}_PkWSpdMph_030m'
            if wind_col in df.columns:
                prox_col = f'{wind_col}_dist_to_4mph'
                feature_dfs.append((df[wind_col] - 4.0).to_frame(prox_col))
                feature_names.append(prox_col)
    
    # Combine all features
    X = pd.concat(feature_dfs, axis=1)
    
    # Target: shift forward by forecast horizon (EVENT-SPECIFIC)
    # Note: horizon_name passed as parameter to this function
    horizon_steps = config.FORECAST_HORIZONS_PER_EVENT[target_col][horizon_name]
    print(f"  Creating target (shift -{horizon_steps} steps ahead for {horizon_name})...", flush=True)
    print(f"    Event: {target_col}, Resample: {config.EVENT_RESAMPLE_RULES[target_col]}", flush=True)
    
    # Ensure target column is numeric
    if df[target_col].dtype == 'bool':
        y = df[target_col].astype(int).shift(-horizon_steps)
    else:
        y = df[target_col].shift(-horizon_steps)
    
    # Add timestamp for metadata
    metadata = df[['timestamp']].copy()
    
    # DEBUG: Show data before dropna
    print(f"\n  📊 BEFORE dropna:", flush=True)
    print(f"     Features shape: {X.shape}", flush=True)
    print(f"     Target shape: {y.shape}", flush=True)
    print(f"     NaN in features: {X.isna().any(axis=1).sum():,} rows", flush=True)
    print(f"     NaN in target: {y.isna().sum():,} rows", flush=True)
    
    # DEBUG: Show what's in target column BEFORE shift
    print(f"\n  📊 TARGET COLUMN DEBUG (before shift):", flush=True)
    print(f"     df['{target_col}'] dtype: {df[target_col].dtype}", flush=True)
    print(f"     df['{target_col}'] unique values: {df[target_col].unique()}", flush=True)
    print(f"     df['{target_col}'] value counts:\n{df[target_col].value_counts()}", flush=True)
    print(f"     df['{target_col}'] sum: {df[target_col].sum()}", flush=True)
    
    # DEBUG: Show what's in y (after shift)
    print(f"\n  📊 TARGET AFTER SHIFT (y):", flush=True)
    print(f"     y dtype: {y.dtype}", flush=True)
    print(f"     y unique values: {y.unique()[:10]}", flush=True)
    print(f"     y value counts:\n{y.value_counts()}", flush=True)
    if y.notna().any():
        print(f"     Event rate (with NaN): {y.mean()*100:.2f}% ({y.sum():,} events)", flush=True)
    else:
        print(f"     ⚠️  y is ALL NaN!", flush=True)
    
    # Drop rows with NaN in features or target
    print(f"\n  Dropping rows with NaN values...", flush=True)
    valid_idx = X.notna().all(axis=1) & y.notna()
    
    X = X[valid_idx].reset_index(drop=True)
    y = y[valid_idx].reset_index(drop=True)
    metadata = metadata[valid_idx].reset_index(drop=True)
    
    print(f"\n✓ Final dataset: {len(X):,} samples", flush=True)
    print(f"✓ Features: {len(feature_names)}", flush=True)
    if len(X) > 0:
        print(f"✓ Event rate: {y.mean()*100:.2f}% ({y.sum():,} events)", flush=True)
    else:
        print(f"✓ Event rate: nan% (0 events)", flush=True)
    
    # Show sample of final data that will be inputted to model
    if len(X) > 0:
        print(f"\n" + "="*80, flush=True)
        print(f"📊 FINAL DATA TO BE INPUTTED TO MODEL", flush=True)
        print(f"="*80, flush=True)
        print(f"X (features) shape: {X.shape}", flush=True)
        print(f"y (target) shape: {y.shape}", flush=True)
        print(f"y dtype: {y.dtype}", flush=True)
        print(f"y value counts:\n{y.value_counts()}", flush=True)
        print(f"\nFirst 5 samples of X:", flush=True)
        print(X.head(5).to_string(), flush=True)
        print(f"\nFirst 20 target values (y): {y.head(20).values}", flush=True)
        print(f"\n" + "-"*80, flush=True)
        print(f"SAMPLE DATA BEFORE MODEL INPUT (first 5 rows, first 10 features):", flush=True)
        print("-"*80, flush=True)
        sample_df = pd.DataFrame(X.iloc[:5, :10].values, columns=feature_names[:10])
        sample_df.insert(0, 'timestamp', metadata['timestamp'].iloc[:5].values)
        sample_df['target'] = y.iloc[:5].values
        print(sample_df.to_string(index=False), flush=True)
        print(f"\nFull feature count: {X.shape[1]}", flush=True)
        print("-"*80 + "\n", flush=True)
    
    return X, y, feature_names, metadata

# ==================== MODEL TRAINING ====================
def train_lightgbm_model(X_train, y_train, X_val, y_val, params, sample_weights=None):
    """Train LightGBM model"""
    
    # Create datasets
    train_data = lgb.Dataset(X_train, label=y_train, weight=sample_weights)
    val_data = lgb.Dataset(X_val, label=y_val, reference=train_data)
    
    # Train
    model = lgb.train(
        params,
        train_data,
        valid_sets=[val_data],
        callbacks=[lgb.early_stopping(stopping_rounds=100, verbose=False)]  # Increased from 50
    )
    
    return model


def train_xgboost_model(X_train, y_train, X_val, y_val, params, sample_weights=None):
    """Train XGBoost model"""
    
    # Calculate scale_pos_weight from sample_weights if provided
    if sample_weights is not None:
        pos_weight = sample_weights[y_train == 1].mean() / sample_weights[y_train == 0].mean()
        params = params.copy()
        params['scale_pos_weight'] = pos_weight
    
    # Create DMatrix
    dtrain = xgb.DMatrix(X_train, label=y_train, weight=sample_weights)
    dval = xgb.DMatrix(X_val, label=y_val)
    
    # Train
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=params.get('n_estimators', 1000),
        evals=[(dval, 'validation')],
        early_stopping_rounds=100,  # Increased from 50
        verbose_eval=False
    )
    
    return model


def calculate_all_metrics(y_true, y_pred, y_pred_proba):
    """Calculate comprehensive classification metrics"""
    # Handle case where only one class is predicted (e.g., all zeros or all ones)
    cm = confusion_matrix(y_true, y_pred)
    if cm.size == 1:
        # Only one class present - handle gracefully
        if y_pred[0] == 0:  # All predicted as negative
            tn, fp, fn, tp = cm[0, 0], 0, 0, 0
        else:  # All predicted as positive
            tn, fp, fn, tp = 0, 0, 0, cm[0, 0]
    else:
        tn, fp, fn, tp = cm.ravel()
    
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


def calculate_shap_importance(model, X_sample, feature_names, model_type='lightgbm', max_samples=1000):
    """Calculate SHAP values for feature importance"""
    print(f"\n      Computing SHAP values for feature importance...", flush=True)
    
    # Limit sample size for computational efficiency
    n_samples = min(max_samples, len(X_sample))
    X_shap = X_sample.iloc[:n_samples] if hasattr(X_sample, 'iloc') else X_sample[:n_samples]
    
    try:
        # Create explainer based on model type
        if model_type.lower() == 'xgboost':
            explainer = shap.TreeExplainer(model)
        else:  # lightgbm
            explainer = shap.TreeExplainer(model)
        
        # Calculate SHAP values
        shap_values = explainer.shap_values(X_shap)
        
        # For binary classification, shap_values might be a list
        if isinstance(shap_values, list):
            shap_values = shap_values[1]  # Use positive class
        
        # Calculate mean absolute SHAP values for each feature
        mean_abs_shap = np.abs(shap_values).mean(axis=0)
        
        # Create importance dataframe
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'shap_importance': mean_abs_shap
        }).sort_values('shap_importance', ascending=False)
        
        print(f"      ✓ SHAP values computed for {n_samples} samples", flush=True)
        
        return importance_df, shap_values
        
    except Exception as e:
        print(f"      ⚠️  SHAP calculation failed: {e}", flush=True)
        return None, None


def train_multitower_event_models(X: pd.DataFrame, y: pd.Series, 
                                  metadata: pd.DataFrame,
                                  config: MultiTowerConfig,
                                  event_name: str,
                                  horizon_name: str) -> Dict:
    """
    Train ONE model for ALL towers for a specific event
    
    Key innovation: Single model learns patterns across all towers
    """
    
    # ==================== TEMPORAL SPLIT: 80% train, 20% final test ====================
    n_samples = len(X)
    train_size = int(n_samples * 0.8)
    
    X_train_full = X.iloc[:train_size]
    y_train_full = y.iloc[:train_size]
    X_test = X.iloc[train_size:]
    y_test = y.iloc[train_size:]
    metadata_test = metadata.iloc[train_size:]
    
    print(f"      Training samples: {len(X_train_full):,}, Test samples: {len(X_test):,}", flush=True)
    print(f"      Forecast horizon: {horizon_name}", flush=True)
    
    # K-fold CV on training portion only
    tscv = TimeSeriesSplit(n_splits=config.N_SPLITS)
    
    results = {
        'event': event_name,
        'models': {}
    }
    
    # Initialize results for each model type
    for model_type in config.MODELS_TO_TRAIN:
        results['models'][model_type] = {
            'trained_models': [],
            'fold_metadata': [],
            'final_test_metrics': {},
            'cv_metrics': {}
        }
        # Initialize metric lists
        for metric in ['accuracy', 'balanced_accuracy', 'precision', 'recall', 'f1', 
                      'auc_roc', 'auc_pr', 'specificity', 'mcc', 'cohen_kappa',
                      'true_positives', 'true_negatives', 'false_positives', 'false_negatives']:
            results['models'][model_type]['cv_metrics'][metric] = []
    
    # ==================== K-FOLD CV ON TRAINING PORTION ====================
    print(f"      Running {config.N_SPLITS}-fold cross-validation...", flush=True)
    
    for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train_full), 1):
        X_train, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_train, y_val = y.iloc[train_idx], y.iloc[val_idx]
        
        # 🎯 MAXIMUM class weights for extreme imbalance (1.5% positive)
        # Use power 0.75 (same as CNN/DL) for strong minority class emphasis
        if config.USE_CLASS_WEIGHTS:
            n_pos = y_train.sum()
            n_neg = len(y_train) - n_pos
            
            # Power 0.75: (65.67)^0.75 = 21.5× for 1.5% rate
            # Much stronger than balanced (1.0× to 1.0×)
            pos_weight = np.power(n_neg / n_pos, 0.75) if n_pos > 0 else 1.0
            sample_weights = np.where(y_train == 1, pos_weight, 1.0)
            
            print(f"         Class weight: 1.0 (neg) vs {pos_weight:.2f} (pos)", flush=True)
        else:
            sample_weights = None
        
        # Train each model type
        for model_type in config.MODELS_TO_TRAIN:
            
            if model_type == 'lightgbm':
                model_params = config.LGBM_PARAMS
                model = train_lightgbm_model(X_train, y_train, X_val, y_val, 
                                            model_params, sample_weights)
                y_pred_proba = model.predict(X_val)
                
            elif model_type == 'xgboost':
                model_params = config.XGB_PARAMS
                model = train_xgboost_model(X_train, y_train, X_val, y_val,
                                           model_params, sample_weights)
                dval = xgb.DMatrix(X_val)
                y_pred_proba = model.predict(dval)
            
            # Optimal threshold using F2 score (prioritizes recall 2x more than precision)
            from sklearn.metrics import fbeta_score
            
            best_threshold = 0.3
            best_f2 = 0
            for threshold in np.arange(0.1, 0.6, 0.05):
                y_pred_temp = (y_pred_proba >= threshold).astype(int)
                f2 = fbeta_score(y_val, y_pred_temp, beta=2.0, zero_division=0)
                if f2 > best_f2:
                    best_f2 = f2
                    best_threshold = threshold
            
            optimal_threshold = best_threshold
            y_pred = (y_pred_proba >= optimal_threshold).astype(int)
            
            # Calculate all metrics
            metrics = calculate_all_metrics(y_val, y_pred, y_pred_proba)
            
            # Get confusion matrix for this fold
            cm_fold = confusion_matrix(y_val, y_pred)
            tn_fold, fp_fold, fn_fold, tp_fold = cm_fold.ravel()
            
            # Print confusion matrix for this fold
            print(f"\n         {model_type.upper()} Fold {fold}/{config.N_SPLITS} Confusion Matrix:", flush=True)
            print(f"                              Predicted Negative | Predicted Positive", flush=True)
            print(f"         Actual Negative:  TN={tn_fold:6,} | FP={fp_fold:6,}", flush=True)
            print(f"         Actual Positive:  FN={fn_fold:6,} | TP={tp_fold:6,}", flush=True)
            print(f"         Metrics: F1={metrics['f1']:.4f}, Recall={metrics['recall']:.4f}, Precision={metrics['precision']:.4f}", flush=True)
            
            # Store metrics
            for metric_name, metric_value in metrics.items():
                results['models'][model_type]['cv_metrics'][metric_name].append(metric_value)
            
            # Store model
            results['models'][model_type]['trained_models'].append(model)
        
        print(f"\n         Fold {fold}/{config.N_SPLITS} complete", flush=True)
    
    # ==================== FINAL TEST SET EVALUATION ====================
    print(f"      Evaluating on final held-out test set...", flush=True)
    
    for model_type in config.MODELS_TO_TRAIN:
        
        if model_type == 'lightgbm':
            model_params = config.LGBM_PARAMS
        else:
            model_params = config.XGB_PARAMS
        
        # 🎯 MAXIMUM class weights for extreme imbalance (1.5% positive)
        # Use power 0.75 (same as CNN/DL) for strong minority class emphasis
        if config.USE_CLASS_WEIGHTS:
            n_pos = y_train_full.sum()
            n_neg = len(y_train_full) - n_pos
            
            # Power 0.75: (65.67)^0.75 = 21.5× for 1.5% rate
            pos_weight = np.power(n_neg / n_pos, 0.75) if n_pos > 0 else 1.0
            sample_weights = np.where(y_train_full == 1, pos_weight, 1.0)
        else:
            sample_weights = None
        
        # Train final model on ALL training data
        if model_type == 'lightgbm':
            final_model = train_lightgbm_model(X_train_full, y_train_full, X_test, y_test,
                                              model_params, sample_weights)
            y_test_pred_proba = final_model.predict(X_test)
        else:
            final_model = train_xgboost_model(X_train_full, y_train_full, X_test, y_test,
                                             model_params, sample_weights)
            dtest = xgb.DMatrix(X_test)
            y_test_pred_proba = final_model.predict(dtest)
        
        # Optimal threshold using F2 score (prioritizes recall 2x more than precision)
        from sklearn.metrics import fbeta_score
        
        best_threshold = 0.3
        best_f2 = 0
        for threshold in np.arange(0.1, 0.6, 0.05):
            y_pred_temp = (y_test_pred_proba >= threshold).astype(int)
            f2 = fbeta_score(y_test, y_pred_temp, beta=2.0, zero_division=0)
            if f2 > best_f2:
                best_f2 = f2
                best_threshold = threshold
        
        optimal_threshold = best_threshold
        y_test_pred = (y_test_pred_proba >= optimal_threshold).astype(int)
        
        # Calculate test metrics
        test_metrics = calculate_all_metrics(y_test, y_test_pred, y_test_pred_proba)
        
        # Get confusion matrix
        cm = confusion_matrix(y_test, y_test_pred)
        tn, fp, fn, tp = cm.ravel()
        
        # Store final model and test metrics
        results['models'][model_type]['final_model'] = final_model
        results['models'][model_type]['final_test_metrics'] = test_metrics
        results['models'][model_type]['optimal_threshold'] = optimal_threshold
        results['models'][model_type]['confusion_matrix'] = cm
        
        # Print confusion matrix
        print(f"\n      {model_type.upper()} Confusion Matrix:", flush=True)
        print(f"                    Predicted Negative | Predicted Positive", flush=True)
        print(f"         Actual Negative:  TN={tn:6,} | FP={fp:6,}", flush=True)
        print(f"         Actual Positive:  FN={fn:6,} | TP={tp:6,}", flush=True)
        
        # Calculate SHAP importance
        shap_importance, shap_values = calculate_shap_importance(
            final_model, X_test, X.columns.tolist(), model_type=model_type, max_samples=1000
        )
        
        if shap_importance is not None:
            print(f"\n      Top 20 Most Important Features (SHAP):", flush=True)
            for idx, row in shap_importance.head(20).iterrows():
                print(f"         {idx+1:2d}. {row['feature'][:45]:45s}: {row['shap_importance']:.6f}", flush=True)
            
            # Store SHAP results
            results['models'][model_type]['shap_importance'] = shap_importance
            results['models'][model_type]['shap_values'] = shap_values
        
        # Print results
        print(f"\n      {model_type.upper()} Results:", flush=True)
        print(f"         CV: AUC={np.mean(results['models'][model_type]['cv_metrics']['auc_roc']):.4f}, "
              f"F1={np.mean(results['models'][model_type]['cv_metrics']['f1']):.4f}, "
              f"MCC={np.mean(results['models'][model_type]['cv_metrics']['mcc']):.4f}", flush=True)
        print(f"         Test: AUC={test_metrics['auc_roc']:.4f}, "
              f"F1={test_metrics['f1']:.4f}, "
              f"MCC={test_metrics['mcc']:.4f}", flush=True)
    
    return results


# ==================== SAVE RESULTS ====================
def save_multitower_results(all_results: Dict, config: MultiTowerConfig):
    """Save all results to disk"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"ml_results_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Save config
    config_dict = {
        'target_events': config.TARGET_EVENTS,
        'models_trained': config.MODELS_TO_TRAIN,
        'n_splits': config.N_SPLITS,
        'lag_config': config.SELECTED_LAG_CONFIG,
        'rolling_windows_per_event': config.ROLLING_WINDOWS_PER_EVENT,
        'forecast_horizons_per_event': config.SELECTED_HORIZONS_PER_EVENT,
        'timestamp': timestamp
    }
    
    with open(f"{output_dir}/config.json", 'w') as f:
        json.dump(config_dict, f, indent=2)
    
    # Save models
    models_dir = f"{output_dir}/models"
    os.makedirs(models_dir, exist_ok=True)
    
    # Create summary report
    report_lines = []
    report_lines.append("="*80)
    report_lines.append("MULTI-TOWER ML RESULTS SUMMARY")
    report_lines.append("="*80)
    report_lines.append(f"Timestamp: {timestamp}")
    report_lines.append(f"Approach: Single model per event, trained on ALL towers")
    report_lines.append("")
    
    # Process each horizon and event
    for horizon_name, horizon_results in all_results.items():
        
        report_lines.append(f"\n{'#'*80}")
        report_lines.append(f"# FORECAST HORIZON: {horizon_name}")
        report_lines.append(f"{'#'*80}")
        
        for event_name, event_results in horizon_results.items():
            
            report_lines.append(f"\n{'='*70}")
            report_lines.append(f"EVENT: {event_name}")
            report_lines.append(f"{'='*70}")
            
            for model_type in config.MODELS_TO_TRAIN:
                if model_type not in event_results['models']:
                    continue
                    
                model_data = event_results['models'][model_type]
                
                # Save final model with horizon in filename
                model_path = f"{models_dir}/{event_name}_{horizon_name}_{model_type}_final.pkl"
                with open(model_path, 'wb') as f:
                    pickle.dump(model_data['final_model'], f)
                
                # Get metrics
                cv_metrics = model_data['cv_metrics']
                test_metrics = model_data['final_test_metrics']
            
            report_lines.append(f"\n{model_type.upper()}:")
            report_lines.append(f"  Cross-Validation (5-fold):")
            report_lines.append(f"    AUC: {np.mean(cv_metrics['auc_roc']):.4f} ± {np.std(cv_metrics['auc_roc']):.4f}")
            report_lines.append(f"    F1:  {np.mean(cv_metrics['f1']):.4f} ± {np.std(cv_metrics['f1']):.4f}")
            report_lines.append(f"    MCC: {np.mean(cv_metrics['mcc']):.4f} ± {np.std(cv_metrics['mcc']):.4f}")
            
            report_lines.append(f"\n  Final Test Set (Overall):")
            report_lines.append(f"    AUC: {test_metrics['auc_roc']:.4f}")
            report_lines.append(f"    F1:  {test_metrics['f1']:.4f}")
            report_lines.append(f"    MCC: {test_metrics['mcc']:.4f}")
            report_lines.append(f"    Balanced Accuracy: {test_metrics['balanced_accuracy']:.4f}")
            report_lines.append(f"    Cohen's Kappa: {test_metrics['cohen_kappa']:.4f}")
            report_lines.append(f"    Precision: {test_metrics['precision']:.4f}")
            report_lines.append(f"    Recall: {test_metrics['recall']:.4f}")
            
            # Add SHAP importance
            if 'shap_importance' in model_data and model_data['shap_importance'] is not None:
                shap_importance = model_data['shap_importance']
                report_lines.append(f"\n  Top 10 Most Important Features (SHAP):")
                for idx, row in shap_importance.head(10).iterrows():
                    report_lines.append(f"    {idx+1:2d}. {row['feature'][:40]:40s}: {row['shap_importance']:.6f}")
                
                # Save full SHAP importance to CSV
                shap_csv = f"{models_dir}/{event_name}_{horizon_name}_{model_type}_shap_importance.csv"
                shap_importance.to_csv(shap_csv, index=False)
                report_lines.append(f"\n  SHAP importance saved to: {shap_csv}")
            
            # Only print per-tower if available (removed in current version)
            if 'per_tower_test_metrics' in model_data and model_data['per_tower_test_metrics']:
                report_lines.append(f"\n  Per-Tower Test Performance:")
                per_tower = model_data['per_tower_test_metrics']
                for tower in sorted(per_tower.keys()):
                    tm = per_tower[tower]
                    report_lines.append(f"    {tower}: F1={tm['f1']:.4f}, MCC={tm['mcc']:.4f}, "
                                      f"AUC={tm['auc_roc']:.4f}, Events={tm['true_positives'] + tm['false_negatives']}")
    
    # Save report
    report_text = "\n".join(report_lines)
    with open(f"{output_dir}/experiment_report.txt", 'w') as f:
        f.write(report_text)
    
    print(f"\n✅ Results saved to: {output_dir}")
    print(f"   - Models: {models_dir}")
    print(f"   - Report: {output_dir}/experiment_report.txt")
    print(f"   - Config: {output_dir}/config.json")
    
    return output_dir


# ==================== MAIN EXECUTION ====================
if __name__ == "__main__":
    
    print("\n" + "="*80)
    print("RUNNING MULTI-TOWER EXPERIMENTS")
    print("="*80)
    print(f"Models: {', '.join(config.MODELS_TO_TRAIN)}")
    print(f"Events: {len(config.TARGET_EVENTS)}")
    print(f"Event-specific resampling: {config.EVENT_RESAMPLE_RULES}")
    print("="*80)
    
    # Collect all unique horizons across all events
    all_horizons = set()
    for event_horizons in config.SELECTED_HORIZONS_PER_EVENT.values():
        all_horizons.update(event_horizons)
    all_horizons = sorted(all_horizons, key=lambda x: config.FORECAST_HORIZONS_PER_EVENT[config.TARGET_EVENTS[0]].get(x, 999))
    
    print(f"Horizons to test: {', '.join(all_horizons)}")
    print("="*80)
    
    # Store results per horizon
    all_results = {}  # Format: {horizon: {event: results}}
    summary_data = []  # For comprehensive table
    
    # Train models for each horizon
    for horizon_name in all_horizons:
        
        print(f"\n{'='*70}")
        print(f"⏰ FORECAST HORIZON: {horizon_name}")
        print(f"{'='*70}")
        
        all_results[horizon_name] = {}
        
        # Train ONE model per event (across all towers) for this horizon
        for event_col in config.TARGET_EVENTS:
            
            print(f"\n{'='*70}")
            print(f"📍 EVENT: {event_col} | HORIZON: {horizon_name}")
            print(f"{'='*70}")
            
            # Check if this horizon is valid for this event
            if horizon_name not in config.SELECTED_HORIZONS_PER_EVENT.get(event_col, []):
                print(f"   ⏭️  Horizon {horizon_name} not configured for {event_col}, skipping...")
                continue
            
            try:
                # 🎯 RESAMPLE DATA FOR THIS EVENT
                resample_rule = config.EVENT_RESAMPLE_RULES[event_col]
                df_event = resample_for_event(df_filtered, event_col, resample_rule, event_cols)
                
                # Prepare multi-tower features with specific horizon
                X, y, feature_names, metadata = prepare_multitower_features(
                    df_event, config, event_col,
                    lag_config=config.SELECTED_LAG_CONFIG,
                    use_rolling=True,
                    rolling_windows=None,  # Will use event-specific windows
                    horizon_name=horizon_name
                )
                
                # Check if sufficient data
                if len(y) < 100:
                    print(f"   ⚠️  Insufficient data ({len(y)} samples), skipping...")
                    continue
                
                if y.sum() < 10:
                    print(f"   ⚠️  Too few positive events ({y.sum()}), skipping...")
                    continue
                
                # Train models
                results = train_multitower_event_models(X, y, metadata, config, event_col, horizon_name)
                
                all_results[horizon_name][event_col] = results
                
                # Extract metrics for summary table
                for model_type in config.MODELS_TO_TRAIN:
                    if model_type in results['models']:
                        model_data = results['models'][model_type]
                        test_metrics = model_data['final_test_metrics']
                        cv_metrics = model_data['cv_metrics']
                        
                        summary_data.append({
                            'Horizon': horizon_name,
                            'Event': event_col.replace('event_', ''),
                            'Model': model_type.upper(),
                            'CV_AUC': np.mean(cv_metrics['auc_roc']),
                            'CV_F1': np.mean(cv_metrics['f1']),
                            'CV_MCC': np.mean(cv_metrics['mcc']),
                            'Test_AUC': test_metrics['auc_roc'],
                            'Test_F1': test_metrics['f1'],
                            'Test_MCC': test_metrics['mcc'],
                            'Test_Precision': test_metrics['precision'],
                            'Test_Recall': test_metrics['recall'],
                            'Test_BalAcc': test_metrics['balanced_accuracy'],
                            'Test_Kappa': test_metrics['cohen_kappa']
                        })
                
            except Exception as e:
                print(f"   ❌ Error: {str(e)}")
                import traceback
                traceback.print_exc()
    
    # Create and save comprehensive summary table
    if summary_data:
        summary_df = pd.DataFrame(summary_data)
        
        # Save all results
        output_dir = save_multitower_results(all_results, config)
        
        # Save summary table
        summary_csv = f"{output_dir}/summary_all_horizons.csv"
        summary_df.to_csv(summary_csv, index=False, float_format='%.4f')
        
        # Print summary table
        print("\n" + "="*80)
        print("📊 COMPREHENSIVE SUMMARY: ALL HORIZONS, EVENTS, AND MODELS")
        print("="*80)
        print(summary_df.to_string(index=False))
        
        # Print best performers per metric
        print("\n" + "="*80)
        print("🏆 BEST PERFORMERS")
        print("="*80)
        
        for metric in ['Test_AUC', 'Test_F1', 'Test_MCC']:
            best_row = summary_df.loc[summary_df[metric].idxmax()]
            print(f"\nBest {metric}:")
            print(f"  {best_row['Model']} on {best_row['Event']} at {best_row['Horizon']}")
            print(f"  {metric}={best_row[metric]:.4f}")
        

        print("\n" + "="*80)
        print("✅ MULTI-TOWER EXPERIMENTS COMPLETE!")
        print("="*80)
        print(f"Results saved to: {output_dir}")
        print(f"Summary table: {summary_csv}")
        print(f"LaTeX table: {latex_file}")
    else:
        print("\n⚠️  No results to save")


def generate_latex_summary_table(df: pd.DataFrame) -> str:
    """
    Generate a LaTeX table from the summary dataframe
    
    Creates a publication-ready table with:
    - Best results highlighted per event
    - Grouped by horizon
    - Clean formatting
    """
    
    lines = []
    
    # Table header
    lines.append("\\begin{table}[htbp]")
    lines.append("\\centering")
    lines.append("\\caption{Multi-Tower ML Performance Across Prediction Horizons}")
    lines.append("\\label{tab:multitower_results}")
    lines.append("\\resizebox{\\textwidth}{!}{%")
    lines.append("\\begin{tabular}{llcccccccc}")
    lines.append("\\toprule")
    lines.append("\\textbf{Horizon} & \\textbf{Event} & \\textbf{Model} & \\textbf{CV AUC} & \\textbf{CV F1} & \\textbf{CV MCC} & \\textbf{Test AUC} & \\textbf{Test F1} & \\textbf{Test MCC} & \\textbf{Test Recall} \\\\")
    lines.append("\\midrule")
    
    # Group by horizon and event
    for horizon in df['Horizon'].unique():
        horizon_df = df[df['Horizon'] == horizon]
        
        # Add horizon header
        lines.append(f"\\multicolumn{{10}}{{l}}{{\\textit{{{horizon}}}}} \\\\")
        
        for event in horizon_df['Event'].unique():
            event_df = horizon_df[horizon_df['Event'] == event]
            
            # Find best model for this event-horizon
            best_idx = event_df['Test_F1'].idxmax()
            
            for idx, row in event_df.iterrows():
                model = row['Model']
                
                # Highlight best model
                if idx == best_idx:
                    model_str = f"\\textbf{{{model}}}"
                    f1_str = f"\\textbf{{{row['Test_F1']:.4f}}}"
                else:
                    model_str = model
                    f1_str = f"{row['Test_F1']:.4f}"
                
                # Clean event name
                event_clean = event.replace('_', ' ')
                
                line = (f"& {event_clean:25s} & {model_str:15s} & "
                       f"{row['CV_AUC']:.4f} & {row['CV_F1']:.4f} & {row['CV_MCC']:.4f} & "
                       f"{row['Test_AUC']:.4f} & {f1_str} & {row['Test_MCC']:.4f} & "
                       f"{row['Test_Recall']:.4f} \\\\")
                lines.append(line)
            
            lines.append("\\addlinespace[0.5em]")
        
        lines.append("\\midrule")
    
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}%")
    lines.append("}")
    lines.append("\\end{table}")
    
    return "\n".join(lines)

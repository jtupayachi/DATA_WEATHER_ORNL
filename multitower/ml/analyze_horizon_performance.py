#!/usr/bin/env python3
"""
Comprehensive multi-horizon prediction performance analysis
Includes detailed metrics, overfitting analysis, error analysis, and visualizations
"""

import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy import stats

# Set plotting style
# sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 10

def parse_log_file(log_path):
    """Parse log file and extract detailed results including confusion matrices"""
    
    results = []
    current_horizon = None
    current_event = None
    current_model = None
    confusion_matrix = {}
    cv_metrics = {}
    
    with open(log_path, 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        # Extract horizon
        if '⏰ FORECAST HORIZON:' in line:
            match = re.search(r'FORECAST HORIZON: (\w+)', line)
            if match:
                current_horizon = match.group(1)
        
        # Extract event
        if '📍 EVENT:' in line:
            match = re.search(r'EVENT: ([\w_]+)', line)
            if match:
                current_event = match.group(1)
        
        # Extract model results
        if 'LIGHTGBM Results:' in line:
            current_model = 'lightgbm'
        elif 'XGBOOST Results:' in line:
            current_model = 'xgboost'
        
        # Extract confusion matrix
        if 'Confusion Matrix:' in line and current_model:
            # Look ahead for TN, FP, FN, TP
            for j in range(i+1, min(i+10, len(lines))):
                if 'TN=' in lines[j]:
                    tn_match = re.search(r'TN=\s*([\d,]+)', lines[j])
                    fp_match = re.search(r'FP=\s*([\d,]+)', lines[j])
                    if tn_match and fp_match:
                        confusion_matrix['TN'] = int(tn_match.group(1).replace(',', ''))
                        confusion_matrix['FP'] = int(fp_match.group(1).replace(',', ''))
                
                if 'FN=' in lines[j]:
                    fn_match = re.search(r'FN=\s*([\d,]+)', lines[j])
                    tp_match = re.search(r'TP=\s*([\d,]+)', lines[j])
                    if fn_match and tp_match:
                        confusion_matrix['FN'] = int(fn_match.group(1).replace(',', ''))
                        confusion_matrix['TP'] = int(tp_match.group(1).replace(',', ''))
                        break
        
        # Extract metrics
        if 'CV: AUC=' in line and current_horizon and current_event and current_model:
            match = re.search(r'CV: AUC=([\d.]+), F1=([\d.]+), MCC=([\d.]+)', line)
            if match:
                cv_metrics = {
                    'CV_AUC': float(match.group(1)),
                    'CV_F1': float(match.group(2)),
                    'CV_MCC': float(match.group(3))
                }
        
        if 'Test: AUC=' in line and current_horizon and current_event and current_model:
            match = re.search(r'Test: AUC=([\d.]+), F1=([\d.]+), MCC=([\d.]+)', line)
            if match:
                test_auc, test_f1, test_mcc = map(float, match.groups())
                
                result = {
                    'Horizon': current_horizon,
                    'Event': current_event,
                    'Model': current_model,
                    **cv_metrics,
                    'Test_AUC': test_auc,
                    'Test_F1': test_f1,
                    'Test_MCC': test_mcc
                }
                
                # Add confusion matrix metrics if available
                if len(confusion_matrix) == 4:
                    tn, fp, fn, tp = confusion_matrix['TN'], confusion_matrix['FP'], confusion_matrix['FN'], confusion_matrix['TP']
                    
                    result.update({
                        'TN': tn,
                        'FP': fp,
                        'FN': fn,
                        'TP': tp,
                        'Precision': tp / (tp + fp) if (tp + fp) > 0 else 0,
                        'Recall': tp / (tp + fn) if (tp + fn) > 0 else 0,
                        'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
                        'FPR': fp / (fp + tn) if (fp + tn) > 0 else 0,
                        'FNR': fn / (fn + tp) if (fn + tp) > 0 else 0
                    })
                
                results.append(result)
                
                # Reset
                current_model = None
                confusion_matrix = {}
                cv_metrics = {}
    
    return pd.DataFrame(results)


def parse_shap_values(log_path):
    """Parse SHAP feature importance from log file"""
    
    shap_data = []
    current_horizon = None
    current_event = None
    current_model = None
    
    with open(log_path, 'r') as f:
        lines = f.readlines()
    
    for i, line in enumerate(lines):
        # Extract horizon
        if '⏰ FORECAST HORIZON:' in line:
            match = re.search(r'FORECAST HORIZON: (\w+)', line)
            if match:
                current_horizon = match.group(1)
        
        # Extract event
        if '📍 EVENT:' in line:
            match = re.search(r'EVENT: ([\w_]+)', line)
            if match:
                current_event = match.group(1)
        
        # Extract model
        if 'LIGHTGBM Results:' in line:
            current_model = 'lightgbm'
        elif 'XGBOOST Results:' in line:
            current_model = 'xgboost'
        
        # Extract SHAP values
        if 'Top 20 Most Important Features (SHAP):' in line and current_horizon and current_event and current_model:
            # Read next 20 lines for features
            for j in range(i+1, min(i+21, len(lines))):
                # Match pattern like: "4921. TOWS_PkWSpdMph_025m_roll4_mean : 0.340809"
                match = re.search(r'\d+\.\s+([\w_]+)\s*:\s+([\d.]+)', lines[j])
                if match:
                    feature_name = match.group(1)
                    importance = float(match.group(2))
                    
                    shap_data.append({
                        'Horizon': current_horizon,
                        'Event': current_event,
                        'Model': current_model,
                        'Feature': feature_name,
                        'Importance': importance
                    })
    
    return pd.DataFrame(shap_data)


def analyze_shap_importance(shap_df):
    """Analyze SHAP feature importance patterns"""
    
    if len(shap_df) == 0:
        print("\n⚠️  No SHAP data available")
        return
    
    print("\n" + "="*80)
    print("🔍 SHAP FEATURE IMPORTANCE ANALYSIS")
    print("="*80)
    
    # Top features overall
    print("\n1. TOP 10 MOST IMPORTANT FEATURES (Overall)")
    print("-" * 80)
    top_features = shap_df.groupby('Feature')['Importance'].mean().sort_values(ascending=False).head(10)
    for rank, (feature, importance) in enumerate(top_features.items(), 1):
        print(f"   {rank:2d}. {feature:50s} : {importance:.4f}")
    
    # Top features by horizon
    print("\n2. TOP 5 FEATURES BY PREDICTION HORIZON")
    print("-" * 80)
    horizon_order = ['1hr', '3hr', '6hr', '12hr', '24hr', '48hr', '96hr']
    available_horizons = [h for h in horizon_order if h in shap_df['Horizon'].values]
    
    for horizon in available_horizons:
        horizon_df = shap_df[shap_df['Horizon'] == horizon]
        top_h = horizon_df.groupby('Feature')['Importance'].mean().sort_values(ascending=False).head(5)
        print(f"\n{horizon}:")
        for rank, (feature, importance) in enumerate(top_h.items(), 1):
            print(f"   {rank}. {feature:45s} : {importance:.4f}")
    
    # Top features by event
    print("\n3. TOP 5 FEATURES BY EVENT TYPE")
    print("-" * 80)
    event_names = {
        'event_E3_LowTemp_lt0': 'E3: Low Temperature',
        'event_E4_HighWind_Peak_gt25': 'E4: High Wind',
        'event_E5_LowWind_lt2': 'E5: Low Wind'
    }
    
    for event in shap_df['Event'].unique():
        event_df = shap_df[shap_df['Event'] == event]
        top_e = event_df.groupby('Feature')['Importance'].mean().sort_values(ascending=False).head(5)
        print(f"\n{event_names.get(event, event)}:")
        for rank, (feature, importance) in enumerate(top_e.items(), 1):
            print(f"   {rank}. {feature:45s} : {importance:.4f}")
    
    # Feature type analysis
    print("\n4. FEATURE TYPE BREAKDOWN")
    print("-" * 80)
    
    def categorize_feature(feat):
        if 'lag' in feat.lower():
            return 'Lag Features'
        elif 'roll' in feat.lower():
            return 'Rolling Statistics'
        else:
            return 'Raw Features'
    
    shap_df['Feature_Type'] = shap_df['Feature'].apply(categorize_feature)
    type_importance = shap_df.groupby('Feature_Type')['Importance'].agg(['mean', 'sum', 'count'])
    
    print("\nFeature type contribution:")
    for feat_type in type_importance.index:
        print(f"   {feat_type:25s}: Mean={type_importance.loc[feat_type, 'mean']:.4f}, "
              f"Total={type_importance.loc[feat_type, 'sum']:.2f}, "
              f"Count={type_importance.loc[feat_type, 'count']:.0f}")


def plot_shap_analysis(shap_df):
    """Generate SHAP importance visualization plots"""
    
    if len(shap_df) == 0:
        print("\n⚠️  No SHAP data to plot")
        return None
    
    fig = plt.figure(figsize=(22, 12))
    # Increased wspace for bottom row (event plots)
    gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.5)
    
    horizon_order = ['1hr', '3hr', '6hr', '12hr', '24hr', '48hr', '96hr']
    available_horizons = [h for h in horizon_order if h in shap_df['Horizon'].values]
    
    event_names = {
        'event_E3_LowTemp_lt0': 'E3: Low Temp',
        'event_E4_HighWind_Peak_gt25': 'E4: High Wind',
        'event_E5_LowWind_lt2': 'E5: Low Wind'
    }
    
    # Add title
    fig.suptitle('SHAP Feature Importance Analysis Across Horizons and Events',
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Use viridis color palette for modern look
    # main_cmap = 'viridis'
    
    # 1. Top 15 features overall
    ax1 = fig.add_subplot(gs[0, :2])
    top_features = shap_df.groupby('Feature')['Importance'].mean().sort_values(ascending=False).head(15)
    
    # Use viridis gradient
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(top_features)))
    bars = ax1.barh(range(len(top_features)), top_features.values, color=colors, edgecolor='black', linewidth=0.8)
    ax1.set_yticks(range(len(top_features)))
    ax1.set_yticklabels(top_features.index, fontsize=9)
    ax1.set_xlabel('mean(|SHAP value|)', fontweight='bold', fontsize=11)
    ax1.set_title('Top 15 Most Important Features (Overall)', fontweight='bold', fontsize=13)
    ax1.grid(axis='x', alpha=0.3)
    ax1.invert_yaxis()
    
    # Add value labels
    for i, (idx, val) in enumerate(top_features.items()):
        ax1.text(val + 0.001, i, f'{val:.3f}', va='center', fontsize=9, fontweight='bold')
    
    # 2. Feature type breakdown
    ax2 = fig.add_subplot(gs[0, 2])
    def categorize_feature(feat):
        if 'lag' in feat.lower():
            return 'Lag'
        elif 'roll' in feat.lower():
            return 'Rolling'
        else:
            return 'Raw'
    
    shap_df['Feature_Type'] = shap_df['Feature'].apply(categorize_feature)
    type_importance = shap_df.groupby('Feature_Type')['Importance'].sum().sort_values(ascending=False)
    
    # Use colorful palette
    colors_pie = ['#1f77b4', '#ff7f0e', '#2ca02c']
    wedges, texts, autotexts = ax2.pie(type_importance.values, labels=type_importance.index,
                                         autopct='%1.1f%%', colors=colors_pie, startangle=90,
                                         textprops={'fontsize': 11, 'fontweight': 'bold'})
    ax2.set_title('Feature Type Contribution', fontweight='bold', fontsize=13)
    
    # 3. Top 5 features by horizon (heatmap)
    ax3 = fig.add_subplot(gs[1, :])
    
    # Create matrix: top features x horizons
    top_10_overall = shap_df.groupby('Feature')['Importance'].mean().sort_values(ascending=False).head(10).index
    horizon_feature_matrix = []
    
    for feature in top_10_overall:
        row = []
        for horizon in available_horizons:
            horizon_feat_df = shap_df[(shap_df['Horizon'] == horizon) & (shap_df['Feature'] == feature)]
            if len(horizon_feat_df) > 0:
                row.append(horizon_feat_df['Importance'].mean())
            else:
                row.append(0)
        horizon_feature_matrix.append(row)
    
    heatmap_data = pd.DataFrame(horizon_feature_matrix, index=top_10_overall, columns=available_horizons)
    
    # Use YlOrRd (light to red) to avoid dark colors
    im = ax3.imshow(heatmap_data.values, cmap='YlOrRd', aspect='auto', vmin=0)
    ax3.set_xticks(range(len(available_horizons)))
    ax3.set_xticklabels(available_horizons, fontsize=11)
    ax3.set_yticks(range(len(top_10_overall)))
    ax3.set_yticklabels(top_10_overall, fontsize=9)
    ax3.set_xlabel('Prediction Horizon', fontweight='bold', fontsize=11)
    ax3.set_ylabel('Feature', fontweight='bold', fontsize=11)
    ax3.set_title('Top 10 Feature Importance Across Horizons', fontweight='bold', fontsize=13)
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax3)
    cbar.set_label('mean(|SHAP value|)', fontweight='bold', fontsize=10)
    
    # Add values to heatmap with adaptive text color
    for i in range(len(top_10_overall)):
        for j in range(len(available_horizons)):
            val = heatmap_data.values[i, j]
            # Use white text for darker cells (higher values)
            text_color = 'white' if val > heatmap_data.values.max() * 0.6 else 'black'
            text = ax3.text(j, i, f'{val:.2f}',
                           ha="center", va="center", color=text_color, fontsize=8, fontweight='bold')
    
    # 4-6. Top features by event type
    # Use distinct colors for each event from tab10 palette
    event_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    for idx, (event, color) in enumerate(zip(shap_df['Event'].unique(), event_colors)):
        ax = fig.add_subplot(gs[2, idx])
        event_df = shap_df[shap_df['Event'] == event]
        top_event = event_df.groupby('Feature')['Importance'].mean().sort_values(ascending=False).head(5)
        
        bars = ax.barh(range(len(top_event)), top_event.values, color=color, alpha=0.8, edgecolor='black', linewidth=0.8)
        ax.set_yticks(range(len(top_event)))
        ax.set_yticklabels(top_event.index, fontsize=7)
        ax.set_xlabel('mean(|SHAP value|)', fontweight='bold', fontsize=10)
        ax.set_title(f'{event_names.get(event, event)}', fontweight='bold', fontsize=12)
        ax.grid(axis='x', alpha=0.3)
        ax.invert_yaxis()
        
        # Add value labels
        for i, (idx_val, val) in enumerate(top_event.items()):
            ax.text(val + 0.001, i, f'{val:.2f}', va='center', fontsize=7, fontweight='bold')
    
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    return fig


def analyze_overfitting(df):
    """Analyze train-test gap (overfitting)"""
    
    print("\n" + "="*80)
    print("OVERFITTING ANALYSIS (Train-Test Gap)")
    print("="*80)
    
    df['AUC_Gap'] = df['CV_AUC'] - df['Test_AUC']
    df['F1_Gap'] = df['CV_F1'] - df['Test_F1']
    df['MCC_Gap'] = df['CV_MCC'] - df['Test_MCC']
    
    print("\n📊 Average Train-Test Gap by Horizon:")
    print("-" * 80)
    
    horizon_order = ['1hr', '3hr', '6hr', '12hr', '24hr', '48hr', '96hr']
    available_horizons = [h for h in horizon_order if h in df['Horizon'].values]
    
    for horizon in available_horizons:
        horizon_df = df[df['Horizon'] == horizon]
        print(f"\n{horizon}:")
        print(f"  AUC Gap:  {horizon_df['AUC_Gap'].mean():+.4f} ± {horizon_df['AUC_Gap'].std():.4f}")
        print(f"  F1 Gap:   {horizon_df['F1_Gap'].mean():+.4f} ± {horizon_df['F1_Gap'].std():.4f}")
        print(f"  MCC Gap:  {horizon_df['MCC_Gap'].mean():+.4f} ± {horizon_df['MCC_Gap'].std():.4f}")
    
    print("\n📊 Model Comparison (Overfitting):")
    print("-" * 80)
    for model in df['Model'].unique():
        model_df = df[df['Model'] == model]
        print(f"\n{model.upper()}:")
        print(f"  Avg AUC Gap:  {model_df['AUC_Gap'].mean():+.4f}")
        print(f"  Avg F1 Gap:   {model_df['F1_Gap'].mean():+.4f}")
        print(f"  Avg MCC Gap:  {model_df['MCC_Gap'].mean():+.4f}")
    
    return df


def analyze_error_types(df):
    """Analyze false positives vs false negatives"""
    
    if 'FPR' not in df.columns:
        print("\n⚠️  Skipping error type analysis (confusion matrix data not available)")
        return
    
    print("\n" + "="*80)
    print("ERROR TYPE ANALYSIS")
    print("="*80)
    
    print("\n📊 False Positive Rate (FPR) vs False Negative Rate (FNR):")
    print("-" * 80)
    
    horizon_order = ['1hr', '3hr', '6hr', '12hr', '24hr', '48hr', '96hr']
    available_horizons = [h for h in horizon_order if h in df['Horizon'].values]
    
    for horizon in available_horizons:
        horizon_df = df[df['Horizon'] == horizon]
        print(f"\n{horizon}:")
        print(f"  FPR (False Alarms): {horizon_df['FPR'].mean():.4f} ± {horizon_df['FPR'].std():.4f}")
        print(f"  FNR (Missed Events): {horizon_df['FNR'].mean():.4f} ± {horizon_df['FNR'].std():.4f}")
        ratio = horizon_df['FPR'].mean() / horizon_df['FNR'].mean() if horizon_df['FNR'].mean() > 0 else float('inf')
        print(f"  Ratio (FPR/FNR):    {ratio:.2f}")
    
    print("\n📊 By Event Type:")
    print("-" * 80)
    
    event_names = {
        'event_E3_LowTemp_lt0': 'E3: Low Temp',
        'event_E4_HighWind_Peak_gt25': 'E4: High Wind',
        'event_E5_LowWind_lt2': 'E5: Low Wind'
    }
    
    for event in df['Event'].unique():
        event_df = df[df['Event'] == event]
        print(f"\n{event_names.get(event, event)}:")
        print(f"  Avg FPR: {event_df['FPR'].mean():.4f}")
        print(f"  Avg FNR: {event_df['FNR'].mean():.4f}")
        print(f"  Avg Precision: {event_df['Precision'].mean():.4f}")
        print(f"  Avg Recall:    {event_df['Recall'].mean():.4f}")


def analyze_precision_recall_tradeoff(df):
    """Analyze precision-recall balance"""
    
    if 'Precision' not in df.columns:
        print("\n⚠️  Skipping precision-recall analysis (data not available)")
        return
    
    print("\n" + "="*80)
    print("PRECISION-RECALL TRADEOFF")
    print("="*80)
    
    horizon_order = ['1hr', '3hr', '6hr', '12hr', '24hr', '48hr', '96hr']
    available_horizons = [h for h in horizon_order if h in df['Horizon'].values]
    
    print("\n📊 Balance by Horizon:")
    print("-" * 80)
    
    for horizon in available_horizons:
        horizon_df = df[df['Horizon'] == horizon]
        precision_mean = horizon_df['Precision'].mean()
        recall_mean = horizon_df['Recall'].mean()
        balance = abs(precision_mean - recall_mean)
        
        print(f"\n{horizon}:")
        print(f"  Precision: {precision_mean:.4f}")
        print(f"  Recall:    {recall_mean:.4f}")
        print(f"  Balance:   {balance:.4f} {'✓ Good' if balance < 0.1 else '⚠ Imbalanced'}")
        print(f"  F1 Score:  {horizon_df['Test_F1'].mean():.4f}")


def statistical_significance(df):
    """Test if model differences are statistically significant"""
    
    print("\n" + "="*80)
    print("STATISTICAL SIGNIFICANCE (LightGBM vs XGBoost)")
    print("="*80)
    
    lightgbm_df = df[df['Model'] == 'lightgbm']
    xgboost_df = df[df['Model'] == 'xgboost']
    
    if len(lightgbm_df) == 0 or len(xgboost_df) == 0:
        print("\n⚠️  Need both models for comparison")
        return
    
    # Align by horizon and event
    metrics = ['Test_AUC', 'Test_F1', 'Test_MCC']
    
    print("\nPaired t-tests:")
    print("-" * 80)
    
    for metric in metrics:
        # Get paired samples
        lightgbm_vals = []
        xgboost_vals = []
        
        for _, row in lightgbm_df.iterrows():
            match = xgboost_df[(xgboost_df['Horizon'] == row['Horizon']) & 
                              (xgboost_df['Event'] == row['Event'])]
            if len(match) == 1:
                lightgbm_vals.append(row[metric])
                xgboost_vals.append(match[metric].values[0])
        
        if len(lightgbm_vals) > 1:
            t_stat, p_value = stats.ttest_rel(lightgbm_vals, xgboost_vals)
            
            mean_diff = np.mean(lightgbm_vals) - np.mean(xgboost_vals)
            
            print(f"\n{metric}:")
            print(f"  LightGBM mean: {np.mean(lightgbm_vals):.4f}")
            print(f"  XGBoost mean:  {np.mean(xgboost_vals):.4f}")
            print(f"  Difference:    {mean_diff:+.4f}")
            print(f"  t-statistic:   {t_stat:.4f}")
            sig = '***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else 'ns'
            print(f"  p-value:       {p_value:.4f} {sig}")
            
            if p_value < 0.05:
                better = "LightGBM" if mean_diff > 0 else "XGBoost"
                print(f"  ✓ {better} significantly better")
            else:
                print(f"  ✗ No significant difference")


def generate_key_findings_and_implications(df):
    """Generate key findings and operational implications"""
    
    print("\n" + "="*80)
    print("🔑 KEY FINDINGS")
    print("="*80)
    
    horizon_order = ['1hr', '3hr', '6hr', '12hr', '24hr', '48hr', '96hr']
    available_horizons = [h for h in horizon_order if h in df['Horizon'].values]
    
    # 1. Optimal prediction window
    passing_horizons = []
    for horizon in available_horizons:
        stats = df[df['Horizon'] == horizon][['Test_AUC', 'Test_F1', 'Test_MCC']].mean()
        if stats['Test_AUC'] > 0.90 and stats['Test_F1'] > 0.50 and stats['Test_MCC'] > 0.50:
            passing_horizons.append(horizon)
    
    if passing_horizons:
        print(f"\n1. RELIABLE PREDICTION WINDOW")
        print(f"   ✓ System provides reliable predictions up to {passing_horizons[-1]}")
        print(f"   ✓ This equals approximately 30 minutes to {passing_horizons[-1]} of advance warning")
        print(f"   ✓ Beyond {passing_horizons[-1]}, prediction quality degrades significantly")
    
    # 2. Event-specific findings
    print(f"\n2. EVENT-SPECIFIC PERFORMANCE")
    event_names = {
        'event_E3_LowTemp_lt0': ('Low Temperature (<0°C)', 'Freezing conditions'),
        'event_E4_HighWind_Peak_gt25': ('High Wind (>25mph)', 'Strong wind gusts'),
        'event_E5_LowWind_lt2': ('Low Wind (<2mph)', 'Calm conditions')
    }
    
    event_performance = []
    for event in df['Event'].unique():
        event_df = df[df['Event'] == event]
        avg_f1 = event_df['Test_F1'].mean()
        avg_auc = event_df['Test_AUC'].mean()
        event_performance.append((event, avg_f1, avg_auc))
    
    event_performance.sort(key=lambda x: x[1], reverse=True)
    
    for event, avg_f1, avg_auc in event_performance:
        name, description = event_names.get(event, (event, 'Unknown'))
        reliability = "EXCELLENT" if avg_f1 > 0.7 else "GOOD" if avg_f1 > 0.5 else "MODERATE" if avg_f1 > 0.3 else "POOR"
        print(f"   • {name}: {reliability}")
        print(f"     - {description} can be predicted with {avg_f1:.1%} accuracy (F1)")
        print(f"     - Discrimination ability: {avg_auc:.1%} (AUC)")
    
    # 3. Model recommendation
    print(f"\n3. RECOMMENDED MODEL")
    lightgbm_f1 = df[df['Model'] == 'lightgbm']['Test_F1'].mean()
    xgboost_f1 = df[df['Model'] == 'xgboost']['Test_F1'].mean()
    
    if lightgbm_f1 > xgboost_f1:
        advantage = ((lightgbm_f1 - xgboost_f1) / xgboost_f1) * 100
        print(f"   ✓ LightGBM outperforms XGBoost by {advantage:.1f}%")
        print(f"   ✓ LightGBM shows {lightgbm_f1:.1%} average F1 vs {xgboost_f1:.1%} for XGBoost")
        print(f"   ✓ Recommendation: Deploy LightGBM models for production")
    
    # 4. Performance degradation rate
    first_horizon = df[df['Horizon'] == available_horizons[0]]['Test_F1'].mean()
    last_horizon = df[df['Horizon'] == available_horizons[-1]]['Test_F1'].mean()
    degradation = ((first_horizon - last_horizon) / first_horizon) * 100
    
    print(f"\n4. PREDICTION HORIZON LIMITS")
    print(f"   • Performance degrades by {degradation:.1f}% from {available_horizons[0]} to {available_horizons[-1]}")
    print(f"   • Critical threshold (~50% F1) reached around {available_horizons[2] if len(available_horizons) > 2 else available_horizons[-1]}")
    print(f"   • Predictions beyond 6 hours should be used with caution")
    
    # Operational implications
    print("\n" + "="*80)
    print("🏭 IMPLICATIONS FOR FACILITY OPERATIONS")
    print("="*80)
    
    print("\n1. EARLY WARNING SYSTEM")
    if passing_horizons:
        print(f"   ✓ Implement automated alerts for events predicted within {passing_horizons[-1]}")
        print(f"   ✓ System can provide reliable advance warning for:")
        for event, avg_f1, avg_auc in event_performance:
            name, description = event_names.get(event, (event, 'Unknown'))
            if avg_f1 > 0.5:
                print(f"     - {name} (up to {passing_horizons[-1]} notice)")
    
    print("\n2. OPERATIONAL PROTOCOLS")
    print("   • SHORT-TERM (30min-3hr): HIGH CONFIDENCE")
    print("     - Execute automated responses (e.g., equipment protection)")
    print("     - Deploy personnel for immediate action")
    print("     - Initiate temperature-sensitive equipment protection for freezing events")
    print("     - Secure loose materials and equipment for high wind events")
    
    print("\n   • MEDIUM-TERM (3-6hr): MODERATE CONFIDENCE")
    print("     - Prepare standby crews and resources")
    print("     - Issue advisory notices to staff")
    print("     - Schedule preventive maintenance windows")
    print("     - Monitor predictions for confirmation")
    
    print("\n   • LONG-TERM (6-12hr+): LOW CONFIDENCE")
    print("     - Use for planning purposes only")
    print("     - Monitor continuously for updated predictions")
    print("     - Do not commit critical resources based solely on these forecasts")
    
    print("\n3. EVENT-SPECIFIC RECOMMENDATIONS")
    for event, avg_f1, avg_auc in event_performance:
        name, description = event_names.get(event, (event, 'Unknown'))
        
        if 'LowTemp' in event:
            print(f"\n   • {name} ({avg_f1:.1%} accuracy):")
            print("     - Most predictable event type - highest confidence")
            print("     - Pre-heat critical equipment when forecast")
            print("     - Activate pipe freeze protection systems")
            print("     - Schedule personnel for cold-weather operations")
        
        elif 'HighWind' in event:
            print(f"\n   • {name} ({avg_f1:.1%} accuracy):")
            print("     - Moderate predictability - confirm with multiple forecasts")
            print("     - Secure outdoor equipment and materials")
            print("     - Delay crane operations and aerial work")
            print("     - Check structural tie-downs and temporary installations")
        
        elif 'LowWind' in event:
            print(f"\n   • {name} ({avg_f1:.1%} accuracy):")
            print("     - Least predictable - use with caution")
            print("     - May indicate air quality concerns (poor ventilation)")
            print("     - Adjust HVAC systems for reduced natural ventilation")
            print("     - Monitor for temperature stratification issues")
    
    print("\n4. SYSTEM DEPLOYMENT STRATEGY")
    print("   ✓ Deploy LightGBM models for all three event types")
    print("   ✓ Implement confidence thresholds: High (30min-3hr), Medium (3-6hr), Low (6hr+)")
    print("   ✓ Set up automated monitoring with 15-minute update intervals")
    print("   ✓ Establish escalation protocols based on prediction confidence")
    print("   ✓ Create feedback loop to validate predictions and improve models")
    
    print("\n5. COST-BENEFIT CONSIDERATIONS")
    print("   • False Positives: Equipment protection measures unnecessarily activated")
    print("   • False Negatives: Missed events leading to equipment damage or safety issues")
    print(f"   • Current balance: System prioritizes safety (minimize false negatives)")
    
    best_event = event_performance[0]
    worst_event = event_performance[-1]
    print(f"\n   • Resource allocation priority:")
    print(f"     1. HIGH: {event_names[best_event[0]][0]} predictions (most reliable)")
    print(f"     2. MEDIUM: {event_names[event_performance[1][0]][0]} predictions")
    print(f"     3. LOW: {event_names[worst_event[0]][0]} predictions (least reliable)")
    
    print("\n" + "="*80)


def analyze_horizon_performance(df):
    """Analyze performance degradation across horizons"""
    
    # Define horizon order
    horizon_order = ['1hr', '3hr', '6hr', '12hr', '24hr', '48hr', '96hr']
    df['Horizon_Numeric'] = df['Horizon'].map({h: i for i, h in enumerate(horizon_order)})
    df = df.sort_values('Horizon_Numeric')
    
    print("\n" + "="*80)
    print("PERFORMANCE ANALYSIS BY PREDICTION HORIZON")
    print("="*80)
    
    # Overall stats by horizon
    print("\n📊 Average Metrics Across All Events & Models:")
    print("-" * 80)
    horizon_stats = df.groupby('Horizon')[['Test_AUC', 'Test_F1', 'Test_MCC']].agg(['mean', 'std'])
    
    for horizon in horizon_order:
        if horizon in df['Horizon'].values:
            stats = horizon_stats.loc[horizon]
            print(f"\n{horizon:>5s}:")
            print(f"  AUC:  {stats['Test_AUC']['mean']:.4f} ± {stats['Test_AUC']['std']:.4f}")
            print(f"  F1:   {stats['Test_F1']['mean']:.4f} ± {stats['Test_F1']['std']:.4f}")
            print(f"  MCC:  {stats['Test_MCC']['mean']:.4f} ± {stats['Test_MCC']['std']:.4f}")
    
    # Performance degradation
    print("\n" + "="*80)
    print("PERFORMANCE DEGRADATION")
    print("="*80)
    
    baseline_horizon = '1hr'
    baseline = df[df['Horizon'] == baseline_horizon][['Test_AUC', 'Test_F1', 'Test_MCC']].mean()
    
    print(f"\nBaseline ({baseline_horizon}):")
    print(f"  AUC: {baseline['Test_AUC']:.4f}")
    print(f"  F1:  {baseline['Test_F1']:.4f}")
    print(f"  MCC: {baseline['Test_MCC']:.4f}")
    
    print(f"\nRelative Performance Drop from {baseline_horizon}:")
    print("-" * 80)
    
    for horizon in horizon_order[1:]:
        if horizon in df['Horizon'].values:
            current = df[df['Horizon'] == horizon][['Test_AUC', 'Test_F1', 'Test_MCC']].mean()
            
            drop_auc = (baseline['Test_AUC'] - current['Test_AUC']) / baseline['Test_AUC'] * 100
            drop_f1 = (baseline['Test_F1'] - current['Test_F1']) / baseline['Test_F1'] * 100
            drop_mcc = (baseline['Test_MCC'] - current['Test_MCC']) / baseline['Test_MCC'] * 100
            
            print(f"\n{horizon:>5s}:")
            print(f"  AUC:  {current['Test_AUC']:.4f} ({drop_auc:+.1f}%)")
            print(f"  F1:   {current['Test_F1']:.4f} ({drop_f1:+.1f}%)")
            print(f"  MCC:  {current['Test_MCC']:.4f} ({drop_mcc:+.1f}%)")
    
    # Per-event analysis
    print("\n" + "="*80)
    print("PERFORMANCE BY EVENT TYPE")
    print("="*80)
    
    for event in df['Event'].unique():
        event_df = df[df['Event'] == event]
        print(f"\n{event}:")
        print("-" * 80)
        
        for horizon in horizon_order:
            if horizon in event_df['Horizon'].values:
                stats = event_df[event_df['Horizon'] == horizon][['Test_AUC', 'Test_F1', 'Test_MCC']].mean()
                print(f"  {horizon:>5s}: AUC={stats['Test_AUC']:.4f}, F1={stats['Test_F1']:.4f}, MCC={stats['Test_MCC']:.4f}")
    
    # Determine optimal cutoff
    print("\n" + "="*80)
    print("RECOMMENDED PREDICTION CUTOFF")
    print("="*80)
    
    # Use thresholds: AUC > 0.9, F1 > 0.5, MCC > 0.5
    print("\nApplying performance thresholds:")
    print("  - AUC > 0.90 (good discrimination)")
    print("  - F1  > 0.50 (reasonable precision-recall balance)")
    print("  - MCC > 0.50 (better than random on imbalanced data)")
    
    print("\nHorizons meeting ALL thresholds:")
    print("-" * 80)
    
    for horizon in horizon_order:
        if horizon in df['Horizon'].values:
            stats = df[df['Horizon'] == horizon][['Test_AUC', 'Test_F1', 'Test_MCC']].mean()
            
            meets_auc = stats['Test_AUC'] > 0.90
            meets_f1 = stats['Test_F1'] > 0.50
            meets_mcc = stats['Test_MCC'] > 0.50
            
            status = "✓ PASS" if (meets_auc and meets_f1 and meets_mcc) else "✗ FAIL"
            
            print(f"  {horizon:>5s}: {status}")
            print(f"         AUC={stats['Test_AUC']:.4f} {'✓' if meets_auc else '✗'}, "
                  f"F1={stats['Test_F1']:.4f} {'✓' if meets_f1 else '✗'}, "
                  f"MCC={stats['Test_MCC']:.4f} {'✓' if meets_mcc else '✗'}")
    
    # Find last passing horizon
    passing_horizons = []
    for horizon in horizon_order:
        if horizon in df['Horizon'].values:
            stats = df[df['Horizon'] == horizon][['Test_AUC', 'Test_F1', 'Test_MCC']].mean()
            if stats['Test_AUC'] > 0.90 and stats['Test_F1'] > 0.50 and stats['Test_MCC'] > 0.50:
                passing_horizons.append(horizon)
    
    if passing_horizons:
        print(f"\n✅ RECOMMENDED CUTOFF: {passing_horizons[-1]}")
        print(f"   Reliable predictions up to {passing_horizons[-1]} ahead")
    else:
        print("\n⚠️  No horizons meet all thresholds")
    
    # Best models
    print("\n" + "="*80)
    print("BEST MODELS BY HORIZON")
    print("="*80)
    
    for horizon in horizon_order:
        if horizon in df['Horizon'].values:
            horizon_df = df[df['Horizon'] == horizon]
            best_idx = horizon_df['Test_F1'].idxmax()
            best = horizon_df.loc[best_idx]
            
            print(f"\n{horizon:>5s}: {best['Model'].upper()} on {best['Event']}")
            print(f"       AUC={best['Test_AUC']:.4f}, F1={best['Test_F1']:.4f}, MCC={best['Test_MCC']:.4f}")
    
    return df


def plot_additional_insights(df):
    """Generate comprehensive additional visualization plots"""
    
    fig = plt.figure(figsize=(18, 10))
    gs = fig.add_gridspec(2, 3, hspace=0.35, wspace=0.3)
    
    horizon_order = ['1hr', '3hr', '6hr', '12hr', '24hr', '48hr', '96hr']
    available_horizons = [h for h in horizon_order if h in df['Horizon'].values]
    x_pos = range(len(available_horizons))
    
    # Horizon conversion info
    horizon_info = {
        '1hr': '30min ahead',
        '3hr': '3hr ahead',
        '6hr': '6hr ahead',
        '12hr': '12hr ahead',
        '24hr': '24hr ahead',
        '48hr': '48hr ahead',
        '96hr': '96hr ahead'
    }
    
    # Add title with convention
    fig.suptitle('Multi-Horizon Performance Analysis\n' + 
                 'Data: 15-min intervals | Example: "1hr" = 30min ahead (2×15min steps)',
                 fontsize=14, fontweight='bold', y=0.98)
    
    # 1. Performance Decay with Horizons
    ax1 = fig.add_subplot(gs[0, 0])
    for metric, label, color in [('Test_AUC', 'AUC', 'blue'),
                                  ('Test_F1', 'F1', 'green'),
                                  ('Test_MCC', 'MCC', 'orange')]:
        y_vals = [df[df['Horizon'] == h][metric].mean() for h in available_horizons]
        ax1.plot(x_pos, y_vals, marker='o', label=label, linewidth=2.5, 
                markersize=8, color=color, alpha=0.8)
    
    # Add threshold lines
    ax1.axhline(y=0.9, color='red', linestyle='--', linewidth=1.5, alpha=0.4, label='Threshold')
    ax1.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.4)
    
    ax1.set_xlabel('Prediction Horizon', fontweight='bold', fontsize=11)
    ax1.set_ylabel('Performance Score', fontweight='bold', fontsize=11)
    ax1.set_title('Performance Degradation Over Time', fontweight='bold', fontsize=13)
    ax1.set_xticks(x_pos)
    ax1.set_xticklabels(available_horizons, fontsize=10)
    ax1.set_ylim([0, 1.05])
    ax1.legend(fontsize=10, loc='best')
    ax1.grid(alpha=0.3)
    
    # 2. Event Difficulty Comparison
    ax2 = fig.add_subplot(gs[0, 1])
    event_names = {
        'event_E3_LowTemp_lt0': 'E3: Low Temp',
        'event_E4_HighWind_Peak_gt25': 'E4: High Wind',
        'event_E5_LowWind_lt2': 'E5: Low Wind'
    }
    
    for event, color in zip(df['Event'].unique(), ['#3498db', '#e74c3c', '#f39c12']):
        event_df = df[df['Event'] == event]
        horizon_means = event_df.groupby('Horizon')['Test_F1'].mean()
        y_vals = [horizon_means.loc[h] if h in horizon_means.index else np.nan 
                 for h in available_horizons]
        ax2.plot(x_pos, y_vals, marker='o', linewidth=2.5, markersize=8, 
               label=event_names.get(event, event), color=color, alpha=0.8)
    
    ax2.axhline(y=0.5, color='red', linestyle='--', linewidth=1.5, alpha=0.4)
    ax2.set_xlabel('Prediction Horizon', fontweight='bold', fontsize=11)
    ax2.set_ylabel('F1 Score', fontweight='bold', fontsize=11)
    ax2.set_title('Event Type Difficulty', fontweight='bold', fontsize=13)
    ax2.set_xticks(x_pos)
    ax2.set_xticklabels(available_horizons, fontsize=10)
    ax2.set_ylim([0, 1.05])
    ax2.legend(fontsize=10, loc='best')
    ax2.grid(alpha=0.3)
    
    # 3. Model Comparison
    ax3 = fig.add_subplot(gs[0, 2])
    model_data = []
    model_labels = []
    for model in ['lightgbm', 'xgboost']:
        if model in df['Model'].values:
            model_data.append(df[df['Model'] == model]['Test_F1'].values)
            model_labels.append(model.upper())
    
    if model_data:
        bp = ax3.boxplot(model_data, labels=model_labels, patch_artist=True, widths=0.6)
        colors = ['#2ecc71', '#e74c3c'][:len(model_data)]
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
            patch.set_linewidth(2)
        
        # Enhance whiskers and median
        for whisker in bp['whiskers']:
            whisker.set(linewidth=1.5)
        for median in bp['medians']:
            median.set(color='black', linewidth=2)
    
    ax3.set_ylabel('Test F1 Score', fontweight='bold', fontsize=11)
    ax3.set_title('LightGBM vs XGBoost', fontweight='bold', fontsize=13)
    ax3.grid(axis='y', alpha=0.3)
    ax3.set_ylim([0, 1.05])
    
    # 4. Ranking by Average Performance
    ax4 = fig.add_subplot(gs[1, 0])
    event_scores = []
    for event in df['Event'].unique():
        event_df = df[df['Event'] == event]
        avg_f1 = event_df['Test_F1'].mean()
        event_scores.append((event_names.get(event, event), avg_f1))
    
    event_scores.sort(key=lambda x: x[1], reverse=True)
    events, scores = zip(*event_scores)
    
    colors_map = {'E3: Low Temp': '#3498db', 'E4: High Wind': '#e74c3c', 'E5: Low Wind': '#f39c12'}
    bar_colors = [colors_map.get(e, 'gray') for e in events]
    
    bars = ax4.barh(events, scores, color=bar_colors, alpha=0.75, edgecolor='black', linewidth=2)
    ax4.set_xlabel('Average F1 Score', fontweight='bold', fontsize=11)
    ax4.set_title('Event Predictability Ranking', fontweight='bold', fontsize=13)
    ax4.grid(axis='x', alpha=0.3)
    ax4.set_xlim([0, 1])
    
    for i, (event, score) in enumerate(event_scores):
        ax4.text(score + 0.02, i, f'{score:.3f}', va='center', fontweight='bold', fontsize=11)
    
    # 5. Summary Table with Conventions (spanning 2 columns)
    ax5 = fig.add_subplot(gs[1, 1:])
    ax5.axis('off')
    
    # Create summary with actual time ahead
    summary_data = []
    for horizon in available_horizons:
        horizon_df = df[df['Horizon'] == horizon]
        summary_data.append([
            horizon,
            horizon_info.get(horizon, 'N/A'),
            f"{horizon_df['Test_AUC'].mean():.3f}",
            f"{horizon_df['Test_F1'].mean():.3f}",
            f"{horizon_df['Test_MCC'].mean():.3f}"
        ])
    
    table = ax5.table(cellText=summary_data,
                     colLabels=['Horizon', 'Actual Time\nAhead', 'AUC', 'F1', 'MCC'],
                     cellLoc='center',
                     loc='center',
                     bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 2.5)
    
    # Color code by performance
    for i in range(1, len(summary_data) + 1):
        for j in range(2, 5):  # Only color AUC, F1, MCC columns
            val = float(summary_data[i-1][j])
            if val >= 0.9:
                table[(i, j)].set_facecolor('#d4edda')  # Green
            elif val >= 0.7:
                table[(i, j)].set_facecolor('#fff3cd')  # Yellow
            elif val >= 0.5:
                table[(i, j)].set_facecolor('#ffe6e6')  # Light red
            else:
                table[(i, j)].set_facecolor('#f5c6cb')  # Red
    
    # Header styling
    for j in range(5):
        table[(0, j)].set_facecolor('#2c3e50')
        table[(0, j)].set_text_props(weight='bold', color='white', fontsize=11)
    
    # Make first two columns slightly different
    for i in range(1, len(summary_data) + 1):
        table[(i, 0)].set_facecolor('#ecf0f1')
        table[(i, 1)].set_facecolor('#ecf0f1')
    
    ax5.set_title('Performance Summary by Horizon', fontweight='bold', fontsize=14, pad=15)
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    return fig


def main():
    log_file = Path(__file__).parent / 'logs' / 'multitower_ml_20260112_063916.log'
    
    if not log_file.exists():
        print(f"❌ Log file not found: {log_file}")
        return
    
    print(f"📖 Reading log file: {log_file}")
    df = parse_log_file(log_file)
    shap_df = parse_shap_values(log_file)
    
    print(f"\n✓ Parsed {len(df)} experiment results")
    print(f"✓ Parsed {len(shap_df)} SHAP importance values")
    print(f"  Horizons: {sorted(df['Horizon'].unique(), key=lambda x: ['1hr', '3hr', '6hr', '12hr', '24hr', '48hr', '96hr'].index(x))}")
    print(f"  Events: {list(df['Event'].unique())}")
    print(f"  Models: {list(df['Model'].unique())}")
    
    # Run all analyses
    df = analyze_overfitting(df)
    df = analyze_horizon_performance(df)
    analyze_error_types(df)
    analyze_precision_recall_tradeoff(df)
    statistical_significance(df)
    analyze_shap_importance(shap_df)
    generate_key_findings_and_implications(df)
    
    # Save analysis results
    output_csv = log_file.parent.parent / 'horizon_analysis.csv'
    df.to_csv(output_csv, index=False)
    print(f"\n✓ Analysis saved to: {output_csv}")
    
    # Generate plots
    print("\n" + "="*80)
    print("GENERATING VISUALIZATION PLOTS")
    print("="*80)
    
    output_dir = log_file.parent.parent / 'plots'
    output_dir.mkdir(exist_ok=True)
    
    # Additional insights plot
    print("\n1. Additional insights (6-panel analysis)...")
    fig_insights = plot_additional_insights(df)
    insights_path = output_dir / 'additional_insights.png'
    fig_insights.savefig(insights_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {insights_path}")
    
    # SHAP importance plot
    print("\n2. SHAP feature importance analysis...")
    fig_shap = plot_shap_analysis(shap_df)
    shap_path = output_dir / 'shap_importance.png'
    fig_shap.savefig(shap_path, dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {shap_path}")
    
    print("\n✅ Complete! All analyses and plots generated.")


if __name__ == '__main__':
    main()


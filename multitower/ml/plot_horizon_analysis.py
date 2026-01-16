#!/usr/bin/env python3
"""
Visualize multi-horizon prediction performance
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (16, 10)
plt.rcParams['font.size'] = 10

def load_data():
    """Load horizon analysis results"""
    csv_path = Path(__file__).parent / 'horizon_analysis.csv'
    df = pd.read_csv(csv_path)
    
    # Define horizon order
    horizon_order = ['1hr', '3hr', '6hr', '12hr', '24hr', '48hr', '96hr']
    df['Horizon_Numeric'] = df['Horizon'].map({h: i for i, h in enumerate(horizon_order)})
    df = df.sort_values('Horizon_Numeric')
    
    return df, horizon_order

def plot_performance_degradation(df, horizon_order):
    """Plot average performance across horizons"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Aggregate by horizon
    horizon_stats = df.groupby('Horizon')[['Test_AUC', 'Test_F1', 'Test_MCC']].agg(['mean', 'std'])
    
    # Filter to available horizons
    available_horizons = [h for h in horizon_order if h in df['Horizon'].values]
    x_pos = range(len(available_horizons))
    
    metrics = [
        ('Test_AUC', 'AUC-ROC', 'Blues'),
        ('Test_F1', 'F1 Score', 'Greens'),
        ('Test_MCC', 'MCC', 'Oranges')
    ]
    
    for ax, (metric, title, color) in zip(axes, metrics):
        means = [horizon_stats.loc[h, (metric, 'mean')] for h in available_horizons]
        stds = [horizon_stats.loc[h, (metric, 'std')] for h in available_horizons]
        
        # Bar plot with error bars
        bars = ax.bar(x_pos, means, yerr=stds, capsize=5, 
                     color=sns.color_palette(color, len(available_horizons)),
                     edgecolor='black', linewidth=1.5, alpha=0.8)
        
        # Add threshold lines
        if metric == 'Test_AUC':
            ax.axhline(y=0.9, color='red', linestyle='--', linewidth=2, label='Threshold (0.90)')
        else:
            ax.axhline(y=0.5, color='red', linestyle='--', linewidth=2, label='Threshold (0.50)')
        
        # Styling
        ax.set_xlabel('Prediction Horizon', fontsize=12, fontweight='bold')
        ax.set_ylabel(title, fontsize=12, fontweight='bold')
        ax.set_title(f'{title} by Prediction Horizon', fontsize=14, fontweight='bold')
        ax.set_xticks(x_pos)
        ax.set_xticklabels(available_horizons, fontsize=11)
        ax.set_ylim([0, 1.05])
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
        
        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, means)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                   f'{val:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    return fig

def plot_event_comparison(df, horizon_order):
    """Plot performance by event type"""
    
    fig, axes = plt.subplots(3, 3, figsize=(18, 14))
    
    available_horizons = [h for h in horizon_order if h in df['Horizon'].values]
    events = df['Event'].unique()
    
    # Clean event names
    event_names = {
        'event_E3_LowTemp_lt0': 'E3: Low Temp (<0°C)',
        'event_E4_HighWind_Peak_gt25': 'E4: High Wind (>25mph)',
        'event_E5_LowWind_lt2': 'E5: Low Wind (<2mph)'
    }
    
    colors = ['#e74c3c', '#3498db', '#2ecc71']
    
    for row_idx, metric in enumerate(['Test_AUC', 'Test_F1', 'Test_MCC']):
        metric_name = {'Test_AUC': 'AUC-ROC', 'Test_F1': 'F1 Score', 'Test_MCC': 'MCC'}[metric]
        
        # Combined plot (column 0)
        ax = axes[row_idx, 0]
        for event, color in zip(events, colors):
            event_df = df[df['Event'] == event]
            horizon_means = event_df.groupby('Horizon')[metric].mean()
            
            y_vals = [horizon_means.loc[h] if h in horizon_means.index else np.nan 
                     for h in available_horizons]
            
            ax.plot(range(len(available_horizons)), y_vals, 
                   marker='o', linewidth=2.5, markersize=8, 
                   label=event_names[event], color=color, alpha=0.8)
        
        # Add threshold
        threshold = 0.9 if metric == 'Test_AUC' else 0.5
        ax.axhline(y=threshold, color='red', linestyle='--', linewidth=2, alpha=0.5)
        
        ax.set_xlabel('Prediction Horizon', fontsize=11, fontweight='bold')
        ax.set_ylabel(metric_name, fontsize=11, fontweight='bold')
        ax.set_title(f'{metric_name} - All Events', fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(available_horizons)))
        ax.set_xticklabels(available_horizons)
        ax.set_ylim([0, 1.05])
        ax.legend(fontsize=9, loc='best')
        ax.grid(alpha=0.3)
        
        # Individual event plots (columns 1, 2)
        for col_idx, (event, color) in enumerate(zip(events[:2], colors[:2]), start=1):
            ax = axes[row_idx, col_idx]
            event_df = df[df['Event'] == event]
            
            # Get both models
            for model in ['lightgbm', 'xgboost']:
                model_df = event_df[event_df['Model'] == model]
                horizon_vals = model_df.groupby('Horizon')[metric].mean()
                
                y_vals = [horizon_vals.loc[h] if h in horizon_vals.index else np.nan 
                         for h in available_horizons]
                
                marker = 'o' if model == 'lightgbm' else 's'
                ax.plot(range(len(available_horizons)), y_vals,
                       marker=marker, linewidth=2, markersize=7,
                       label=model.upper(), alpha=0.8)
            
            # Add threshold
            ax.axhline(y=threshold, color='red', linestyle='--', linewidth=1.5, alpha=0.5)
            
            ax.set_xlabel('Prediction Horizon', fontsize=10, fontweight='bold')
            ax.set_ylabel(metric_name, fontsize=10, fontweight='bold')
            ax.set_title(f'{event_names[event]}', fontsize=11, fontweight='bold')
            ax.set_xticks(range(len(available_horizons)))
            ax.set_xticklabels(available_horizons)
            ax.set_ylim([0, 1.05])
            ax.legend(fontsize=8)
            ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_model_comparison(df, horizon_order):
    """Compare LightGBM vs XGBoost"""
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    available_horizons = [h for h in horizon_order if h in df['Horizon'].values]
    
    metrics = [
        ('Test_AUC', 'AUC-ROC'),
        ('Test_F1', 'F1 Score'),
        ('Test_MCC', 'MCC')
    ]
    
    for ax, (metric, title) in zip(axes, metrics):
        # Calculate means by model and horizon
        model_stats = df.groupby(['Horizon', 'Model'])[metric].mean().reset_index()
        
        for model, color, marker in [('lightgbm', '#2ecc71', 'o'), ('xgboost', '#e74c3c', 's')]:
            model_df = model_stats[model_stats['Model'] == model]
            y_vals = [model_df[model_df['Horizon'] == h][metric].values[0] 
                     if h in model_df['Horizon'].values else np.nan
                     for h in available_horizons]
            
            ax.plot(range(len(available_horizons)), y_vals,
                   marker=marker, linewidth=3, markersize=10,
                   label=model.upper(), color=color, alpha=0.8)
        
        # Add threshold
        threshold = 0.9 if metric == 'Test_AUC' else 0.5
        ax.axhline(y=threshold, color='gray', linestyle='--', linewidth=2, alpha=0.5)
        
        ax.set_xlabel('Prediction Horizon', fontsize=12, fontweight='bold')
        ax.set_ylabel(title, fontsize=12, fontweight='bold')
        ax.set_title(f'{title}: LightGBM vs XGBoost', fontsize=14, fontweight='bold')
        ax.set_xticks(range(len(available_horizons)))
        ax.set_xticklabels(available_horizons, fontsize=11)
        ax.set_ylim([0, 1.05])
        ax.legend(fontsize=11, loc='best')
        ax.grid(alpha=0.3)
    
    plt.tight_layout()
    return fig

def plot_heatmap(df, horizon_order):
    """Create heatmap of F1 scores"""
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    available_horizons = [h for h in horizon_order if h in df['Horizon'].values]
    
    # Clean event names
    event_names = {
        'event_E3_LowTemp_lt0': 'Low Temp',
        'event_E4_HighWind_Peak_gt25': 'High Wind',
        'event_E5_LowWind_lt2': 'Low Wind'
    }
    
    for ax, model in zip(axes, ['lightgbm', 'xgboost']):
        model_df = df[df['Model'] == model]
        
        # Create pivot table
        pivot = model_df.pivot_table(
            values='Test_F1',
            index='Event',
            columns='Horizon',
            aggfunc='mean'
        )
        
        # Reorder
        pivot = pivot[[h for h in available_horizons if h in pivot.columns]]
        pivot.index = [event_names[e] for e in pivot.index]
        
        # Plot heatmap
        sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', 
                   vmin=0, vmax=1, cbar_kws={'label': 'F1 Score'},
                   ax=ax, linewidths=1, linecolor='black')
        
        ax.set_title(f'{model.upper()} - F1 Score Heatmap', fontsize=14, fontweight='bold')
        ax.set_xlabel('Prediction Horizon', fontsize=12, fontweight='bold')
        ax.set_ylabel('Event Type', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    return fig

def main():
    print("📊 Generating horizon performance plots...")
    
    # Load data
    df, horizon_order = load_data()
    print(f"✓ Loaded {len(df)} results")
    
    # Create output directory
    output_dir = Path(__file__).parent / 'plots'
    output_dir.mkdir(exist_ok=True)
    
    # Generate plots
    print("\n1. Performance degradation plot...")
    fig1 = plot_performance_degradation(df, horizon_order)
    fig1.savefig(output_dir / 'horizon_performance_degradation.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {output_dir / 'horizon_performance_degradation.png'}")
    
    print("\n2. Event comparison plot...")
    fig2 = plot_event_comparison(df, horizon_order)
    fig2.savefig(output_dir / 'horizon_event_comparison.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {output_dir / 'horizon_event_comparison.png'}")
    
    print("\n3. Model comparison plot...")
    fig3 = plot_model_comparison(df, horizon_order)
    fig3.savefig(output_dir / 'horizon_model_comparison.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {output_dir / 'horizon_model_comparison.png'}")
    
    print("\n4. Heatmap...")
    fig4 = plot_heatmap(df, horizon_order)
    fig4.savefig(output_dir / 'horizon_heatmap.png', dpi=300, bbox_inches='tight')
    print(f"   ✓ Saved: {output_dir / 'horizon_heatmap.png'}")
    
    print(f"\n✅ All plots saved to: {output_dir}/")
    
    # Show plots
    plt.show()

if __name__ == '__main__':
    main()

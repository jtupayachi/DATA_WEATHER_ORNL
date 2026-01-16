"""
Plotting utilities for multi-tower ML results
Creates loss curves and confusion matrices
"""

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import json
import os
from pathlib import Path

def plot_loss_curves(train_losses, val_losses, model_type, event_name, horizon, output_dir):
    """Plot training and validation loss curves"""
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss', linewidth=2)
    plt.plot(val_losses, label='Validation Loss', linewidth=2)
    plt.xlabel('Iteration', fontsize=12)
    plt.ylabel('Log Loss', fontsize=12)
    plt.title(f'{model_type.upper()} - {event_name} - Horizon: {horizon}', fontsize=14, fontweight='bold')
    plt.legend(fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Save
    filename = f"{output_dir}/loss_curve_{event_name}_{model_type}_{horizon}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"      📊 Loss curve saved: {filename}")

def plot_confusion_matrix(cm, event_name, model_type, horizon, output_dir):
    """Plot confusion matrix heatmap"""
    plt.figure(figsize=(8, 6))
    
    # Normalize by row (actual class)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    # Create annotations with both count and percentage
    annotations = np.empty_like(cm).astype(str)
    tn, fp, fn, tp = cm.ravel()
    annotations[0, 0] = f'TN\n{tn:,}\n({cm_normalized[0,0]:.1%})'
    annotations[0, 1] = f'FP\n{fp:,}\n({cm_normalized[0,1]:.1%})'
    annotations[1, 0] = f'FN\n{fn:,}\n({cm_normalized[1,0]:.1%})'
    annotations[1, 1] = f'TP\n{tp:,}\n({cm_normalized[1,1]:.1%})'
    
    # Plot
    sns.heatmap(cm, annot=annotations, fmt='', cmap='Blues', 
                cbar=True, square=True, linewidths=2,
                xticklabels=['Predicted Negative', 'Predicted Positive'],
                yticklabels=['Actual Negative', 'Actual Positive'])
    
    plt.title(f'{model_type.upper()} - {event_name} - Horizon: {horizon}', 
              fontsize=14, fontweight='bold', pad=20)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    
    # Save
    filename = f"{output_dir}/confusion_matrix_{event_name}_{model_type}_{horizon}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"      📊 Confusion matrix saved: {filename}")

def plot_multi_horizon_comparison(results_dirs, output_dir):
    """
    Compare results across multiple forecast horizons
    
    Args:
        results_dirs: Dict of {horizon: results_dir_path}
        output_dir: Where to save plots
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Load all results
    all_data = {}
    for horizon, results_dir in results_dirs.items():
        config_file = f"{results_dir}/config.json"
        if os.path.exists(config_file):
            with open(config_file, 'r') as f:
                all_data[horizon] = json.load(f)
    
    if not all_data:
        print("No results found to compare")
        return
    
    # Extract metrics for comparison
    # This is a placeholder - implement based on your actual result structure
    print(f"Loaded results for horizons: {list(all_data.keys())}")
    print(f"Comparison plots will be saved to: {output_dir}")

if __name__ == '__main__':
    # Example usage
    print("Loss curve and confusion matrix plotting utilities loaded")
    print("Import these functions into your training script")

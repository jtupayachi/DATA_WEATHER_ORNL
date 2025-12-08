#!/usr/bin/env python3
"""
Feature Importance Visualization Tool
======================================
Visualizes feature importance across towers, events, and variables
from ML experiment results.

Usage:
    python viz_importance.py <results_folder_path>
    
Example:
    python viz_importance.py BESTML_multi_event_results_20251208_155338
"""

import os
import sys
import glob
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from datetime import datetime
from scipy import stats
from scipy.stats import spearmanr
from scipy.interpolate import griddata
from itertools import combinations
from mpl_toolkits.mplot3d import Axes3D

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")

# =============================================================================
# TOWER GEOGRAPHIC COORDINATES AND TERRAIN FEATURES (ORNL Weather Towers)
# =============================================================================
# Placeholder coordinates for each tower - UPDATE WITH ACTUAL VALUES
# Additional geographic/terrain features for analysis
TOWER_COORDINATES = {
    'TOWA': {
        'lat': 35.9312, 'lon': -84.3108, 'elevation_m': 280,
        'name': 'Tower A - Main Campus',
        'terrain_type': 'Valley Floor',
        'slope_deg': 2.5,           # Terrain slope in degrees
        'aspect_deg': 180,          # Direction slope faces (0=N, 90=E, 180=S, 270=W)
        'canopy_cover_pct': 15,     # Vegetation canopy coverage
        'dist_to_ridge_m': 450,     # Distance to nearest ridge
        'soil_type': 'Clay Loam',
        'land_use': 'Research Facility',
    },
    'TOWB': {
        'lat': 35.9285, 'lon': -84.3045, 'elevation_m': 295,
        'name': 'Tower B - East Ridge',
        'terrain_type': 'Ridge Slope',
        'slope_deg': 8.0,
        'aspect_deg': 270,          # West-facing
        'canopy_cover_pct': 45,
        'dist_to_ridge_m': 120,
        'soil_type': 'Sandy Loam',
        'land_use': 'Mixed Forest',
    },
    'TOWD': {
        'lat': 35.9350, 'lon': -84.3200, 'elevation_m': 310,
        'name': 'Tower D - West Valley',
        'terrain_type': 'Valley Slope',
        'slope_deg': 5.5,
        'aspect_deg': 90,           # East-facing
        'canopy_cover_pct': 30,
        'dist_to_ridge_m': 280,
        'soil_type': 'Silt Loam',
        'land_use': 'Grassland',
    },
    'TOWF': {
        'lat': 35.9220, 'lon': -84.3150, 'elevation_m': 265,
        'name': 'Tower F - South Field',
        'terrain_type': 'Valley Floor',
        'slope_deg': 1.5,
        'aspect_deg': 0,            # Flat/North
        'canopy_cover_pct': 5,
        'dist_to_ridge_m': 600,
        'soil_type': 'Clay',
        'land_use': 'Open Field',
    },
    'TOWS': {
        'lat': 35.9380, 'lon': -84.2980, 'elevation_m': 340,
        'name': 'Tower S - North Summit',
        'terrain_type': 'Ridge Top',
        'slope_deg': 3.0,
        'aspect_deg': 45,           # NE-facing
        'canopy_cover_pct': 60,
        'dist_to_ridge_m': 0,       # On the ridge
        'soil_type': 'Rocky Loam',
        'land_use': 'Deciduous Forest',
    },
    'TOWY': {
        'lat': 35.9255, 'lon': -84.3250, 'elevation_m': 275,
        'name': 'Tower Y - Southwest',
        'terrain_type': 'Valley Slope',
        'slope_deg': 6.0,
        'aspect_deg': 135,          # SE-facing
        'canopy_cover_pct': 25,
        'dist_to_ridge_m': 380,
        'soil_type': 'Loam',
        'land_use': 'Shrubland',
    },
}

# Geographic region info
REGION_INFO = {
    'name': 'Oak Ridge National Laboratory (ORNL)',
    'state': 'Tennessee, USA',
    'center_lat': 35.931,
    'center_lon': -84.310,
    'terrain': 'Ridge and Valley Province, Appalachian Region',
    'climate': 'Humid subtropical (Köppen: Cfa)',
    'avg_annual_temp_c': 14.4,
    'avg_annual_precip_mm': 1370,
    'prevailing_wind': 'Southwest',
}


def load_importance_files(folder_path: str) -> dict:
    """Load all importance CSV files from the specified folder."""
    pattern = os.path.join(folder_path, "importance_*.csv")
    files = glob.glob(pattern)
    
    if not files:
        print(f"No importance files found in {folder_path}")
        return {}
    
    data = {}
    for file_path in files:
        filename = os.path.basename(file_path)
        # Parse filename: importance_TOWA_event_E3_LowTemp_lt0.csv
        match = re.match(r'importance_(\w+)_event_(.+)\.csv', filename)
        if match:
            tower = match.group(1)
            event = match.group(2)
            df = pd.read_csv(file_path)
            df['tower'] = tower
            df['event'] = event
            key = f"{tower}_{event}"
            data[key] = df
            print(f"Loaded: {filename} ({len(df)} features)")
    
    return data


def get_combined_dataframe(data: dict) -> pd.DataFrame:
    """Combine all importance dataframes into one."""
    if not data:
        return pd.DataFrame()
    return pd.concat(data.values(), ignore_index=True)


def plot_top_features_by_tower_event(df: pd.DataFrame, tower: str, event: str, 
                                      top_n: int = 20, save_path: str = None):
    """Plot top N features for a specific tower and event."""
    subset = df[(df['tower'] == tower) & (df['event'] == event)].copy()
    
    if subset.empty:
        print(f"No data found for tower={tower}, event={event}")
        return
    
    subset = subset.nsmallest(top_n, 'rank')
    
    fig, ax = plt.subplots(figsize=(12, 8))
    
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(subset)))
    
    bars = ax.barh(range(len(subset)), subset['importance_pct'], color=colors)
    ax.set_yticks(range(len(subset)))
    ax.set_yticklabels(subset['feature'])
    ax.invert_yaxis()
    
    ax.set_xlabel('Importance (%)', fontsize=12)
    ax.set_title(f'Top {top_n} Feature Importance\nTower: {tower} | Event: {event}', 
                 fontsize=14, fontweight='bold')
    
    # Add value labels
    for i, (bar, pct) in enumerate(zip(bars, subset['importance_pct'])):
        ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                f'{pct:.2f}%', va='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    plt.close()


def plot_feature_comparison_across_towers(df: pd.DataFrame, event: str, 
                                          top_n: int = 15, save_path: str = None):
    """Compare top features across all towers for a specific event."""
    subset = df[df['event'] == event].copy()
    
    if subset.empty:
        print(f"No data found for event={event}")
        return
    
    towers = subset['tower'].unique()
    
    # Get top features across all towers for this event
    top_features = (subset.groupby('feature')['importance_pct']
                    .mean()
                    .nlargest(top_n)
                    .index.tolist())
    
    subset_top = subset[subset['feature'].isin(top_features)]
    
    # Pivot for heatmap
    pivot = subset_top.pivot_table(values='importance_pct', 
                                    index='feature', 
                                    columns='tower',
                                    aggfunc='first')
    
    # Sort by mean importance
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
    
    fig, ax = plt.subplots(figsize=(12, 10))
    
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='YlOrRd', 
                ax=ax, cbar_kws={'label': 'Importance (%)'})
    
    ax.set_title(f'Feature Importance Comparison Across Towers\nEvent: {event}', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Tower', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    plt.close()


def plot_all_towers_per_event(df: pd.DataFrame, event: str, top_n: int = 15, save_path: str = None):
    """
    Create a single figure with subplots showing top features for ALL towers 
    for a specific event (target variable).
    """
    subset = df[df['event'] == event].copy()
    
    if subset.empty:
        print(f"No data found for event={event}")
        return
    
    towers = sorted(subset['tower'].unique())
    n_towers = len(towers)
    
    # Calculate grid dimensions
    n_cols = 3
    n_rows = (n_towers + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 6 * n_rows))
    axes = axes.flatten() if n_towers > 1 else [axes]
    
    # Color palette for towers
    tower_colors = plt.cm.Set2(np.linspace(0, 1, n_towers))
    
    for idx, (tower, color) in enumerate(zip(towers, tower_colors)):
        ax = axes[idx]
        tower_data = subset[subset['tower'] == tower].nsmallest(top_n, 'rank')
        
        if tower_data.empty:
            ax.text(0.5, 0.5, f'No data for {tower}', ha='center', va='center')
            ax.set_title(f'{tower}', fontsize=12, fontweight='bold')
            continue
        
        # Create gradient colors based on importance
        colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(tower_data)))
        
        bars = ax.barh(range(len(tower_data)), tower_data['importance_pct'], 
                       color=colors, edgecolor='darkgray', linewidth=0.5)
        ax.set_yticks(range(len(tower_data)))
        ax.set_yticklabels(tower_data['feature'], fontsize=9)
        ax.invert_yaxis()
        
        ax.set_xlabel('Importance (%)', fontsize=10)
        ax.set_title(f'Tower: {tower}', fontsize=12, fontweight='bold', 
                     color='darkblue', pad=10)
        
        # Add value labels
        for bar, pct in zip(bars, tower_data['importance_pct']):
            ax.text(bar.get_width() + 0.2, bar.get_y() + bar.get_height()/2,
                    f'{pct:.1f}%', va='center', fontsize=8, color='black')
        
        ax.set_xlim(0, tower_data['importance_pct'].max() * 1.25)
        ax.grid(axis='x', alpha=0.3)
    
    # Hide empty subplots
    for idx in range(n_towers, len(axes)):
        axes[idx].set_visible(False)
    
    # Add main title
    event_display = event.replace('_', ' ').replace('lt', '<').replace('gt', '>')
    fig.suptitle(f'Top {top_n} Feature Importance by Tower\nTarget Event: {event_display}', 
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    plt.close()


def plot_consolidated_all_events(df: pd.DataFrame, top_n: int = 15, save_path: str = None):
    """
    Create ONE consolidated figure with ALL events, showing ALL towers side by side.
    Each row = one event (target variable)
    Each column = one tower
    """
    towers = sorted(df['tower'].unique())
    events = sorted(df['event'].unique())
    
    n_towers = len(towers)
    n_events = len(events)
    
    fig, axes = plt.subplots(n_events, n_towers, figsize=(4 * n_towers, 5 * n_events))
    
    # Ensure axes is 2D
    if n_events == 1:
        axes = axes.reshape(1, -1)
    if n_towers == 1:
        axes = axes.reshape(-1, 1)
    
    for row_idx, event in enumerate(events):
        for col_idx, tower in enumerate(towers):
            ax = axes[row_idx, col_idx]
            
            subset = df[(df['tower'] == tower) & (df['event'] == event)]
            subset = subset.nsmallest(top_n, 'rank')
            
            if subset.empty:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center')
                continue
            
            # Create gradient colors
            colors = plt.cm.YlOrRd(np.linspace(0.3, 0.9, len(subset)))
            
            bars = ax.barh(range(len(subset)), subset['importance_pct'], 
                           color=colors, edgecolor='gray', linewidth=0.3)
            ax.set_yticks(range(len(subset)))
            ax.set_yticklabels(subset['feature'], fontsize=7)
            ax.invert_yaxis()
            
            # Add value labels
            for bar, pct in zip(bars, subset['importance_pct']):
                ax.text(bar.get_width() + 0.1, bar.get_y() + bar.get_height()/2,
                        f'{pct:.1f}%', va='center', fontsize=6, color='black')
            
            ax.set_xlim(0, subset['importance_pct'].max() * 1.3)
            ax.grid(axis='x', alpha=0.3)
            
            # Add tower name as column header (only first row)
            if row_idx == 0:
                ax.set_title(f'{tower}', fontsize=11, fontweight='bold', color='darkblue')
            
            # Add event name on left side (only first column)
            if col_idx == 0:
                event_display = event.replace('_', '\n').replace('lt', '<').replace('gt', '>')
                ax.set_ylabel(event_display, fontsize=9, fontweight='bold', 
                             color='darkgreen', rotation=0, ha='right', labelpad=60)
    
    fig.suptitle(f'Feature Importance: All Towers × All Events (Top {top_n} Features)', 
                 fontsize=16, fontweight='bold', y=1.01)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    plt.close()


def plot_grouped_bars_per_event(df: pd.DataFrame, event: str, top_n: int = 10, save_path: str = None):
    """
    Create a dot plot (Cleveland dot plot style) showing top features with ALL towers 
    as clustered dots for a specific event.
    """
    subset = df[df['event'] == event].copy()
    
    if subset.empty:
        print(f"No data found for event={event}")
        return
    
    towers = sorted(subset['tower'].unique())
    
    # Get top features (by mean importance across towers)
    top_features = (subset.groupby('feature')['importance_pct']
                    .mean()
                    .nlargest(top_n)
                    .index.tolist())
    
    subset_top = subset[subset['feature'].isin(top_features)]
    
    # Pivot data
    pivot = subset_top.pivot_table(values='importance_pct', 
                                    index='feature', 
                                    columns='tower',
                                    aggfunc='first',
                                    fill_value=0)
    
    # Sort by mean importance
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
    
    # Only use towers that exist in the pivot
    available_towers = [t for t in towers if t in pivot.columns]
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    y_positions = np.arange(len(pivot.index))
    
    # Color palette and markers for towers
    colors = plt.cm.Set1(np.linspace(0, 1, len(available_towers)))
    markers = ['o', 's', '^', 'D', 'v', 'p', 'h', '*']
    
    for i, (tower, color) in enumerate(zip(available_towers, colors)):
        marker = markers[i % len(markers)]
        values = pivot[tower].values
        
        # Plot dots with slight vertical offset for each tower
        offset = (i - len(available_towers)/2 + 0.5) * 0.12
        ax.scatter(values, y_positions + offset, 
                   s=150, c=[color], marker=marker, 
                   label=tower, edgecolors='black', linewidth=0.5,
                   alpha=0.85, zorder=3)
        
        # Add value labels next to dots
        for j, val in enumerate(values):
            if val > 0:
                ax.annotate(f'{val:.1f}', (val, y_positions[j] + offset),
                           xytext=(5, 0), textcoords='offset points',
                           fontsize=7, va='center', color=color, fontweight='bold')
    
    # Add horizontal lines for each feature
    for y in y_positions:
        ax.axhline(y=y, color='lightgray', linestyle='-', linewidth=0.5, zorder=1)
    
    # Add vertical line at x=0
    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.5, zorder=1)
    
    ax.set_yticks(y_positions)
    ax.set_yticklabels(pivot.index, fontsize=10)
    ax.invert_yaxis()
    
    ax.set_xlabel('Importance (%)', fontsize=12)
    event_display = event.replace('_', ' ').replace('lt', '<').replace('gt', '>')
    ax.set_title(f'Feature Importance by Tower (Dot Plot)\nEvent: {event_display}', 
                 fontsize=14, fontweight='bold')
    ax.legend(title='Tower', bbox_to_anchor=(1.02, 1), loc='upper left',
              markerscale=1.2, fontsize=10)
    ax.grid(axis='x', alpha=0.3, zorder=0)
    ax.set_xlim(left=-0.5)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    plt.close()


def get_tower_geo_dataframe(df: pd.DataFrame, event: str, top_n: int = 5) -> pd.DataFrame:
    """
    Create a dataframe with tower geographic features and top N feature importance.
    """
    towers = sorted(df['tower'].unique())
    subset = df[df['event'] == event].copy()
    
    tower_data = []
    for tower in towers:
        if tower not in TOWER_COORDINATES:
            continue
        
        tower_df = subset[subset['tower'] == tower]
        coords = TOWER_COORDINATES[tower]
        
        # Get top N features
        top_features = tower_df.nsmallest(top_n, 'rank')
        top_importance_sum = top_features['importance_pct'].sum()
        top_feature_1 = top_features.iloc[0]['feature'] if len(top_features) > 0 else 'N/A'
        top_feature_1_imp = top_features.iloc[0]['importance_pct'] if len(top_features) > 0 else 0
        
        # Calculate distance from center
        lat_diff = coords['lat'] - REGION_INFO['center_lat']
        lon_diff = coords['lon'] - REGION_INFO['center_lon']
        dist_from_center_km = np.sqrt((lat_diff * 111)**2 + (lon_diff * 111 * np.cos(np.radians(coords['lat'])))**2)
        
        tower_data.append({
            'tower': tower,
            'name': coords['name'],
            'lat': coords['lat'],
            'lon': coords['lon'],
            'elevation_m': coords['elevation_m'],
            'terrain_type': coords['terrain_type'],
            'slope_deg': coords['slope_deg'],
            'aspect_deg': coords['aspect_deg'],
            'canopy_cover_pct': coords['canopy_cover_pct'],
            'dist_to_ridge_m': coords['dist_to_ridge_m'],
            'dist_from_center_km': dist_from_center_km,
            'soil_type': coords['soil_type'],
            'land_use': coords['land_use'],
            'top5_importance': top_importance_sum,
            'top1_feature': top_feature_1,
            'top1_importance': top_feature_1_imp,
        })
    
    return pd.DataFrame(tower_data)


def plot_geographic_features_vs_importance(df: pd.DataFrame, output_folder: str = None):
    """
    Create comprehensive geographic analysis plots - ONE IMAGE PER EVENT.
    Shows how geographic/terrain features relate to top 5 feature importance.
    Generates 3 images (one per target variable/event).
    """
    events = sorted(df['event'].unique())
    
    for event in events:
        event_display = event.replace('_', ' ').replace('lt', '<').replace('gt', '>')
        
        # Get tower geographic data for this event
        geo_df = get_tower_geo_dataframe(df, event, top_n=5)
        
        if geo_df.empty:
            print(f"No data for event: {event}")
            continue
        
        # Create figure with 3x2 subplots
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        fig.suptitle(f'Geographic Features vs Feature Importance\nTarget: {event_display}\n'
                     f'{REGION_INFO["name"]}, {REGION_INFO["state"]}',
                     fontsize=16, fontweight='bold', y=1.02)
        
        # Color palette for towers
        tower_colors = dict(zip(geo_df['tower'], plt.cm.Set2(np.linspace(0, 1, len(geo_df)))))
        
        # =====================================================================
        # Plot 1: Elevation vs Top 5 Importance
        # =====================================================================
        ax1 = axes[0, 0]
        for _, row in geo_df.iterrows():
            ax1.scatter(row['elevation_m'], row['top5_importance'], 
                       s=200, c=[tower_colors[row['tower']]], 
                       edgecolors='black', linewidth=1.5, zorder=3)
            ax1.annotate(row['tower'], (row['elevation_m'], row['top5_importance']),
                        xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')
        
        # Add trend line
        z = np.polyfit(geo_df['elevation_m'], geo_df['top5_importance'], 1)
        p = np.poly1d(z)
        x_line = np.linspace(geo_df['elevation_m'].min()-10, geo_df['elevation_m'].max()+10, 100)
        ax1.plot(x_line, p(x_line), 'r--', alpha=0.7, linewidth=2, label='Trend')
        
        # Correlation
        corr, pval = spearmanr(geo_df['elevation_m'], geo_df['top5_importance'])
        ax1.text(0.05, 0.95, f'r={corr:.3f}\np={pval:.3f}', transform=ax1.transAxes,
                fontsize=10, verticalalignment='top', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax1.set_xlabel('Elevation (m)', fontsize=11)
        ax1.set_ylabel('Top 5 Features Importance (%)', fontsize=11)
        ax1.set_title('Elevation vs Importance', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # =====================================================================
        # Plot 2: Slope vs Top 5 Importance
        # =====================================================================
        ax2 = axes[0, 1]
        for _, row in geo_df.iterrows():
            ax2.scatter(row['slope_deg'], row['top5_importance'],
                       s=200, c=[tower_colors[row['tower']]],
                       edgecolors='black', linewidth=1.5, zorder=3)
            ax2.annotate(row['tower'], (row['slope_deg'], row['top5_importance']),
                        xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')
        
        corr, pval = spearmanr(geo_df['slope_deg'], geo_df['top5_importance'])
        ax2.text(0.05, 0.95, f'r={corr:.3f}\np={pval:.3f}', transform=ax2.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax2.set_xlabel('Terrain Slope (degrees)', fontsize=11)
        ax2.set_ylabel('Top 5 Features Importance (%)', fontsize=11)
        ax2.set_title('Slope vs Importance', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # =====================================================================
        # Plot 3: Canopy Cover vs Top 5 Importance
        # =====================================================================
        ax3 = axes[0, 2]
        for _, row in geo_df.iterrows():
            ax3.scatter(row['canopy_cover_pct'], row['top5_importance'],
                       s=200, c=[tower_colors[row['tower']]],
                       edgecolors='black', linewidth=1.5, zorder=3)
            ax3.annotate(row['tower'], (row['canopy_cover_pct'], row['top5_importance']),
                        xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')
        
        corr, pval = spearmanr(geo_df['canopy_cover_pct'], geo_df['top5_importance'])
        ax3.text(0.05, 0.95, f'r={corr:.3f}\np={pval:.3f}', transform=ax3.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax3.set_xlabel('Canopy Cover (%)', fontsize=11)
        ax3.set_ylabel('Top 5 Features Importance (%)', fontsize=11)
        ax3.set_title('Vegetation Cover vs Importance', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # =====================================================================
        # Plot 4: Distance to Ridge vs Top 5 Importance
        # =====================================================================
        ax4 = axes[1, 0]
        for _, row in geo_df.iterrows():
            ax4.scatter(row['dist_to_ridge_m'], row['top5_importance'],
                       s=200, c=[tower_colors[row['tower']]],
                       edgecolors='black', linewidth=1.5, zorder=3)
            ax4.annotate(row['tower'], (row['dist_to_ridge_m'], row['top5_importance']),
                        xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')
        
        corr, pval = spearmanr(geo_df['dist_to_ridge_m'], geo_df['top5_importance'])
        ax4.text(0.05, 0.95, f'r={corr:.3f}\np={pval:.3f}', transform=ax4.transAxes,
                fontsize=10, verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
        
        ax4.set_xlabel('Distance to Ridge (m)', fontsize=11)
        ax4.set_ylabel('Top 5 Features Importance (%)', fontsize=11)
        ax4.set_title('Ridge Proximity vs Importance', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # =====================================================================
        # Plot 5: Terrain Type (categorical) vs Top 5 Importance - Box/Bar
        # =====================================================================
        ax5 = axes[1, 1]
        terrain_order = ['Valley Floor', 'Valley Slope', 'Ridge Slope', 'Ridge Top']
        terrain_data = geo_df.groupby('terrain_type')['top5_importance'].mean().reindex(terrain_order).dropna()
        
        bars = ax5.bar(range(len(terrain_data)), terrain_data.values, 
                      color=plt.cm.terrain(np.linspace(0.2, 0.8, len(terrain_data))),
                      edgecolor='black', linewidth=1.5)
        ax5.set_xticks(range(len(terrain_data)))
        ax5.set_xticklabels(terrain_data.index, rotation=30, ha='right', fontsize=10)
        
        # Add tower labels on bars
        for terrain in terrain_data.index:
            towers_in_terrain = geo_df[geo_df['terrain_type'] == terrain]['tower'].tolist()
            idx = list(terrain_data.index).index(terrain)
            ax5.text(idx, terrain_data[terrain] + 1, ', '.join(towers_in_terrain),
                    ha='center', fontsize=9, fontweight='bold')
        
        ax5.set_xlabel('Terrain Type', fontsize=11)
        ax5.set_ylabel('Mean Top 5 Importance (%)', fontsize=11)
        ax5.set_title('Terrain Type vs Importance', fontsize=12, fontweight='bold')
        ax5.grid(axis='y', alpha=0.3)
        
        # =====================================================================
        # Plot 6: Geographic Map with Importance as color/size
        # =====================================================================
        ax6 = axes[1, 2]
        
        # Size by importance, color by elevation
        sizes = (geo_df['top5_importance'] / geo_df['top5_importance'].max()) * 800 + 100
        scatter = ax6.scatter(geo_df['lon'], geo_df['lat'], 
                             s=sizes, c=geo_df['elevation_m'],
                             cmap='terrain', edgecolors='black', linewidth=2, alpha=0.8)
        
        # Add tower labels
        for _, row in geo_df.iterrows():
            ax6.annotate(f"{row['tower']}\n{row['top5_importance']:.1f}%",
                        (row['lon'], row['lat']),
                        xytext=(8, 8), textcoords='offset points',
                        fontsize=9, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
        
        cbar = plt.colorbar(scatter, ax=ax6, shrink=0.8)
        cbar.set_label('Elevation (m)', fontsize=10)
        
        ax6.set_xlabel('Longitude (°W)', fontsize=11)
        ax6.set_ylabel('Latitude (°N)', fontsize=11)
        ax6.set_title('Geographic Distribution\n(size=importance, color=elevation)', 
                     fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        
        # =====================================================================
        # Add summary table as text
        # =====================================================================
        summary_text = "TOWER SUMMARY:\n" + "-"*50 + "\n"
        for _, row in geo_df.sort_values('top5_importance', ascending=False).iterrows():
            summary_text += f"{row['tower']}: {row['top5_importance']:.1f}% | {row['terrain_type']} | {row['elevation_m']}m\n"
            summary_text += f"       Top feature: {row['top1_feature'][:30]}... ({row['top1_importance']:.1f}%)\n"
        
        fig.text(0.02, 0.01, summary_text, fontsize=8, fontfamily='monospace',
                verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
        
        plt.tight_layout()
        plt.subplots_adjust(bottom=0.15)
        
        if output_folder:
            save_path = os.path.join(output_folder, f'geo_features_vs_importance_{event}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.show()
        plt.close()


def plot_3d_terrain_importance(df: pd.DataFrame, event: str, output_folder: str = None):
    """
    Create a 3D terrain visualization showing elevation profile with importance markers.
    """
    event_display = event.replace('_', ' ').replace('lt', '<').replace('gt', '>')
    geo_df = get_tower_geo_dataframe(df, event, top_n=5)
    
    if geo_df.empty:
        print(f"No data for event: {event}")
        return
    
    # Create 3D figure
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Convert lat/lon to km from center
    lat_center = REGION_INFO['center_lat']
    lon_center = REGION_INFO['center_lon']
    geo_df['x_km'] = (geo_df['lon'] - lon_center) * 111 * np.cos(np.radians(lat_center))
    geo_df['y_km'] = (geo_df['lat'] - lat_center) * 111
    
    # Create terrain mesh
    x_range = np.linspace(geo_df['x_km'].min() - 0.5, geo_df['x_km'].max() + 0.5, 30)
    y_range = np.linspace(geo_df['y_km'].min() - 0.5, geo_df['y_km'].max() + 0.5, 30)
    X, Y = np.meshgrid(x_range, y_range)
    
    # Interpolate terrain elevation
    from scipy.interpolate import griddata
    points = geo_df[['x_km', 'y_km']].values
    values = geo_df['elevation_m'].values
    Z = griddata(points, values, (X, Y), method='cubic', fill_value=np.mean(values))
    
    # Plot terrain surface
    surf = ax.plot_surface(X, Y, Z, cmap='terrain', alpha=0.6, linewidth=0, antialiased=True)
    
    # Plot towers as vertical bars from terrain to importance height
    for idx, row in geo_df.iterrows():
        # Find terrain height at tower location
        terrain_height = row['elevation_m']
        importance_height = terrain_height + row['top5_importance'] * 2  # Scale for visibility
        
        # Plot vertical line (tower)
        ax.plot([row['x_km'], row['x_km']], 
                [row['y_km'], row['y_km']], 
                [terrain_height, importance_height],
                color='red', linewidth=4, alpha=0.9)
        
        # Plot importance sphere at top
        ax.scatter([row['x_km']], [row['y_km']], [importance_height],
                  s=row['top5_importance']*10 + 100, c='red', 
                  edgecolors='darkred', linewidth=2, alpha=0.8, zorder=5)
        
        # Tower label
        ax.text(row['x_km'], row['y_km'], importance_height + 10,
               f"{row['tower']}\n{row['top5_importance']:.1f}%",
               ha='center', va='bottom', fontsize=10, fontweight='bold')
        
        # Base marker
        ax.scatter([row['x_km']], [row['y_km']], [terrain_height],
                  s=100, c='brown', marker='^', edgecolors='black', zorder=5)
    
    # Labels
    ax.set_xlabel('West ← Distance (km) → East', fontsize=11, labelpad=10)
    ax.set_ylabel('South ← Distance (km) → North', fontsize=11, labelpad=10)
    ax.set_zlabel('Elevation (m) + Scaled Importance', fontsize=11, labelpad=10)
    
    ax.set_title(f'3D Terrain Profile with Feature Importance\nTarget: {event_display}\n'
                f'{REGION_INFO["name"]} - {REGION_INFO["terrain"]}',
                fontsize=14, fontweight='bold')
    
    # Add legend/info
    info_text = (f"Red bars: Top 5 Feature Importance\n"
                f"Terrain: Elevation interpolated from tower data\n"
                f"Climate: {REGION_INFO['climate']}")
    ax.text2D(0.02, 0.98, info_text, transform=ax.transAxes, fontsize=9,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    ax.view_init(elev=25, azim=45)
    
    plt.tight_layout()
    
    if output_folder:
        save_path = os.path.join(output_folder, f'geo_3d_terrain_{event}.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    plt.close()


def generate_all_geographic_visualizations(df: pd.DataFrame, output_folder: str):
    """
    Generate all geographic visualizations - 3 images per event type.
    """
    events = sorted(df['event'].unique())
    
    print("\n" + "="*60)
    print("Generating Geographic Feature Analysis (3 images per event)")
    print("="*60)
    
    # 1. Generate 2D scatter plots (geographic features vs importance)
    print("\n>> Generating 2D Geographic Feature Analysis plots...")
    plot_geographic_features_vs_importance(df, output_folder)
    
    # 2. Generate 3D terrain visualizations
    print("\n>> Generating 3D Terrain visualizations...")
    for event in events:
        plot_3d_terrain_importance(df, event, output_folder)
    
    print(f"\n>> All geographic visualizations saved to: {output_folder}")
    print(f"   Generated {len(events)} 2D feature plots + {len(events)} 3D terrain plots")


def plot_feature_comparison_across_events(df: pd.DataFrame, tower: str, 
                                          top_n: int = 15, save_path: str = None):
    """Compare top features across all events for a specific tower."""
    subset = df[df['tower'] == tower].copy()
    
    if subset.empty:
        print(f"No data found for tower={tower}")
        return
    
    events = subset['event'].unique()
    
    # Get top features across all events for this tower
    top_features = (subset.groupby('feature')['importance_pct']
                    .mean()
                    .nlargest(top_n)
                    .index.tolist())
    
    subset_top = subset[subset['feature'].isin(top_features)]
    
    # Pivot for heatmap
    pivot = subset_top.pivot_table(values='importance_pct', 
                                    index='feature', 
                                    columns='event',
                                    aggfunc='first')
    
    # Sort by mean importance
    pivot = pivot.loc[pivot.mean(axis=1).sort_values(ascending=False).index]
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    sns.heatmap(pivot, annot=True, fmt='.2f', cmap='YlOrRd', 
                ax=ax, cbar_kws={'label': 'Importance (%)'})
    
    ax.set_title(f'Feature Importance Comparison Across Events\nTower: {tower}', 
                 fontsize=14, fontweight='bold')
    ax.set_xlabel('Event', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    plt.close()


def plot_global_feature_importance(df: pd.DataFrame, top_n: int = 25, save_path: str = None):
    """Plot global feature importance aggregated across all towers and events."""
    
    # Calculate mean importance across all tower-event combinations
    global_importance = (df.groupby('feature')['importance_pct']
                         .agg(['mean', 'std', 'count'])
                         .reset_index())
    global_importance.columns = ['feature', 'mean_importance', 'std_importance', 'count']
    global_importance = global_importance.nlargest(top_n, 'mean_importance')
    
    fig, ax = plt.subplots(figsize=(14, 10))
    
    colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(global_importance)))
    
    bars = ax.barh(range(len(global_importance)), 
                   global_importance['mean_importance'], 
                   xerr=global_importance['std_importance'],
                   color=colors, capsize=3)
    
    ax.set_yticks(range(len(global_importance)))
    ax.set_yticklabels(global_importance['feature'])
    ax.invert_yaxis()
    
    ax.set_xlabel('Mean Importance (%) ± Std', fontsize=12)
    ax.set_title(f'Top {top_n} Global Feature Importance\n(Averaged Across All Towers & Events)', 
                 fontsize=14, fontweight='bold')
    
    # Add value labels
    for i, (bar, mean, std) in enumerate(zip(bars, 
                                              global_importance['mean_importance'],
                                              global_importance['std_importance'])):
        ax.text(bar.get_width() + global_importance['std_importance'].max() * 0.1, 
                bar.get_y() + bar.get_height()/2,
                f'{mean:.2f}%', va='center', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    plt.close()


def plot_variable_category_importance(df: pd.DataFrame, save_path: str = None):
    """Plot importance by variable category (TempC, WSpdMph, RelHum, etc.)."""
    
    # Extract base variable category from feature names
    def get_category(feature):
        # Remove lag, roll, etc. suffixes and extract base variable
        base = re.split(r'_\d+m|_lag|_roll', feature)[0]
        return base
    
    df_cat = df.copy()
    df_cat['category'] = df_cat['feature'].apply(get_category)
    
    # Aggregate by category
    cat_importance = (df_cat.groupby(['category', 'event'])['importance_pct']
                      .sum()
                      .reset_index())
    
    # Pivot for grouped bar chart
    pivot = cat_importance.pivot_table(values='importance_pct', 
                                        index='category', 
                                        columns='event',
                                        aggfunc='sum',
                                        fill_value=0)
    
    # Sort by total importance
    pivot = pivot.loc[pivot.sum(axis=1).sort_values(ascending=False).head(15).index]
    
    fig, ax = plt.subplots(figsize=(14, 8))
    
    pivot.plot(kind='bar', ax=ax, width=0.8)
    
    ax.set_xlabel('Variable Category', fontsize=12)
    ax.set_ylabel('Total Importance (%)', fontsize=12)
    ax.set_title('Feature Category Importance by Event Type', 
                 fontsize=14, fontweight='bold')
    ax.legend(title='Event', bbox_to_anchor=(1.02, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    plt.close()


def create_summary_dashboard(df: pd.DataFrame, output_folder: str = None):
    """Create a comprehensive summary dashboard."""
    
    towers = sorted(df['tower'].unique())
    events = sorted(df['event'].unique())
    
    fig = plt.figure(figsize=(20, 16))
    
    # 1. Global top features (top-left)
    ax1 = fig.add_subplot(2, 2, 1)
    global_imp = (df.groupby('feature')['importance_pct']
                  .mean()
                  .nlargest(15)
                  .sort_values())
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(global_imp)))
    ax1.barh(range(len(global_imp)), global_imp.values, color=colors)
    ax1.set_yticks(range(len(global_imp)))
    ax1.set_yticklabels(global_imp.index)
    ax1.set_xlabel('Mean Importance (%)')
    ax1.set_title('Top 15 Global Features', fontweight='bold')
    
    # 2. Importance by tower (top-right)
    ax2 = fig.add_subplot(2, 2, 2)
    tower_imp = (df.groupby('tower')['importance_pct']
                 .sum()
                 .sort_values(ascending=True))
    colors = plt.cm.Set2(np.linspace(0, 1, len(tower_imp)))
    ax2.barh(tower_imp.index, tower_imp.values, color=colors)
    ax2.set_xlabel('Total Importance (%)')
    ax2.set_title('Total Feature Importance by Tower', fontweight='bold')
    
    # 3. Importance distribution by event (bottom-left)
    ax3 = fig.add_subplot(2, 2, 3)
    event_data = []
    for event in events:
        event_df = df[df['event'] == event]['importance_pct']
        event_data.append(event_df)
    
    bp = ax3.boxplot(event_data, tick_labels=events, patch_artist=True)
    colors = plt.cm.tab10(np.linspace(0, 1, len(events)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
    ax3.set_xlabel('Event')
    ax3.set_ylabel('Importance (%)')
    ax3.set_title('Importance Distribution by Event', fontweight='bold')
    plt.setp(ax3.xaxis.get_majorticklabels(), rotation=45, ha='right')
    
    # 4. Top 5 features per event (bottom-right)
    ax4 = fig.add_subplot(2, 2, 4)
    summary_text = "TOP 5 FEATURES PER EVENT\n" + "="*40 + "\n\n"
    for event in events:
        event_df = df[df['event'] == event]
        top5 = event_df.nsmallest(5, 'rank')[['feature', 'importance_pct']]
        summary_text += f">> {event}:\n"
        for _, row in top5.iterrows():
            summary_text += f"   • {row['feature']}: {row['importance_pct']:.2f}%\n"
        summary_text += "\n"
    
    ax4.text(0.05, 0.95, summary_text, transform=ax4.transAxes, 
             fontsize=9, verticalalignment='top', fontfamily='monospace',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax4.axis('off')
    ax4.set_title('Top Features Summary', fontweight='bold')
    
    plt.suptitle('Feature Importance Dashboard', fontsize=16, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if output_folder:
        save_path = os.path.join(output_folder, 'importance_dashboard.png')
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    plt.close()


def generate_statistical_report(df: pd.DataFrame, output_folder: str = None):
    """
    Generate a comprehensive text report with statistical highlights and significance tests.
    """
    towers = sorted(df['tower'].unique())
    events = sorted(df['event'].unique())
    
    report_lines = []
    
    # Header
    report_lines.append("=" * 80)
    report_lines.append("       FEATURE IMPORTANCE STATISTICAL REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report_lines.append(f"Towers analyzed: {', '.join(towers)}")
    report_lines.append(f"Events analyzed: {', '.join(events)}")
    report_lines.append(f"Total feature records: {len(df)}")
    report_lines.append("")
    
    # =========================================================================
    # GEOGRAPHIC INFORMATION
    # =========================================================================
    report_lines.append("=" * 80)
    report_lines.append("GEOGRAPHIC CONTEXT")
    report_lines.append("=" * 80)
    report_lines.append(f"\nStudy Region: {REGION_INFO['name']}")
    report_lines.append(f"Location: {REGION_INFO['state']}")
    report_lines.append(f"Center Coordinates: {REGION_INFO['center_lat']:.4f}N, {abs(REGION_INFO['center_lon']):.4f}W")
    report_lines.append(f"Terrain: {REGION_INFO['terrain']}")
    report_lines.append(f"Climate: {REGION_INFO['climate']}")
    report_lines.append("")
    report_lines.append("Tower Locations:")
    report_lines.append("-" * 80)
    report_lines.append(f"{'Tower':<8} {'Name':<30} {'Latitude':<12} {'Longitude':<12} {'Elevation':<12}")
    report_lines.append("-" * 80)
    for tower in towers:
        if tower in TOWER_COORDINATES:
            coords = TOWER_COORDINATES[tower]
            report_lines.append(f"{tower:<8} {coords['name']:<30} {coords['lat']:<12.4f} {coords['lon']:<12.4f} {coords['elevation_m']:<10}m")
    
    # Calculate spatial spread
    lats = [TOWER_COORDINATES[t]['lat'] for t in towers if t in TOWER_COORDINATES]
    lons = [TOWER_COORDINATES[t]['lon'] for t in towers if t in TOWER_COORDINATES]
    elevs = [TOWER_COORDINATES[t]['elevation_m'] for t in towers if t in TOWER_COORDINATES]
    
    if lats and lons:
        lat_spread_km = (max(lats) - min(lats)) * 111  # approx km per degree
        lon_spread_km = (max(lons) - min(lons)) * 111 * np.cos(np.radians(np.mean(lats)))
        report_lines.append("")
        report_lines.append(f"Spatial Extent: ~{lat_spread_km:.2f} km (N-S) x ~{abs(lon_spread_km):.2f} km (E-W)")
        report_lines.append(f"Elevation Range: {min(elevs)}m - {max(elevs)}m ({max(elevs)-min(elevs)}m difference)")
    report_lines.append("")
    
    # =========================================================================
    # SECTION 1: GLOBAL TOP FEATURES
    # =========================================================================
    report_lines.append("=" * 80)
    report_lines.append("SECTION 1: GLOBAL TOP FEATURES (Averaged Across All Towers & Events)")
    report_lines.append("=" * 80)
    
    global_importance = (df.groupby('feature')['importance_pct']
                         .agg(['mean', 'std', 'min', 'max', 'count'])
                         .reset_index())
    global_importance.columns = ['feature', 'mean', 'std', 'min', 'max', 'count']
    global_importance = global_importance.sort_values('mean', ascending=False)
    
    report_lines.append("\nTop 20 Most Important Features (by mean importance %):")
    report_lines.append("-" * 80)
    report_lines.append(f"{'Rank':<5} {'Feature':<40} {'Mean%':<10} {'Std%':<10} {'Min%':<10} {'Max%':<10}")
    report_lines.append("-" * 80)
    
    for rank, (_, row) in enumerate(global_importance.head(20).iterrows(), 1):
        report_lines.append(f"{rank:<5} {row['feature']:<40} {row['mean']:>8.3f}  {row['std']:>8.3f}  {row['min']:>8.3f}  {row['max']:>8.3f}")
    
    # =========================================================================
    # SECTION 2: TOP FEATURES PER EVENT
    # =========================================================================
    report_lines.append("\n")
    report_lines.append("=" * 80)
    report_lines.append("SECTION 2: TOP FEATURES PER EVENT (Target Variable)")
    report_lines.append("=" * 80)
    
    for event in events:
        event_display = event.replace('_', ' ').replace('lt', '<').replace('gt', '>')
        report_lines.append(f"\n>> EVENT: {event_display}")
        report_lines.append("-" * 60)
        
        event_df = df[df['event'] == event]
        event_importance = (event_df.groupby('feature')['importance_pct']
                           .agg(['mean', 'std'])
                           .reset_index())
        event_importance.columns = ['feature', 'mean', 'std']
        event_importance = event_importance.sort_values('mean', ascending=False).head(10)
        
        report_lines.append(f"{'Rank':<5} {'Feature':<45} {'Mean%':<12} {'Std%':<12}")
        report_lines.append("-" * 60)
        
        for rank, (_, row) in enumerate(event_importance.iterrows(), 1):
            std_str = f"{row['std']:.3f}" if pd.notna(row['std']) else "N/A"
            report_lines.append(f"{rank:<5} {row['feature']:<45} {row['mean']:>10.3f}  {std_str:>10}")
    
    # =========================================================================
    # SECTION 3: TOP FEATURES PER TOWER
    # =========================================================================
    report_lines.append("\n")
    report_lines.append("=" * 80)
    report_lines.append("SECTION 3: TOP FEATURES PER TOWER")
    report_lines.append("=" * 80)
    
    for tower in towers:
        report_lines.append(f"\n>> TOWER: {tower}")
        report_lines.append("-" * 60)
        
        tower_df = df[df['tower'] == tower]
        tower_importance = (tower_df.groupby('feature')['importance_pct']
                           .agg(['mean', 'std'])
                           .reset_index())
        tower_importance.columns = ['feature', 'mean', 'std']
        tower_importance = tower_importance.sort_values('mean', ascending=False).head(10)
        
        report_lines.append(f"{'Rank':<5} {'Feature':<45} {'Mean%':<12} {'Std%':<12}")
        report_lines.append("-" * 60)
        
        for rank, (_, row) in enumerate(tower_importance.iterrows(), 1):
            std_str = f"{row['std']:.3f}" if pd.notna(row['std']) else "N/A"
            report_lines.append(f"{rank:<5} {row['feature']:<45} {row['mean']:>10.3f}  {std_str:>10}")
    
    # =========================================================================
    # SECTION 4: STATISTICAL TESTS - KRUSKAL-WALLIS (Non-parametric ANOVA)
    # =========================================================================
    report_lines.append("\n")
    report_lines.append("=" * 80)
    report_lines.append("SECTION 4: STATISTICAL SIGNIFICANCE TESTS")
    report_lines.append("=" * 80)
    
    # 4.1 Kruskal-Wallis test across towers for each event
    report_lines.append("\n4.1 Kruskal-Wallis Test: Feature Importance Differences Across Towers")
    report_lines.append("    (Tests if importance distributions differ significantly between towers)")
    report_lines.append("-" * 80)
    
    for event in events:
        event_display = event.replace('_', ' ').replace('lt', '<').replace('gt', '>')
        event_df = df[df['event'] == event]
        
        # Get importance values per tower
        tower_groups = [event_df[event_df['tower'] == t]['importance_pct'].values for t in towers]
        tower_groups = [g for g in tower_groups if len(g) > 0]
        
        if len(tower_groups) >= 2:
            try:
                stat, p_value = stats.kruskal(*tower_groups)
                significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                report_lines.append(f"\n  Event: {event_display}")
                report_lines.append(f"    H-statistic: {stat:.4f}")
                report_lines.append(f"    p-value: {p_value:.6f} {significance}")
                report_lines.append(f"    Interpretation: {'Significant difference' if p_value < 0.05 else 'No significant difference'} between towers")
            except Exception as e:
                report_lines.append(f"\n  Event: {event_display}")
                report_lines.append(f"    Could not perform test: {str(e)}")
    
    # 4.2 Kruskal-Wallis test across events for each tower
    report_lines.append("\n\n4.2 Kruskal-Wallis Test: Feature Importance Differences Across Events")
    report_lines.append("    (Tests if importance distributions differ significantly between events)")
    report_lines.append("-" * 80)
    
    for tower in towers:
        tower_df = df[df['tower'] == tower]
        
        # Get importance values per event
        event_groups = [tower_df[tower_df['event'] == e]['importance_pct'].values for e in events]
        event_groups = [g for g in event_groups if len(g) > 0]
        
        if len(event_groups) >= 2:
            try:
                stat, p_value = stats.kruskal(*event_groups)
                significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < 0.05 else "ns"
                report_lines.append(f"\n  Tower: {tower}")
                report_lines.append(f"    H-statistic: {stat:.4f}")
                report_lines.append(f"    p-value: {p_value:.6f} {significance}")
                report_lines.append(f"    Interpretation: {'Significant difference' if p_value < 0.05 else 'No significant difference'} between events")
            except Exception as e:
                report_lines.append(f"\n  Tower: {tower}")
                report_lines.append(f"    Could not perform test: {str(e)}")
    
    # =========================================================================
    # SECTION 5: PAIRWISE COMPARISONS (Mann-Whitney U Test)
    # =========================================================================
    report_lines.append("\n")
    report_lines.append("=" * 80)
    report_lines.append("SECTION 5: PAIRWISE TOWER COMPARISONS (Mann-Whitney U Test)")
    report_lines.append("=" * 80)
    report_lines.append("\nComparing feature importance distributions between tower pairs:")
    report_lines.append("(Bonferroni-corrected significance level for multiple comparisons)")
    
    n_comparisons = len(list(combinations(towers, 2)))
    alpha_corrected = 0.05 / n_comparisons if n_comparisons > 0 else 0.05
    report_lines.append(f"Corrected alpha: {alpha_corrected:.6f} (original: 0.05, n_comparisons: {n_comparisons})")
    report_lines.append("-" * 80)
    
    for event in events:
        event_display = event.replace('_', ' ').replace('lt', '<').replace('gt', '>')
        report_lines.append(f"\n  Event: {event_display}")
        event_df = df[df['event'] == event]
        
        for t1, t2 in combinations(towers, 2):
            g1 = event_df[event_df['tower'] == t1]['importance_pct'].values
            g2 = event_df[event_df['tower'] == t2]['importance_pct'].values
            
            if len(g1) > 0 and len(g2) > 0:
                try:
                    stat, p_value = stats.mannwhitneyu(g1, g2, alternative='two-sided')
                    significance = "***" if p_value < 0.001 else "**" if p_value < 0.01 else "*" if p_value < alpha_corrected else "ns"
                    report_lines.append(f"    {t1} vs {t2}: U={stat:.1f}, p={p_value:.6f} {significance}")
                except Exception:
                    pass
    
    # =========================================================================
    # SECTION 6: FEATURE CONSISTENCY ANALYSIS
    # =========================================================================
    report_lines.append("\n")
    report_lines.append("=" * 80)
    report_lines.append("SECTION 6: FEATURE CONSISTENCY ANALYSIS")
    report_lines.append("=" * 80)
    report_lines.append("\nFeatures that appear in Top 10 across multiple tower-event combinations:")
    report_lines.append("-" * 80)
    
    # Count how often each feature appears in top 10 for each tower-event
    top_feature_counts = {}
    for tower in towers:
        for event in events:
            subset = df[(df['tower'] == tower) & (df['event'] == event)]
            top10 = subset.nsmallest(10, 'rank')['feature'].tolist()
            for feat in top10:
                if feat not in top_feature_counts:
                    top_feature_counts[feat] = 0
                top_feature_counts[feat] += 1
    
    # Sort by frequency
    sorted_features = sorted(top_feature_counts.items(), key=lambda x: x[1], reverse=True)
    total_combinations = len(towers) * len(events)
    
    report_lines.append(f"\n{'Feature':<50} {'Count':<10} {'%':<10} {'Consistency':<15}")
    report_lines.append("-" * 80)
    
    for feat, count in sorted_features[:20]:
        pct = (count / total_combinations) * 100
        consistency = "Very High" if pct >= 80 else "High" if pct >= 60 else "Moderate" if pct >= 40 else "Low"
        report_lines.append(f"{feat:<50} {count:<10} {pct:>6.1f}%    {consistency:<15}")
    
    # =========================================================================
    # SECTION 7: VARIABLE CATEGORY ANALYSIS
    # =========================================================================
    report_lines.append("\n")
    report_lines.append("=" * 80)
    report_lines.append("SECTION 7: VARIABLE CATEGORY IMPORTANCE SUMMARY")
    report_lines.append("=" * 80)
    
    def get_category(feature):
        base = re.split(r'_\d+m|_lag|_roll', feature)[0]
        return base
    
    df_cat = df.copy()
    df_cat['category'] = df_cat['feature'].apply(get_category)
    
    cat_summary = (df_cat.groupby('category')['importance_pct']
                   .agg(['sum', 'mean', 'std', 'count'])
                   .reset_index())
    cat_summary.columns = ['category', 'total', 'mean', 'std', 'count']
    cat_summary = cat_summary.sort_values('total', ascending=False)
    
    report_lines.append(f"\n{'Category':<25} {'Total%':<12} {'Mean%':<12} {'Std%':<12} {'Count':<10}")
    report_lines.append("-" * 80)
    
    for _, row in cat_summary.head(15).iterrows():
        report_lines.append(f"{row['category']:<25} {row['total']:>10.2f}  {row['mean']:>10.3f}  {row['std']:>10.3f}  {int(row['count']):>8}")
    
    # =========================================================================
    # SECTION 8: KEY FINDINGS & HIGHLIGHTS
    # =========================================================================
    report_lines.append("\n")
    report_lines.append("=" * 80)
    report_lines.append("SECTION 8: SPATIAL ANALYSIS OF FEATURE IMPORTANCE")
    report_lines.append("=" * 80)
    
    report_lines.append("\nFeature importance vs. geographic position (elevation, lat, lon):")
    report_lines.append("-" * 80)
    
    # Calculate top N importance sum per tower
    tower_importance_data = []
    for tower in towers:
        if tower not in TOWER_COORDINATES:
            continue
        tower_df = df[df['tower'] == tower]
        top5_sum = tower_df.nsmallest(5, 'rank')['importance_pct'].sum()
        coords = TOWER_COORDINATES[tower]
        tower_importance_data.append({
            'tower': tower,
            'lat': coords['lat'],
            'lon': coords['lon'],
            'elevation': coords['elevation_m'],
            'top5_importance': top5_sum
        })
    
    if tower_importance_data:
        t_df = pd.DataFrame(tower_importance_data)
        
        # Rank towers by importance
        t_df = t_df.sort_values('top5_importance', ascending=False)
        report_lines.append(f"\n{'Rank':<6} {'Tower':<8} {'Top5 Imp%':<12} {'Elevation':<12} {'Lat':<10} {'Lon':<10}")
        report_lines.append("-" * 70)
        for rank, (_, row) in enumerate(t_df.iterrows(), 1):
            report_lines.append(f"{rank:<6} {row['tower']:<8} {row['top5_importance']:>10.2f}  {row['elevation']:>10}m  {row['lat']:>8.4f}  {row['lon']:>8.4f}")
        
        # Calculate correlation between elevation and importance
        if len(t_df) >= 3:
            corr_elev, p_elev = spearmanr(t_df['elevation'], t_df['top5_importance'])
            report_lines.append(f"\nSpearman correlation (Elevation vs Top5 Importance): r={corr_elev:.3f}, p={p_elev:.4f}")
            if p_elev < 0.05:
                direction = "Higher" if corr_elev > 0 else "Lower"
                report_lines.append(f"  >> {direction} elevation towers tend to have {'more' if corr_elev > 0 else 'less'} concentrated feature importance")
            else:
                report_lines.append("  >> No significant correlation between elevation and feature importance")
    
    report_lines.append("")
    
    # =========================================================================
    report_lines.append("\n")
    report_lines.append("=" * 80)
    report_lines.append("SECTION 9: KEY FINDINGS & HIGHLIGHTS")
    report_lines.append("=" * 80)
    
    # Find the most dominant feature overall
    top_feature = global_importance.iloc[0]
    report_lines.append(f"\n1. MOST IMPORTANT FEATURE OVERALL:")
    report_lines.append(f"   - {top_feature['feature']}")
    report_lines.append(f"   - Mean importance: {top_feature['mean']:.3f}%")
    report_lines.append(f"   - Range: {top_feature['min']:.3f}% - {top_feature['max']:.3f}%")
    
    # Find the most consistent features
    most_consistent = sorted_features[0] if sorted_features else None
    if most_consistent:
        report_lines.append(f"\n2. MOST CONSISTENT FEATURE (appears in Top 10 most often):")
        report_lines.append(f"   - {most_consistent[0]}")
        report_lines.append(f"   - Appears in Top 10 for {most_consistent[1]}/{total_combinations} tower-event combinations ({(most_consistent[1]/total_combinations)*100:.1f}%)")
    
    # Find the most important category
    top_category = cat_summary.iloc[0]
    report_lines.append(f"\n3. MOST IMPORTANT VARIABLE CATEGORY:")
    report_lines.append(f"   - {top_category['category']}")
    report_lines.append(f"   - Total importance: {top_category['total']:.2f}%")
    
    # Event-specific highlights
    report_lines.append(f"\n4. EVENT-SPECIFIC TOP FEATURES:")
    for event in events:
        event_display = event.replace('_', ' ').replace('lt', '<').replace('gt', '>')
        event_df = df[df['event'] == event]
        top_feat = event_df.groupby('feature')['importance_pct'].mean().idxmax()
        top_val = event_df.groupby('feature')['importance_pct'].mean().max()
        report_lines.append(f"   - {event_display}: {top_feat} ({top_val:.2f}%)")
    
    # Tower-specific highlights
    report_lines.append(f"\n5. TOWER WITH HIGHEST FEATURE CONCENTRATION:")
    tower_concentration = df.groupby('tower')['importance_pct'].agg(lambda x: x.nlargest(5).sum())
    top_tower = tower_concentration.idxmax()
    report_lines.append(f"   - {top_tower} (Top 5 features account for {tower_concentration[top_tower]:.2f}% of importance)")
    
    # Geographic highlights
    if tower_importance_data:
        highest_elev_tower = max(tower_importance_data, key=lambda x: x['elevation'])
        lowest_elev_tower = min(tower_importance_data, key=lambda x: x['elevation'])
        report_lines.append(f"\n6. GEOGRAPHIC HIGHLIGHTS:")
        report_lines.append(f"   - Highest elevation tower: {highest_elev_tower['tower']} ({highest_elev_tower['elevation']}m)")
        report_lines.append(f"   - Lowest elevation tower: {lowest_elev_tower['tower']} ({lowest_elev_tower['elevation']}m)")
        
        # Find northernmost and southernmost
        north_tower = max(tower_importance_data, key=lambda x: x['lat'])
        south_tower = min(tower_importance_data, key=lambda x: x['lat'])
        report_lines.append(f"   - Northernmost tower: {north_tower['tower']} ({north_tower['lat']:.4f}N)")
        report_lines.append(f"   - Southernmost tower: {south_tower['tower']} ({south_tower['lat']:.4f}N)")
    
    # Footer
    report_lines.append("\n")
    report_lines.append("=" * 80)
    report_lines.append("STATISTICAL SIGNIFICANCE LEGEND:")
    report_lines.append("  *** p < 0.001 (highly significant)")
    report_lines.append("  **  p < 0.01  (very significant)")
    report_lines.append("  *   p < 0.05  (significant)")
    report_lines.append("  ns  p >= 0.05 (not significant)")
    report_lines.append("=" * 80)
    report_lines.append("END OF REPORT")
    report_lines.append("=" * 80)
    
    # Combine and save
    report_text = "\n".join(report_lines)
    
    if output_folder:
        report_path = os.path.join(output_folder, 'feature_importance_report.txt')
        with open(report_path, 'w') as f:
            f.write(report_text)
        print(f"Report saved: {report_path}")
    
    return report_text


def interactive_menu(df: pd.DataFrame, folder_path: str):
    """Interactive menu for visualization selection."""
    
    towers = sorted(df['tower'].unique())
    events = sorted(df['event'].unique())
    
    # Create output folder for saved figures
    output_folder = os.path.join(folder_path, 'viz_importance')
    os.makedirs(output_folder, exist_ok=True)
    
    while True:
        print("\n" + "="*60)
        print("       FEATURE IMPORTANCE VISUALIZATION TOOL")
        print("="*60)
        print(f"\nData loaded: {len(towers)} towers, {len(events)} events")
        print(f"Towers: {', '.join(towers)}")
        print(f"Events: {', '.join(events)}")
        print("\n" + "-"*60)
        print("OPTIONS:")
        print("-"*60)
        print("  1. View top features for specific tower & event")
        print("  2. Compare features across towers - HEATMAP (same event)")
        print("  3. Compare features across events (same tower)")
        print("  4. Global feature importance (all towers & events)")
        print("  5. Variable category importance analysis")
        print("  6. Generate summary dashboard")
        print("  7. Generate ALL visualizations")
        print("-"*60)
        print("  === CONSOLIDATED VIEWS (ONE IMAGE) ===")
        print("  8. ALL TOWERS per event - Grid subplots")
        print("  9. ALL TOWERS per event - Dot plot (clustered)")
        print(" 10. MASTER VIEW - All towers x All events")
        print("-"*60)
        print("  === GEOGRAPHIC VIEWS ===")
        print(" 11. [NEW] 3D Geographic View - Tower importance in 3D")
        print(" 12. [NEW] 2D Map View - Bubble map of importance")
        print("-"*60)
        print("  === REPORTS ===")
        print(" 13. Generate Statistical Report (.txt with geographic info)")
        print("-"*60)
        print("  0. Exit")
        print("-"*60)
        
        choice = input("\nEnter your choice (0-13): ").strip()
        
        if choice == '0':
            print("\nGoodbye!")
            break
            
        elif choice == '1':
            print(f"\nAvailable towers: {', '.join(towers)}")
            tower = input("Enter tower name: ").strip().upper()
            if tower not in towers:
                print(f"Invalid tower. Choose from: {towers}")
                continue
            
            print(f"\nAvailable events: {', '.join(events)}")
            event = input("Enter event name: ").strip()
            if event not in events:
                print(f"Invalid event. Choose from: {events}")
                continue
            
            top_n = input("Number of top features to show (default 20): ").strip()
            top_n = int(top_n) if top_n.isdigit() else 20
            
            save = input("Save figure? (y/n): ").strip().lower() == 'y'
            save_path = os.path.join(output_folder, f'top_features_{tower}_{event}.png') if save else None
            
            plot_top_features_by_tower_event(df, tower, event, top_n, save_path)
            
        elif choice == '2':
            print(f"\nAvailable events: {', '.join(events)}")
            event = input("Enter event name: ").strip()
            if event not in events:
                print(f"Invalid event. Choose from: {events}")
                continue
            
            top_n = input("Number of top features to show (default 15): ").strip()
            top_n = int(top_n) if top_n.isdigit() else 15
            
            save = input("Save figure? (y/n): ").strip().lower() == 'y'
            save_path = os.path.join(output_folder, f'comparison_towers_{event}.png') if save else None
            
            plot_feature_comparison_across_towers(df, event, top_n, save_path)
            
        elif choice == '3':
            print(f"\nAvailable towers: {', '.join(towers)}")
            tower = input("Enter tower name: ").strip().upper()
            if tower not in towers:
                print(f"Invalid tower. Choose from: {towers}")
                continue
            
            top_n = input("Number of top features to show (default 15): ").strip()
            top_n = int(top_n) if top_n.isdigit() else 15
            
            save = input("Save figure? (y/n): ").strip().lower() == 'y'
            save_path = os.path.join(output_folder, f'comparison_events_{tower}.png') if save else None
            
            plot_feature_comparison_across_events(df, tower, top_n, save_path)
            
        elif choice == '4':
            top_n = input("Number of top features to show (default 25): ").strip()
            top_n = int(top_n) if top_n.isdigit() else 25
            
            save = input("Save figure? (y/n): ").strip().lower() == 'y'
            save_path = os.path.join(output_folder, 'global_importance.png') if save else None
            
            plot_global_feature_importance(df, top_n, save_path)
            
        elif choice == '5':
            save = input("Save figure? (y/n): ").strip().lower() == 'y'
            save_path = os.path.join(output_folder, 'category_importance.png') if save else None
            
            plot_variable_category_importance(df, save_path)
            
        elif choice == '6':
            create_summary_dashboard(df, output_folder)
            
        elif choice == '7':
            print("\nGenerating ALL visualizations...")
            
            # Global importance
            plot_global_feature_importance(df, 25, 
                os.path.join(output_folder, 'global_importance.png'))
            
            # Category importance
            plot_variable_category_importance(df, 
                os.path.join(output_folder, 'category_importance.png'))
            
            # Dashboard
            create_summary_dashboard(df, output_folder)
            
            # Per tower-event
            for tower in towers:
                for event in events:
                    plot_top_features_by_tower_event(df, tower, event, 20,
                        os.path.join(output_folder, f'top_features_{tower}_{event}.png'))
            
            # Cross-tower comparisons
            for event in events:
                plot_feature_comparison_across_towers(df, event, 15,
                    os.path.join(output_folder, f'comparison_towers_{event}.png'))
            
            # Cross-event comparisons
            for tower in towers:
                plot_feature_comparison_across_events(df, tower, 15,
                    os.path.join(output_folder, f'comparison_events_{tower}.png'))
            
            print(f"\nAll visualizations saved to: {output_folder}")
            
        elif choice == '8':
            # ALL TOWERS per event - Grid subplots
            print(f"\nAvailable events: {', '.join(events)}")
            event = input("Enter event name (or 'all' for all events): ").strip()
            
            top_n = input("Number of top features to show (default 15): ").strip()
            top_n = int(top_n) if top_n.isdigit() else 15
            
            if event.lower() == 'all':
                for ev in events:
                    save_path = os.path.join(output_folder, f'consolidated_towers_{ev}.png')
                    plot_all_towers_per_event(df, ev, top_n, save_path)
            else:
                if event not in events:
                    print(f"Invalid event. Choose from: {events}")
                    continue
                save = input("Save figure? (y/n): ").strip().lower() == 'y'
                save_path = os.path.join(output_folder, f'consolidated_towers_{event}.png') if save else None
                plot_all_towers_per_event(df, event, top_n, save_path)
        
        elif choice == '9':
            # ALL TOWERS per event - Dot plot
            print(f"\nAvailable events: {', '.join(events)}")
            event = input("Enter event name (or 'all' for all events): ").strip()
            
            top_n = input("Number of top features to show (default 15): ").strip()
            top_n = int(top_n) if top_n.isdigit() else 15
            
            if event.lower() == 'all':
                for ev in events:
                    save_path = os.path.join(output_folder, f'dotplot_{ev}.png')
                    plot_grouped_bars_per_event(df, ev, top_n, save_path)
            else:
                if event not in events:
                    print(f"Invalid event. Choose from: {events}")
                    continue
                save = input("Save figure? (y/n): ").strip().lower() == 'y'
                save_path = os.path.join(output_folder, f'dotplot_{event}.png') if save else None
                plot_grouped_bars_per_event(df, event, top_n, save_path)
        
        elif choice == '10':
            # MASTER VIEW - All towers × All events
            top_n = input("Number of top features per cell (default 15): ").strip()
            top_n = int(top_n) if top_n.isdigit() else 15
            
            save = input("Save figure? (y/n): ").strip().lower() == 'y'
            save_path = os.path.join(output_folder, 'MASTER_all_towers_all_events.png') if save else None
            
            plot_consolidated_all_events(df, top_n, save_path)
        
        elif choice == '11':
            # 3D Geographic View
            print(f"\nAvailable events: {', '.join(events)}")
            event = input("Enter event name (or 'all' for combined view): ").strip()
            
            top_n = input("Number of top features for height (default 5): ").strip()
            top_n = int(top_n) if top_n.isdigit() else 5
            
            if event.lower() == 'all':
                save = input("Save figure? (y/n): ").strip().lower() == 'y'
                save_path = os.path.join(output_folder, 'geo_3d_all_events.png') if save else None
                plot_3d_geographic_importance(df, event=None, top_n=top_n, save_path=save_path)
            else:
                if event not in events:
                    print(f"Invalid event. Choose from: {events}")
                    continue
                save = input("Save figure? (y/n): ").strip().lower() == 'y'
                save_path = os.path.join(output_folder, f'geo_3d_{event}.png') if save else None
                plot_3d_geographic_importance(df, event=event, top_n=top_n, save_path=save_path)
        
        elif choice == '12':
            # 2D Map View
            print(f"\nAvailable events: {', '.join(events)}")
            event = input("Enter event name (or 'all' for combined view): ").strip()
            
            top_n = input("Number of top features for bubble size (default 5): ").strip()
            top_n = int(top_n) if top_n.isdigit() else 5
            
            if event.lower() == 'all':
                save = input("Save figure? (y/n): ").strip().lower() == 'y'
                save_path = os.path.join(output_folder, 'geo_2d_map_all_events.png') if save else None
                plot_2d_geographic_map(df, event=None, top_n=top_n, save_path=save_path)
            else:
                if event not in events:
                    print(f"Invalid event. Choose from: {events}")
                    continue
                save = input("Save figure? (y/n): ").strip().lower() == 'y'
                save_path = os.path.join(output_folder, f'geo_2d_map_{event}.png') if save else None
                plot_2d_geographic_map(df, event=event, top_n=top_n, save_path=save_path)
        
        elif choice == '13':
            # Generate Statistical Report
            print("\nGenerating Statistical Report with significance tests and geographic info...")
            report = generate_statistical_report(df, output_folder)
            print("\n" + "="*60)
            print("Report Preview (first 60 lines):")
            print("="*60)
            for line in report.split('\n')[:60]:
                print(line)
            print("\n... (see full report in output folder)")
            
        else:
            print("Invalid choice. Please enter 0-13.")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python viz_importance.py <results_folder_path>")
        print("Example: python viz_importance.py BESTML_multi_event_results_20251208_155338")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    
    # Handle relative paths
    if not os.path.isabs(folder_path):
        folder_path = os.path.join(os.getcwd(), folder_path)
    
    if not os.path.isdir(folder_path):
        print(f"Error: Folder not found: {folder_path}")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print("  Loading Feature Importance Data")
    print(f"{'='*60}")
    print(f"Folder: {folder_path}\n")
    
    # Load data
    data = load_importance_files(folder_path)
    
    if not data:
        print("No data to visualize.")
        sys.exit(1)
    
    # Combine into single dataframe
    df = get_combined_dataframe(data)
    print(f"\nTotal records loaded: {len(df)}")
    
    # Run interactive menu
    interactive_menu(df, folder_path)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Feature Importance Visualization Tool
======================================
Visualizes feature importance across towers, events, and variables
from ML experiment results.

Includes geographic analysis with tower metadata loaded from JSON file.
All tower variables are considered: elevation, height, slope, canopy, etc.

Usage:
    python viz_importance.py <results_folder_path> [tower_metadata.json]
    
Examples:
    python viz_importance.py BESTML_multi_event_results_20251208_155338
    python viz_importance.py BESTML_results tower_metadata_generated.json

Tower Metadata JSON Format:
    The script looks for 'tower_metadata_generated.json' in:
    1. Provided path (2nd argument)
    2. Results folder
    3. Parent directory of results folder
    4. Current working directory
    5. Script directory
    
    JSON structure:
    {
        "tower_coordinates": {
            "TOWA": {
                "lat": 35.92004,
                "lon": -84.305079,
                "elevation_m": 260,
                "elevation_msl_m": 263,
                "height_m": 33,
                "name": "ORNL Tower A",
                "terrain_type": "Valley Floor",
                "slope_deg": 0.0,
                ...
            },
            ...
        },
        "region_info": {
            "name": "Oak Ridge Reservation",
            "state": "Tennessee, USA",
            "center_lat": 35.928252,
            "center_lon": -84.317857,
            ...
        }
    }
"""

import os
import sys
import glob
import re
import json
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
# Tower metadata loaded from JSON file

TOWER_COORDINATES = {}
REGION_INFO = {}

# Default JSON metadata file name
DEFAULT_METADATA_FILE = "tower_metadata_generated.json"


def load_tower_metadata(json_path: str = None, folder_path: str = None) -> tuple:
    """
    Load tower coordinates and region info from JSON file.
    
    Args:
        json_path: Direct path to JSON file (optional)
        folder_path: Folder path where to look for default JSON file (optional)
    
    Returns:
        Tuple of (tower_coordinates dict, region_info dict)
    """
    global TOWER_COORDINATES, REGION_INFO
    
    # Try to find JSON file
    search_paths = []
    
    if json_path and os.path.isfile(json_path):
        search_paths.append(json_path)
    
    if folder_path:
        # Check in the results folder
        search_paths.append(os.path.join(folder_path, DEFAULT_METADATA_FILE))
        # Check in parent directory
        search_paths.append(os.path.join(os.path.dirname(folder_path), DEFAULT_METADATA_FILE))
    
    # Check in current working directory
    search_paths.append(os.path.join(os.getcwd(), DEFAULT_METADATA_FILE))
    
    # Check in script directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    search_paths.append(os.path.join(script_dir, DEFAULT_METADATA_FILE))
    
    # Find first existing file
    json_file = None
    for path in search_paths:
        if os.path.isfile(path):
            json_file = path
            break
    
    if json_file is None:
        print(f"WARNING: Tower metadata JSON file not found. Searched in:")
        for path in search_paths:
            print(f"  - {path}")
        print("Geographic visualizations will be limited.")
        return {}, {}
    
    print(f"Loading tower metadata from: {json_file}")
    
    try:
        with open(json_file, 'r') as f:
            metadata = json.load(f)
        
        TOWER_COORDINATES = metadata.get('tower_coordinates', {})
        REGION_INFO = metadata.get('region_info', {})
        
        print(f"  Loaded {len(TOWER_COORDINATES)} towers: {', '.join(TOWER_COORDINATES.keys())}")
        print(f"  Region: {REGION_INFO.get('name', 'Unknown')}")
        
        return TOWER_COORDINATES, REGION_INFO
    
    except json.JSONDecodeError as e:
        print(f"ERROR: Failed to parse JSON file: {e}")
        return {}, {}
    except Exception as e:
        print(f"ERROR: Failed to load metadata: {e}")
        return {}, {}



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
    Includes all variables: height_m, elevation_msl_m, elevation_m, etc.
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
        
        # Calculate distance from center (handle missing REGION_INFO)
        if REGION_INFO and 'center_lat' in REGION_INFO and 'center_lon' in REGION_INFO:
            lat_diff = coords['lat'] - REGION_INFO['center_lat']
            lon_diff = coords['lon'] - REGION_INFO['center_lon']
            dist_from_center_km = np.sqrt((lat_diff * 111)**2 + (lon_diff * 111 * np.cos(np.radians(coords['lat'])))**2)
        else:
            dist_from_center_km = 0
        
        # Get all tower variables including height and MSL elevation
        tower_entry = {
            'tower': tower,
            'name': coords.get('name', tower),
            'lat': coords.get('lat', 0),
            'lon': coords.get('lon', 0),
            'elevation_m': coords.get('elevation_m', 0),
            'elevation_msl_m': coords.get('elevation_msl_m', coords.get('elevation_m', 0)),  # MSL elevation
            'height_m': coords.get('height_m', 0),  # Tower height
            'total_height_m': coords.get('elevation_m', 0) + coords.get('height_m', 0),  # Total height (ground + tower)
            'terrain_type': coords.get('terrain_type', 'Unknown'),
            'slope_deg': coords.get('slope_deg', 0),
            'aspect_deg': coords.get('aspect_deg', 0),
            'canopy_cover_pct': coords.get('canopy_cover_pct', 0),
            'dist_to_ridge_m': coords.get('dist_to_ridge_m', 0),
            'dist_from_center_km': dist_from_center_km,
            'soil_type': coords.get('soil_type', 'Unknown'),
            'land_use': coords.get('land_use', 'Unknown'),
            'color': coords.get('color', '#333333'),  # Tower color from JSON
            'top5_importance': top_importance_sum,
            'top1_feature': top_feature_1,
            'top1_importance': top_feature_1_imp,
        }
        tower_data.append(tower_entry)
    
    return pd.DataFrame(tower_data)


def plot_geographic_features_vs_importance(df: pd.DataFrame, output_folder: str = None):
    """
    Create comprehensive geographic analysis plots - ONE IMAGE PER EVENT.
    Shows how geographic/terrain features relate to top 5 feature importance.
    Generates images (one per target variable/event).
    Includes tower height and MSL elevation analysis.
    """
    events = sorted(df['event'].unique())
    
    for event in events:
        event_display = event.replace('_', ' ').replace('lt', '<').replace('gt', '>')
        
        # Get tower geographic data for this event
        geo_df = get_tower_geo_dataframe(df, event, top_n=5)
        
        if geo_df.empty:
            print(f"No data for event: {event}")
            continue
        
        # Create figure with 3x3 subplots to include tower height variables
        fig, axes = plt.subplots(3, 3, figsize=(20, 18))
        region_name = REGION_INFO.get("name", "Unknown Region") if REGION_INFO else "Unknown Region"
        region_state = REGION_INFO.get("state", "") if REGION_INFO else ""
        fig.suptitle(f'Geographic Features vs Feature Importance\nTarget: {event_display}\n'
                     f'{region_name}, {region_state}',
                     fontsize=16, fontweight='bold', y=1.02)
        
        # Use tower colors from JSON if available, otherwise use colormap
        tower_colors = {}
        for _, row in geo_df.iterrows():
            if row.get('color'):
                tower_colors[row['tower']] = row['color']
            else:
                tower_colors[row['tower']] = plt.cm.Set2(list(geo_df['tower']).index(row['tower']) / len(geo_df))
        
        # Helper function to plot scatter with trend line
        def plot_scatter_with_trend(ax, x_col, y_col, xlabel, title):
            for _, row in geo_df.iterrows():
                ax.scatter(row[x_col], row[y_col], 
                          s=200, c=[tower_colors[row['tower']]], 
                          edgecolors='black', linewidth=1.5, zorder=3)
                ax.annotate(row['tower'], (row[x_col], row[y_col]),
                           xytext=(5, 5), textcoords='offset points', fontsize=10, fontweight='bold')
            
            # Add trend line
            if len(geo_df) >= 2:
                z = np.polyfit(geo_df[x_col], geo_df[y_col], 1)
                p = np.poly1d(z)
                x_line = np.linspace(geo_df[x_col].min() - (geo_df[x_col].max() - geo_df[x_col].min()) * 0.1, 
                                    geo_df[x_col].max() + (geo_df[x_col].max() - geo_df[x_col].min()) * 0.1, 100)
                ax.plot(x_line, p(x_line), 'r--', alpha=0.7, linewidth=2, label='Trend')
                
                # Correlation
                corr, pval = spearmanr(geo_df[x_col], geo_df[y_col])
                ax.text(0.05, 0.95, f'r={corr:.3f}\np={pval:.3f}', transform=ax.transAxes,
                       fontsize=10, verticalalignment='top', 
                       bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
            
            ax.set_xlabel(xlabel, fontsize=11)
            ax.set_ylabel('Top 5 Features Importance (%)', fontsize=11)
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)
        
        # =====================================================================
        # Plot 1: Ground Elevation vs Top 5 Importance
        # =====================================================================
        plot_scatter_with_trend(axes[0, 0], 'elevation_m', 'top5_importance', 
                               'Ground Elevation (m)', 'Ground Elevation vs Importance')
        
        # =====================================================================
        # Plot 2: Tower Height vs Top 5 Importance (NEW)
        # =====================================================================
        plot_scatter_with_trend(axes[0, 1], 'height_m', 'top5_importance',
                               'Tower Height (m)', 'Tower Height vs Importance')
        
        # =====================================================================
        # Plot 3: Total Height (ground + tower) vs Top 5 Importance (NEW)
        # =====================================================================
        plot_scatter_with_trend(axes[0, 2], 'total_height_m', 'top5_importance',
                               'Total Height (Ground + Tower) (m)', 'Total Height vs Importance')
        
        # =====================================================================
        # Plot 4: Slope vs Top 5 Importance
        # =====================================================================
        plot_scatter_with_trend(axes[1, 0], 'slope_deg', 'top5_importance',
                               'Terrain Slope (degrees)', 'Slope vs Importance')
        
        # =====================================================================
        # Plot 5: Canopy Cover vs Top 5 Importance
        # =====================================================================
        plot_scatter_with_trend(axes[1, 1], 'canopy_cover_pct', 'top5_importance',
                               'Canopy Cover (%)', 'Vegetation Cover vs Importance')
        
        # =====================================================================
        # Plot 6: Distance to Ridge vs Top 5 Importance
        # =====================================================================
        plot_scatter_with_trend(axes[1, 2], 'dist_to_ridge_m', 'top5_importance',
                               'Distance to Ridge (m)', 'Ridge Proximity vs Importance')
        
        # =====================================================================
        # Plot 7: Terrain Type (categorical) vs Top 5 Importance - Box/Bar
        # =====================================================================
        ax7 = axes[2, 0]
        terrain_order = ['Valley Floor', 'Valley Slope', 'Ridge Slope', 'Ridge Top']
        terrain_data = geo_df.groupby('terrain_type')['top5_importance'].mean().reindex(terrain_order).dropna()
        
        if len(terrain_data) > 0:
            bars = ax7.bar(range(len(terrain_data)), terrain_data.values, 
                          color=plt.cm.terrain(np.linspace(0.2, 0.8, len(terrain_data))),
                          edgecolor='black', linewidth=1.5)
            ax7.set_xticks(range(len(terrain_data)))
            ax7.set_xticklabels(terrain_data.index, rotation=30, ha='right', fontsize=10)
            
            # Add tower labels on bars
            for terrain in terrain_data.index:
                towers_in_terrain = geo_df[geo_df['terrain_type'] == terrain]['tower'].tolist()
                idx = list(terrain_data.index).index(terrain)
                ax7.text(idx, terrain_data[terrain] + 0.5, ', '.join(towers_in_terrain),
                        ha='center', fontsize=9, fontweight='bold')
        
        ax7.set_xlabel('Terrain Type', fontsize=11)
        ax7.set_ylabel('Mean Top 5 Importance (%)', fontsize=11)
        ax7.set_title('Terrain Type vs Importance', fontsize=12, fontweight='bold')
        ax7.grid(axis='y', alpha=0.3)
        
        # =====================================================================
        # Plot 8: Geographic Map with Importance as color/size
        # =====================================================================
        ax8 = axes[2, 1]
        
        # Size by importance, color by total height
        sizes = (geo_df['top5_importance'] / geo_df['top5_importance'].max()) * 800 + 100
        scatter = ax8.scatter(geo_df['lon'], geo_df['lat'], 
                             s=sizes, c=geo_df['total_height_m'],
                             cmap='viridis', edgecolors='black', linewidth=2, alpha=0.8)
        
        # Add tower labels with height info
        for _, row in geo_df.iterrows():
            ax8.annotate(f"{row['tower']}\n{row['height_m']}m tower",
                        (row['lon'], row['lat']),
                        xytext=(8, 8), textcoords='offset points',
                        fontsize=9, fontweight='bold',
                        bbox=dict(boxstyle='round,pad=0.2', facecolor='white', alpha=0.7))
        
        cbar = plt.colorbar(scatter, ax=ax8, shrink=0.8)
        cbar.set_label('Total Height (Ground + Tower) (m)', fontsize=10)
        
        ax8.set_xlabel('Longitude (°W)', fontsize=11)
        ax8.set_ylabel('Latitude (°N)', fontsize=11)
        ax8.set_title('Geographic Distribution\n(size=importance, color=total height)', 
                     fontsize=12, fontweight='bold')
        ax8.grid(True, alpha=0.3)
        
        # =====================================================================
        # Plot 9: Tower Info Summary Table
        # =====================================================================
        ax9 = axes[2, 2]
        ax9.axis('off')
        
        # Create summary table with all variables
        summary_text = "TOWER SUMMARY (sorted by importance)\n" + "="*55 + "\n\n"
        summary_text += f"{'Tower':<6} {'Elev(m)':<8} {'TowHt(m)':<9} {'Total(m)':<9} {'Slope°':<7} {'Canopy%':<8} {'Imp%':<6}\n"
        summary_text += "-"*55 + "\n"
        
        for _, row in geo_df.sort_values('top5_importance', ascending=False).iterrows():
            summary_text += f"{row['tower']:<6} {row['elevation_m']:<8.0f} {row['height_m']:<9.0f} {row['total_height_m']:<9.0f} "
            summary_text += f"{row['slope_deg']:<7.1f} {row['canopy_cover_pct']:<8.0f} {row['top5_importance']:<6.1f}\n"
        
        summary_text += "\n" + "="*55 + "\n"
        summary_text += f"{'Terrain Type':<15} | {'Land Use':<18} | Tower\n"
        summary_text += "-"*55 + "\n"
        for _, row in geo_df.iterrows():
            summary_text += f"{row['terrain_type']:<15} | {row['land_use']:<18} | {row['tower']}\n"
        
        ax9.text(0.05, 0.95, summary_text, transform=ax9.transAxes, fontsize=9, 
                fontfamily='monospace', verticalalignment='top',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.9))
        ax9.set_title('Tower Characteristics Summary', fontsize=12, fontweight='bold')
        
        plt.tight_layout()
        
        if output_folder:
            save_path = os.path.join(output_folder, f'geo_features_vs_importance_{event}.png')
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved: {save_path}")
        
        plt.show()
        plt.close()


def plot_3d_terrain_importance(df: pd.DataFrame, event: str, output_folder: str = None):
    """
    Create a 3D terrain visualization showing elevation profile with importance markers.
    Includes tower height visualization.
    """
    event_display = event.replace('_', ' ').replace('lt', '<').replace('gt', '>')
    geo_df = get_tower_geo_dataframe(df, event, top_n=5)
    
    if geo_df.empty:
        print(f"No data for event: {event}")
        return
    
    if not REGION_INFO or 'center_lat' not in REGION_INFO:
        print(f"No region info available for 3D terrain plot")
        return
    
    # Create 3D figure
    fig = plt.figure(figsize=(18, 14))
    ax = fig.add_subplot(111, projection='3d')
    
    # Convert lat/lon to km from center
    lat_center = REGION_INFO.get('center_lat', geo_df['lat'].mean())
    lon_center = REGION_INFO.get('center_lon', geo_df['lon'].mean())
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
    
    # Plot towers as vertical bars from terrain to tower top height + importance
    for idx, row in geo_df.iterrows():
        # Get heights
        terrain_height = row['elevation_m']
        tower_height = row.get('height_m', 0)
        tower_top = terrain_height + tower_height
        importance_scaled = row['top5_importance'] * 2  # Scale for visibility
        importance_top = tower_top + importance_scaled
        
        # Use tower color from JSON
        tower_color = row.get('color', 'red')
        
        # Plot tower structure (ground to tower top) - thick gray line
        ax.plot([row['x_km'], row['x_km']], 
                [row['y_km'], row['y_km']], 
                [terrain_height, tower_top],
                color='gray', linewidth=6, alpha=0.8, label='Tower' if idx == 0 else '')
        
        # Plot importance bar (tower top to importance top) - colored line
        ax.plot([row['x_km'], row['x_km']], 
                [row['y_km'], row['y_km']], 
                [tower_top, importance_top],
                color=tower_color, linewidth=4, alpha=0.9)
        
        # Plot importance sphere at top
        ax.scatter([row['x_km']], [row['y_km']], [importance_top],
                  s=row['top5_importance']*10 + 100, c=[tower_color], 
                  edgecolors='darkred', linewidth=2, alpha=0.8, zorder=5)
        
        # Tower label with height info
        label_text = f"{row['tower']}\n{row['height_m']:.0f}m tower\n{row['top5_importance']:.1f}%"
        ax.text(row['x_km'], row['y_km'], importance_top + 10,
               label_text,
               ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Base marker at ground level
        ax.scatter([row['x_km']], [row['y_km']], [terrain_height],
                  s=100, c='brown', marker='^', edgecolors='black', zorder=5)
        
        # Marker at tower top
        ax.scatter([row['x_km']], [row['y_km']], [tower_top],
                  s=50, c='blue', marker='s', edgecolors='black', zorder=5)
    
    # Labels
    ax.set_xlabel('West ← Distance (km) → East', fontsize=11, labelpad=10)
    ax.set_ylabel('South ← Distance (km) → North', fontsize=11, labelpad=10)
    ax.set_zlabel('Height (m)', fontsize=11, labelpad=10)
    
    region_name = REGION_INFO.get('name', 'Unknown')
    region_terrain = REGION_INFO.get('terrain', 'Unknown')
    region_climate = REGION_INFO.get('climate', 'Unknown')
    
    ax.set_title(f'3D Terrain Profile with Tower Heights & Feature Importance\nTarget: {event_display}\n'
                f'{region_name} - {region_terrain}',
                fontsize=14, fontweight='bold')
    
    # Add legend/info
    info_text = (f"Gray bars: Physical tower structure (ground to sensor)\n"
                f"Colored bars: Top 5 Feature Importance (scaled)\n"
                f"Triangle: Ground level | Square: Tower top\n"
                f"Tower heights: {geo_df['height_m'].min():.0f}m - {geo_df['height_m'].max():.0f}m\n"
                f"Climate: {region_climate}")
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


def generate_statistical_report(df: pd.DataFrame, output_folder: str = None, folder_path: str = None):
    """
    Generate a comprehensive text report with statistical highlights.
    Pulls model performance data from summary_results.csv if available.
    """
    towers = sorted(df['tower'].unique())
    events = sorted(df['event'].unique())
    
    report_lines = []
    
    # Try to load model performance data from CSV files
    summary_df = None
    best_event_df = None
    best_tower_df = None
    
    # Determine the base folder path
    base_folder = folder_path if folder_path else (os.path.dirname(output_folder) if output_folder else None)
    
    if base_folder:
        summary_path = os.path.join(base_folder, 'summary_results.csv')
        best_event_path = os.path.join(base_folder, 'best_models_per_event.csv')
        best_tower_path = os.path.join(base_folder, 'best_models_per_tower.csv')
        best_tower_event_path = os.path.join(base_folder, 'best_models_per_tower_event.csv')
        
        if os.path.exists(summary_path):
            summary_df = pd.read_csv(summary_path)
            print(f"Loaded model performance from: {summary_path}")
        if os.path.exists(best_event_path):
            best_event_df = pd.read_csv(best_event_path)
        if os.path.exists(best_tower_path):
            best_tower_df = pd.read_csv(best_tower_path)
        if os.path.exists(best_tower_event_path):
            best_tower_event_df = pd.read_csv(best_tower_event_path)
    
    # Header
    report_lines.append("=" * 80)
    report_lines.append("       FEATURE IMPORTANCE & MODEL PERFORMANCE REPORT")
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
    
    # Handle missing REGION_INFO gracefully
    if REGION_INFO:
        report_lines.append(f"\nStudy Region: {REGION_INFO.get('name', 'Unknown')}")
        report_lines.append(f"Location: {REGION_INFO.get('state', 'Unknown')}")
        if 'center_lat' in REGION_INFO and 'center_lon' in REGION_INFO:
            report_lines.append(f"Center Coordinates: {REGION_INFO['center_lat']:.4f}N, {abs(REGION_INFO['center_lon']):.4f}W")
        report_lines.append(f"Terrain: {REGION_INFO.get('terrain', 'Unknown')}")
        report_lines.append(f"Climate: {REGION_INFO.get('climate', 'Unknown')}")
    else:
        report_lines.append("\nNo region information available.")
    
    report_lines.append("")
    report_lines.append("Tower Locations & Physical Characteristics:")
    report_lines.append("-" * 100)
    report_lines.append(f"{'Tower':<8} {'Terrain Type':<15} {'Elev(m)':<9} {'TowHt(m)':<10} {'Total(m)':<10} {'Slope°':<8} {'Canopy%':<9} {'Land Use':<18}")
    report_lines.append("-" * 100)
    for tower in towers:
        if tower in TOWER_COORDINATES:
            c = TOWER_COORDINATES[tower]
            elev = c.get('elevation_m', 0)
            height = c.get('height_m', 0)
            total_height = elev + height
            report_lines.append(f"{tower:<8} {c.get('terrain_type', 'N/A'):<15} {elev:<9.0f} {height:<10.0f} {total_height:<10.0f} {c.get('slope_deg', 0):<6.1f}°  {c.get('canopy_cover_pct', 0):<7.0f}%  {c.get('land_use', 'N/A'):<18}")
    
    # Calculate spatial spread
    lats = [TOWER_COORDINATES[t]['lat'] for t in towers if t in TOWER_COORDINATES]
    lons = [TOWER_COORDINATES[t]['lon'] for t in towers if t in TOWER_COORDINATES]
    elevs = [TOWER_COORDINATES[t].get('elevation_m', 0) for t in towers if t in TOWER_COORDINATES]
    heights = [TOWER_COORDINATES[t].get('height_m', 0) for t in towers if t in TOWER_COORDINATES]
    
    if lats and lons:
        lat_spread_km = (max(lats) - min(lats)) * 111
        lon_spread_km = (max(lons) - min(lons)) * 111 * np.cos(np.radians(np.mean(lats)))
        report_lines.append("")
        report_lines.append(f"Spatial Extent: ~{lat_spread_km:.2f} km (N-S) x ~{abs(lon_spread_km):.2f} km (E-W)")
        report_lines.append(f"Ground Elevation Range: {min(elevs):.0f}m - {max(elevs):.0f}m ({max(elevs)-min(elevs):.0f}m difference)")
        if heights:
            report_lines.append(f"Tower Height Range: {min(heights):.0f}m - {max(heights):.0f}m")
            total_heights = [e + h for e, h in zip(elevs, heights)]
            report_lines.append(f"Total Height Range (Ground+Tower): {min(total_heights):.0f}m - {max(total_heights):.0f}m")
    report_lines.append("")
    
    # =========================================================================
    # MODEL PERFORMANCE FROM CSV (if available)
    # =========================================================================
    if best_event_df is not None:
        report_lines.append("=" * 80)
        report_lines.append("MODEL PERFORMANCE SUMMARY (from summary_results.csv)")
        report_lines.append("=" * 80)
        
        report_lines.append("\n>> BEST MODEL PERFORMANCE BY EVENT:")
        report_lines.append("-" * 80)
        report_lines.append(f"{'Event':<30} {'Avg Event Rate':<15} {'Best AUC-ROC':<12} {'Best Model':<12} {'Best Tower':<10}")
        report_lines.append("-" * 80)
        
        for _, row in best_event_df.iterrows():
            event_name = row['event'].replace('event_', '').replace('_', ' ')
            # Get best AUC between lightgbm and xgboost
            lgb_auc = row.get('lightgbm_best_auc_roc', 0)
            xgb_auc = row.get('xgboost_best_auc_roc', 0)
            if lgb_auc >= xgb_auc:
                best_auc = lgb_auc
                best_model = 'LightGBM'
                best_tower = row.get('lightgbm_best_auc_roc_tower', 'N/A')
            else:
                best_auc = xgb_auc
                best_model = 'XGBoost'
                best_tower = row.get('xgboost_best_auc_roc_tower', 'N/A')
            
            report_lines.append(f"{event_name:<30} {row['avg_event_rate']*100:>12.2f}%  {best_auc:>10.4f}  {best_model:<12} {best_tower:<10}")
        
        report_lines.append("")
    
    if summary_df is not None and 'best_tower_event_df' in dir():
        report_lines.append("\n>> DETAILED MODEL PERFORMANCE BY TOWER-EVENT:")
        report_lines.append("-" * 100)
        report_lines.append(f"{'Tower':<8} {'Event':<25} {'Event Rate':<12} {'Best AUC':<10} {'Best Model':<12} {'Accuracy':<10}")
        report_lines.append("-" * 100)
        
        for _, row in best_tower_event_df.iterrows():
            event_name = row['event'].replace('event_', '').replace('_', ' ')[:22]
            best_model = row.get('best_model', 'N/A')
            best_auc = row.get('best_auc', 0)
            
            # Get accuracy from best model
            if best_model == 'lightgbm':
                accuracy = row.get('lightgbm_accuracy', 0)
            else:
                accuracy = row.get('xgboost_accuracy', 0)
            
            report_lines.append(f"{row['tower']:<8} {event_name:<25} {row['event_rate']*100:>10.2f}%  {best_auc:>8.4f}  {best_model:<12} {accuracy*100:>8.2f}%")
        
        report_lines.append("")
    
    # =========================================================================
    # SECTION 1: GLOBAL TOP FEATURES
    # =========================================================================
    report_lines.append("=" * 80)
    report_lines.append("FEATURE IMPORTANCE: GLOBAL TOP FEATURES")
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
        std_val = row['std'] if pd.notna(row['std']) else 0
        report_lines.append(f"{rank:<5} {row['feature']:<40} {row['mean']:>8.3f}  {std_val:>8.3f}  {row['min']:>8.3f}  {row['max']:>8.3f}")
    
    # =========================================================================
    # SECTION 2: TOP FEATURES PER EVENT
    # =========================================================================
    report_lines.append("\n")
    report_lines.append("=" * 80)
    report_lines.append("FEATURE IMPORTANCE: TOP FEATURES PER EVENT")
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
    report_lines.append("FEATURE IMPORTANCE: TOP FEATURES PER TOWER")
    report_lines.append("=" * 80)
    
    for tower in towers:
        tower_info = TOWER_COORDINATES.get(tower, {})
        terrain = tower_info.get('terrain_type', 'Unknown')
        elev = tower_info.get('elevation_m', 'N/A')
        
        report_lines.append(f"\n>> TOWER: {tower} ({terrain}, {elev}m)")
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
    # SECTION: VARIABLE CATEGORY ANALYSIS
    # =========================================================================
    report_lines.append("\n")
    report_lines.append("=" * 80)
    report_lines.append("VARIABLE CATEGORY IMPORTANCE SUMMARY")
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
        std_val = row['std'] if pd.notna(row['std']) else 0
        report_lines.append(f"{row['category']:<25} {row['total']:>10.2f}  {row['mean']:>10.3f}  {std_val:>10.3f}  {int(row['count']):>8}")
    
    # =========================================================================
    # SPATIAL ANALYSIS
    # =========================================================================
    report_lines.append("\n")
    report_lines.append("=" * 80)
    report_lines.append("SPATIAL ANALYSIS: GEOGRAPHIC FEATURES vs IMPORTANCE")
    report_lines.append("=" * 80)
    
    tower_importance_data = []
    for tower in towers:
        if tower not in TOWER_COORDINATES:
            continue
        tower_df = df[df['tower'] == tower]
        top5_sum = tower_df.nsmallest(5, 'rank')['importance_pct'].sum()
        coords = TOWER_COORDINATES[tower]
        tower_importance_data.append({
            'tower': tower,
            'terrain': coords.get('terrain_type', 'Unknown'),
            'lat': coords.get('lat', 0),
            'lon': coords.get('lon', 0),
            'elevation': coords.get('elevation_m', 0),
            'height': coords.get('height_m', 0),
            'total_height': coords.get('elevation_m', 0) + coords.get('height_m', 0),
            'slope': coords.get('slope_deg', 0),
            'canopy': coords.get('canopy_cover_pct', 0),
            'top5_importance': top5_sum
        })
    
    if tower_importance_data:
        t_df = pd.DataFrame(tower_importance_data)
        t_df = t_df.sort_values('top5_importance', ascending=False)
        
        report_lines.append(f"\n{'Rank':<6} {'Tower':<8} {'Terrain':<15} {'Elev(m)':<9} {'TowHt(m)':<10} {'Total(m)':<10} {'Slope':<8} {'Top5 Imp%':<12}")
        report_lines.append("-" * 95)
        for rank, (_, row) in enumerate(t_df.iterrows(), 1):
            report_lines.append(f"{rank:<6} {row['tower']:<8} {row['terrain']:<15} {row['elevation']:<9.0f} {row['height']:<10.0f} {row['total_height']:<10.0f} {row['slope']:<6.1f}°  {row['top5_importance']:>10.2f}")
        
        # Correlations with geographic features including tower height
        report_lines.append("\nCorrelations (Spearman) with Top 5 Feature Importance:")
        report_lines.append("-" * 60)
        
        for feature_name, feature_col in [('Ground Elevation', 'elevation'), 
                                           ('Tower Height', 'height'),
                                           ('Total Height (Ground+Tower)', 'total_height'),
                                           ('Slope', 'slope'), 
                                           ('Canopy Cover', 'canopy')]:
            if len(t_df) >= 3:
                corr, pval = spearmanr(t_df[feature_col], t_df['top5_importance'])
                sig = "***" if pval < 0.001 else "**" if pval < 0.01 else "*" if pval < 0.05 else "ns"
                report_lines.append(f"  {feature_name:<30}: r = {corr:>6.3f}, p = {pval:.4f} {sig}")
    
    # =========================================================================
    # KEY FINDINGS
    # =========================================================================
    report_lines.append("\n")
    report_lines.append("=" * 80)
    report_lines.append("KEY FINDINGS & HIGHLIGHTS")
    report_lines.append("=" * 80)
    
    # Most important feature
    top_feature = global_importance.iloc[0]
    report_lines.append(f"\n1. MOST IMPORTANT FEATURE: {top_feature['feature']}")
    report_lines.append(f"   Mean importance: {top_feature['mean']:.3f}%")
    
    # Most important category
    top_category = cat_summary.iloc[0]
    report_lines.append(f"\n2. MOST IMPORTANT CATEGORY: {top_category['category']}")
    report_lines.append(f"   Total importance: {top_category['total']:.2f}%")
    
    # Event-specific top features
    report_lines.append(f"\n3. TOP FEATURE BY EVENT:")
    for event in events:
        event_display = event.replace('_', ' ').replace('lt', '<').replace('gt', '>')
        event_df = df[df['event'] == event]
        top_feat = event_df.groupby('feature')['importance_pct'].mean().idxmax()
        top_val = event_df.groupby('feature')['importance_pct'].mean().max()
        report_lines.append(f"   - {event_display}: {top_feat} ({top_val:.2f}%)")
    
    # Geographic highlights including tower height
    if tower_importance_data:
        report_lines.append(f"\n4. GEOGRAPHIC & TOWER HIGHLIGHTS:")
        highest_elev = max(tower_importance_data, key=lambda x: x['elevation'])
        lowest_elev = min(tower_importance_data, key=lambda x: x['elevation'])
        tallest_tower = max(tower_importance_data, key=lambda x: x['height'])
        shortest_tower = min(tower_importance_data, key=lambda x: x['height'])
        highest_total = max(tower_importance_data, key=lambda x: x['total_height'])
        highest_imp = max(tower_importance_data, key=lambda x: x['top5_importance'])
        
        report_lines.append(f"   Ground Elevation:")
        report_lines.append(f"     - Highest: {highest_elev['tower']} ({highest_elev['elevation']:.0f}m, {highest_elev['terrain']})")
        report_lines.append(f"     - Lowest: {lowest_elev['tower']} ({lowest_elev['elevation']:.0f}m, {lowest_elev['terrain']})")
        report_lines.append(f"   Tower Height:")
        report_lines.append(f"     - Tallest: {tallest_tower['tower']} ({tallest_tower['height']:.0f}m)")
        report_lines.append(f"     - Shortest: {shortest_tower['tower']} ({shortest_tower['height']:.0f}m)")
        report_lines.append(f"   Total Height (Ground + Tower):")
        report_lines.append(f"     - Highest: {highest_total['tower']} ({highest_total['total_height']:.0f}m)")
        report_lines.append(f"   Feature Importance:")
        report_lines.append(f"     - Highest Top5 importance: {highest_imp['tower']} ({highest_imp['top5_importance']:.2f}%)")
    
    # Footer
    report_lines.append("\n")
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


def run_all_visualizations(df: pd.DataFrame, folder_path: str):
    """Execute ALL visualizations and reports automatically."""
    
    towers = sorted(df['tower'].unique())
    events = sorted(df['event'].unique())
    
    # Create output folder for saved figures
    output_folder = os.path.join(folder_path, 'viz_importance')
    os.makedirs(output_folder, exist_ok=True)
    
    print("\n" + "="*60)
    print("       FEATURE IMPORTANCE VISUALIZATION TOOL")
    print("       >>> RUNNING ALL VISUALIZATIONS <<<")
    print("="*60)
    print(f"\nData loaded: {len(towers)} towers, {len(events)} events")
    print(f"Towers: {', '.join(towers)}")
    print(f"Events: {', '.join(events)}")
    print(f"Output folder: {output_folder}")
    print("="*60)
    
    # =====================================================
    # 1. TOP FEATURES PER TOWER & EVENT (Individual plots)
    # =====================================================
    print("\n" + "-"*60)
    print("[1/13] Generating top features for each tower & event...")
    print("-"*60)
    for tower in towers:
        for event in events:
            save_path = os.path.join(output_folder, f'top_features_{tower}_{event}.png')
            try:
                plot_top_features_by_tower_event(df, tower, event, top_n=20, save_path=save_path)
                print(f"  ✓ {tower} - {event}")
            except Exception as e:
                print(f"  ✗ {tower} - {event}: {e}")
    
    # =====================================================
    # 2. HEATMAPS - Compare features across towers per event
    # =====================================================
    print("\n" + "-"*60)
    print("[2/13] Generating heatmaps (features across towers)...")
    print("-"*60)
    for event in events:
        save_path = os.path.join(output_folder, f'heatmap_{event}.png')
        try:
            plot_feature_comparison_across_towers(df, event, top_n=20, save_path=save_path)
            print(f"  ✓ Heatmap for {event}")
        except Exception as e:
            print(f"  ✗ Heatmap for {event}: {e}")
    
    # =====================================================
    # 3. COMPARE FEATURES ACROSS EVENTS (per tower)
    # =====================================================
    print("\n" + "-"*60)
    print("[3/13] Generating cross-event comparisons per tower...")
    print("-"*60)
    for tower in towers:
        save_path = os.path.join(output_folder, f'cross_events_{tower}.png')
        try:
            plot_feature_comparison_across_events(df, tower, top_n=15, save_path=save_path)
            print(f"  ✓ Cross-events for {tower}")
        except Exception as e:
            print(f"  ✗ Cross-events for {tower}: {e}")
    
    # =====================================================
    # 4. GLOBAL FEATURE IMPORTANCE
    # =====================================================
    print("\n" + "-"*60)
    print("[4/13] Generating global feature importance...")
    print("-"*60)
    save_path = os.path.join(output_folder, 'global_feature_importance.png')
    try:
        plot_global_feature_importance(df, top_n=25, save_path=save_path)
        print(f"  ✓ Global feature importance")
    except Exception as e:
        print(f"  ✗ Global feature importance: {e}")
    
    # =====================================================
    # 5. VARIABLE CATEGORY IMPORTANCE
    # =====================================================
    print("\n" + "-"*60)
    print("[5/13] Generating variable category importance...")
    print("-"*60)
    save_path = os.path.join(output_folder, 'variable_category_importance.png')
    try:
        plot_variable_category_importance(df, save_path=save_path)
        print(f"  ✓ Variable category importance")
    except Exception as e:
        print(f"  ✗ Variable category importance: {e}")
    
    # =====================================================
    # 6. SUMMARY DASHBOARD
    # =====================================================
    print("\n" + "-"*60)
    print("[6/13] Generating summary dashboard...")
    print("-"*60)
    try:
        create_summary_dashboard(df, output_folder=output_folder)
        print(f"  ✓ Summary dashboard")
    except Exception as e:
        print(f"  ✗ Summary dashboard: {e}")
    
    # =====================================================
    # 7. ALL BASIC VISUALIZATIONS (already covered above)
    # =====================================================
    # Skipping - already done in steps 1-6
    
    # =====================================================
    # 8. ALL TOWERS PER EVENT - Grid subplots
    # =====================================================
    print("\n" + "-"*60)
    print("[8/13] Generating ALL TOWERS grid subplots per event...")
    print("-"*60)
    for event in events:
        save_path = os.path.join(output_folder, f'ALL_TOWERS_{event}_grid.png')
        try:
            plot_all_towers_per_event(df, event, top_n=15, save_path=save_path)
            print(f"  ✓ Grid for {event}")
        except Exception as e:
            print(f"  ✗ Grid for {event}: {e}")
    
    # =====================================================
    # 9. ALL TOWERS PER EVENT - Dot plots (clustered)
    # =====================================================
    print("\n" + "-"*60)
    print("[9/13] Generating ALL TOWERS dot plots per event...")
    print("-"*60)
    for event in events:
        save_path = os.path.join(output_folder, f'ALL_TOWERS_{event}_dotplot.png')
        try:
            plot_grouped_bars_per_event(df, event, top_n=10, save_path=save_path)
            print(f"  ✓ Dot plot for {event}")
        except Exception as e:
            print(f"  ✗ Dot plot for {event}: {e}")
    
    # =====================================================
    # 10. MASTER VIEW - All towers x All events
    # =====================================================
    print("\n" + "-"*60)
    print("[10/13] Generating MASTER VIEW (all towers x all events)...")
    print("-"*60)
    save_path = os.path.join(output_folder, 'MASTER_all_towers_all_events.png')
    try:
        plot_consolidated_all_events(df, top_n=15, save_path=save_path)
        print(f"  ✓ MASTER VIEW")
    except Exception as e:
        print(f"  ✗ MASTER VIEW: {e}")
    
    # =====================================================
    # 11. GEOGRAPHIC FEATURES VS IMPORTANCE ANALYSIS
    # =====================================================
    print("\n" + "-"*60)
    print("[11/13] Generating Geographic Features vs Importance Analysis...")
    print("-"*60)
    try:
        plot_geographic_features_vs_importance(df, output_folder=output_folder)
        print(f"  ✓ Geographic features analysis")
    except Exception as e:
        print(f"  ✗ Geographic features analysis: {e}")
    
    # =====================================================
    # 12. 3D TERRAIN-IMPORTANCE VISUALIZATION
    # =====================================================
    print("\n" + "-"*60)
    print("[12/13] Generating 3D Terrain-Importance Visualizations...")
    print("-"*60)
    for event in events:
        try:
            plot_3d_terrain_importance(df, event=event, output_folder=output_folder)
            print(f"  ✓ 3D terrain for {event}")
        except Exception as e:
            print(f"  ✗ 3D terrain for {event}: {e}")
    
    # =====================================================
    # 13. STATISTICAL REPORT
    # =====================================================
    print("\n" + "-"*60)
    print("[13/13] Generating Statistical Report...")
    print("-"*60)
    try:
        report = generate_statistical_report(df, output_folder, folder_path)
        print(f"  ✓ Statistical report generated")
        print("\n" + "="*60)
        print("REPORT PREVIEW (first 100 lines):")
        print("="*60)
        for line in report.split('\n')[:100]:
            print(line)
        print("\n... (see full report in output folder)")
    except Exception as e:
        print(f"  ✗ Statistical report: {e}")
    
    # =====================================================
    # COMPLETION SUMMARY
    # =====================================================
    print("\n" + "="*60)
    print("       ALL VISUALIZATIONS COMPLETED!")
    print("="*60)
    print(f"\nOutput folder: {output_folder}")
    
    # Count generated files
    import glob
    png_files = glob.glob(os.path.join(output_folder, '*.png'))
    txt_files = glob.glob(os.path.join(output_folder, '*.txt'))
    print(f"Generated {len(png_files)} PNG files and {len(txt_files)} TXT reports")
    print("="*60)


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
        print(" 11. [NEW] Geographic Features vs Importance Analysis")
        print(" 12. [NEW] 3D Terrain-Importance Visualization")
        print("-"*60)
        print("  === REPORTS ===")
        print(" 13. Generate Statistical Report (model perf from summary_results.csv)")
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
            # Geographic Feature Analysis (Top 5 features per tower against terrain)
            print("\nGenerating Geographic Feature Analysis plots (one per event)...")
            
            save = input("Save figures? (y/n): ").strip().lower() == 'y'
            save_folder = output_folder if save else None
            
            plot_geographic_features_vs_importance(df, output_folder=save_folder)
        
        elif choice == '12':
            # 3D Terrain-Importance Visualization
            print(f"\nAvailable events: {', '.join(events)}")
            event = input("Enter event name (or 'all' for all events): ").strip()
            
            save = input("Save figures? (y/n): ").strip().lower() == 'y'
            save_folder = output_folder if save else None
            
            if event.lower() == 'all':
                for evt in events:
                    print(f"\n--- Plotting 3D terrain for {evt} ---")
                    plot_3d_terrain_importance(df, event=evt, output_folder=save_folder)
            else:
                if event not in events:
                    print(f"Invalid event. Choose from: {events}")
                    continue
                plot_3d_terrain_importance(df, event=event, output_folder=save_folder)
        
        elif choice == '13':
            # Generate Statistical Report
            print("\nGenerating Statistical Report with model performance from summary_results.csv...")
            report = generate_statistical_report(df, output_folder, folder_path)
            print("\n" + "="*60)
            print("Report Preview (first 80 lines):")
            print("="*60)
            for line in report.split('\n')[:80]:
                print(line)
            print("\n... (see full report in output folder)")
            
        else:
            print("Invalid choice. Please enter 0-13.")


def main():
    """Main entry point."""
    if len(sys.argv) < 2:
        print("Usage: python viz_importance.py <results_folder_path> [tower_metadata.json]")
        print("Example: python viz_importance.py BESTML_multi_event_results_20251208_155338")
        print("         python viz_importance.py BESTML_results tower_metadata_generated.json")
        sys.exit(1)
    
    folder_path = sys.argv[1]
    json_path = sys.argv[2] if len(sys.argv) > 2 else None
    
    # Handle relative paths
    if not os.path.isabs(folder_path):
        folder_path = os.path.join(os.getcwd(), folder_path)
    
    if json_path and not os.path.isabs(json_path):
        json_path = os.path.join(os.getcwd(), json_path)
    
    if not os.path.isdir(folder_path):
        print(f"Error: Folder not found: {folder_path}")
        sys.exit(1)
    
    print(f"\n{'='*60}")
    print("  Feature Importance Visualization Tool")
    print(f"{'='*60}")
    print(f"Results Folder: {folder_path}\n")
    
    # Load tower metadata from JSON
    print("-"*60)
    print("Loading Tower Metadata from JSON...")
    print("-"*60)
    load_tower_metadata(json_path=json_path, folder_path=folder_path)
    print()
    
    # Load importance data
    print("-"*60)
    print("Loading Feature Importance Data...")
    print("-"*60)
    data = load_importance_files(folder_path)
    
    if not data:
        print("No data to visualize.")
        sys.exit(1)
    
    # Combine into single dataframe
    df = get_combined_dataframe(data)
    print(f"\nTotal records loaded: {len(df)}")
    
    # Run ALL visualizations automatically (no menu)
    run_all_visualizations(df, folder_path)


if __name__ == "__main__":
    main()

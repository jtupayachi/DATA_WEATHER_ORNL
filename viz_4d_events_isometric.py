#!/usr/bin/env python3
"""
4D Isometric Extreme Weather Events Visualization
=================================================
Creates neural-network-style isometric projections showing extreme weather
event occurrences across towers over time.

Each layer = one time period
Events shown as colored markers on terrain
More layers for finer temporal resolution

Author: Auto-generated
Date: December 2024
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize, LightSource, ListedColormap
from matplotlib.patches import Patch, Polygon, Circle, FancyBboxPatch
from matplotlib.lines import Line2D
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
import warnings
warnings.filterwarnings('ignore')

try:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
except ImportError:
    HAS_PLOTLY = False

# =============================================================================
# CONFIGURATION
# =============================================================================

OUTPUT_DIR = 'viz_4d_events_isometric'
DATA_FILE = '/home/jose/DATA_WEATHER_ORNL/fully_labeled_weather_data_with_events.csv'

TOWER_COORDINATES = {
    'TOWA': {'lat': 35.9312, 'lon': -84.3108, 'elevation_m': 247, 'color': '#e41a1c', 'name': 'Tower A'},
    'TOWB': {'lat': 35.9285, 'lon': -84.3045, 'elevation_m': 278, 'color': '#377eb8', 'name': 'Tower B'},
    'TOWD': {'lat': 35.935, 'lon': -84.32, 'elevation_m': 287, 'color': '#4daf4a', 'name': 'Tower D'},
    'TOWF': {'lat': 35.922, 'lon': -84.315, 'elevation_m': 274, 'color': '#984ea3', 'name': 'Tower F'},
    'TOWS': {'lat': 35.938, 'lon': -84.298, 'elevation_m': 280, 'color': '#ff7f00', 'name': 'Tower S'},
    'TOWY': {'lat': 35.9255, 'lon': -84.325, 'elevation_m': 262, 'color': '#a65628', 'name': 'Tower Y'},
}

# Event type definitions with distinct visual styles
EVENT_TYPES = {
    'event_E3_LowTemp_lt0': {
        'name': 'Low Temperature (<0°C)',
        'short': 'Cold',
        'color': '#2166ac',  # Blue
        'marker': 'v',  # Down triangle (cold = down)
        'marker_alt': '❄',
    },
    'event_E4_HighWind_Peak_gt25': {
        'name': 'High Wind (Peak >25 mph)',
        'short': 'Wind',
        'color': '#d62728',  # Red
        'marker': '^',  # Up triangle (wind = up)
        'marker_alt': '💨',
    },
    'event_E5_LowWind_lt2': {
        'name': 'Low Wind (<2 mph)',
        'short': 'Calm',
        'color': '#2ca02c',  # Green
        'marker': 's',  # Square (stable/calm)
        'marker_alt': '○',
    },
}


def load_data(filepath=DATA_FILE, sample_size=None):
    """Load weather data with events."""
    print(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath, parse_dates=['timestamp'])
    print(f"  Total rows: {len(df):,}")
    print(f"  Date range: {df['timestamp'].min()} to {df['timestamp'].max()}")
    
    if sample_size and len(df) > sample_size:
        print(f"  Sampling {sample_size:,} rows...")
        df = df.sample(n=sample_size, random_state=42).sort_values('timestamp')
    
    return df


def generate_terrain_grid(resolution=40, padding_km=0.5):
    """Generate terrain elevation grid."""
    lats = [t['lat'] for t in TOWER_COORDINATES.values()]
    lons = [t['lon'] for t in TOWER_COORDINATES.values()]
    
    lat_padding = padding_km / 111.0
    lon_padding = padding_km / (111.0 * np.cos(np.radians(np.mean(lats))))
    
    bounds = {
        'min_lat': min(lats) - lat_padding, 'max_lat': max(lats) + lat_padding,
        'min_lon': min(lons) - lon_padding, 'max_lon': max(lons) + lon_padding,
    }
    
    lon = np.linspace(bounds['min_lon'], bounds['max_lon'], resolution)
    lat = np.linspace(bounds['min_lat'], bounds['max_lat'], resolution)
    LON, LAT = np.meshgrid(lon, lat)
    
    tower_points = [(t['lon'], t['lat']) for t in TOWER_COORDINATES.values()]
    tower_elevs = [t['elevation_m'] for t in TOWER_COORDINATES.values()]
    
    elevation = griddata(tower_points, tower_elevs, (LON, LAT),
                        method='cubic', fill_value=np.mean(tower_elevs))
    
    # Add terrain variations
    ridge_angle = np.radians(45)
    lon_center, lat_center = np.mean(lon), np.mean(lat)
    lon_m = (LON - lon_center) * 111000 * np.cos(np.radians(lat_center))
    lat_m = (LAT - lat_center) * 111000
    y_rot = -lon_m * np.sin(ridge_angle) + lat_m * np.cos(ridge_angle)
    ridge_pattern = 12 * np.sin(2 * np.pi * y_rot / 1000)
    
    np.random.seed(42)
    roughness = gaussian_filter(np.random.randn(resolution, resolution) * 2, sigma=2)
    
    elevation = gaussian_filter(elevation + ridge_pattern + roughness, sigma=1)
    
    return lon, lat, elevation, bounds


def isometric_transform(x, y, z, angle=45, scale_z=0.5, depth_offset=0):
    """Transform 3D coordinates to 2D isometric projection."""
    angle_rad = np.radians(angle)
    
    x_rot = x * np.cos(angle_rad) - y * np.sin(angle_rad)
    y_rot = x * np.sin(angle_rad) + y * np.cos(angle_rad)
    
    x_iso = x_rot + depth_offset * 0.7
    y_iso = y_rot * 0.5 + z * scale_z + depth_offset * 0.5
    
    return x_iso, y_iso


def create_events_isometric_monthly(df, n_months=12, output_path=None):
    """
    Create isometric visualization of extreme weather events by month.
    Each layer = one month, showing event occurrences at each tower.
    """
    print(f"\nCreating monthly events isometric visualization ({n_months} layers)...")
    
    # Generate terrain
    lon, lat, elevation, bounds = generate_terrain_grid(resolution=30)
    LON, LAT = np.meshgrid(lon, lat)
    
    # Normalize coordinates
    lon_norm = (LON - lon.min()) / (lon.max() - lon.min())
    lat_norm = (LAT - lat.min()) / (lat.max() - lat.min())
    elev_norm = (elevation - elevation.min()) / (elevation.max() - elevation.min())
    
    # Prepare event columns
    event_cols = list(EVENT_TYPES.keys())
    df_events = df[['timestamp', 'tower'] + event_cols].copy()
    
    # Create monthly bins
    df_events['month'] = df_events['timestamp'].dt.to_period('M')
    months = sorted(df_events['month'].unique())
    
    # Use last n_months
    if len(months) > n_months:
        months = months[-n_months:]
    
    n_slices = len(months)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(24, 20), facecolor='white')
    ax.set_facecolor('white')
    
    # Depth spacing
    depth_spacing = 0.28
    
    # Light source
    ls = LightSource(azdeg=315, altdeg=45)
    
    # Create terrain colormap (grayscale with slight tint)
    terrain_cmap = plt.cm.Greys_r
    
    # Track max events for sizing
    max_event_count = 1
    for month in months:
        month_data = df_events[df_events['month'] == month]
        for event_col in event_cols:
            counts = month_data.groupby('tower')[event_col].sum()
            if len(counts) > 0:
                max_event_count = max(max_event_count, counts.max())
    
    # Plot from back to front
    for slice_idx, month in enumerate(reversed(months)):
        actual_idx = n_slices - 1 - slice_idx
        depth = slice_idx * depth_spacing
        
        month_label = month.strftime('%Y-%m')
        
        # Get event data for this month
        month_data = df_events[df_events['month'] == month]
        
        # Transform terrain to isometric
        x_iso, y_iso = isometric_transform(lon_norm, lat_norm, elev_norm * 0.2,
                                           angle=45, scale_z=0.3, depth_offset=depth)
        
        # Calculate total events for this month (for layer coloring)
        total_events = sum(month_data[col].sum() for col in event_cols)
        
        # Color intensity based on event activity (white = no events, colored = events)
        if total_events > 0:
            activity_intensity = min(total_events / (len(TOWER_COORDINATES) * 100), 1.0)
            base_color = np.array([0.95, 0.95, 0.95])  # Light gray base
        else:
            activity_intensity = 0
            base_color = np.array([0.98, 0.98, 0.98])
        
        # Draw terrain surface
        shaded = ls.shade(elevation, cmap=terrain_cmap, blend_mode='soft')
        
        for i in range(len(lat) - 1):
            for j in range(len(lon) - 1):
                corners_x = [x_iso[i, j], x_iso[i, j+1], x_iso[i+1, j+1], x_iso[i+1, j]]
                corners_y = [y_iso[i, j], y_iso[i, j+1], y_iso[i+1, j+1], y_iso[i+1, j]]
                
                cell_color = shaded[i, j, :3] * 0.3 + base_color * 0.7
                
                poly = Polygon(list(zip(corners_x, corners_y)),
                              facecolor=cell_color, edgecolor='none',
                              alpha=0.92, zorder=actual_idx)
                ax.add_patch(poly)
        
        # Draw border
        border_x = [x_iso[0, 0], x_iso[0, -1], x_iso[-1, -1], x_iso[-1, 0], x_iso[0, 0]]
        border_y = [y_iso[0, 0], y_iso[0, -1], y_iso[-1, -1], y_iso[-1, 0], y_iso[0, 0]]
        ax.plot(border_x, border_y, 'k-', linewidth=1.2, alpha=0.6, zorder=actual_idx + 0.5)
        
        # 3D side panels
        right_x = [x_iso[0, -1], x_iso[-1, -1],
                   x_iso[-1, -1] + 0.015, x_iso[0, -1] + 0.015, x_iso[0, -1]]
        right_y = [y_iso[0, -1], y_iso[-1, -1],
                   y_iso[-1, -1] - 0.04, y_iso[0, -1] - 0.04, y_iso[0, -1]]
        ax.fill(right_x, right_y, color='#555555', alpha=0.5, zorder=actual_idx - 0.1)
        
        bottom_x = [x_iso[-1, 0], x_iso[-1, -1],
                    x_iso[-1, -1] + 0.015, x_iso[-1, 0] + 0.015, x_iso[-1, 0]]
        bottom_y = [y_iso[-1, 0], y_iso[-1, -1],
                    y_iso[-1, -1] - 0.04, y_iso[-1, 0] - 0.04, y_iso[-1, 0]]
        ax.fill(bottom_x, bottom_y, color='#777777', alpha=0.5, zorder=actual_idx - 0.1)
        
        # Month label
        label_x = x_iso[-1, -1] + 0.06
        label_y = y_iso[-1, -1] - 0.01
        
        # Color label by event intensity
        label_bg = 'lightyellow' if total_events > 50 else 'white'
        ax.text(label_x, label_y, f'{month_label}',
               fontsize=9, fontweight='bold', ha='left', va='center',
               bbox=dict(facecolor=label_bg, alpha=0.9, edgecolor='gray',
                        boxstyle='round,pad=0.25'),
               zorder=1000)
        
        # Plot tower positions and events
        for tower, coords in TOWER_COORDINATES.items():
            t_lon_norm = (coords['lon'] - lon.min()) / (lon.max() - lon.min())
            t_lat_norm = (coords['lat'] - lat.min()) / (lat.max() - lat.min())
            
            lon_idx = np.argmin(np.abs(lon - coords['lon']))
            lat_idx = np.argmin(np.abs(lat - coords['lat']))
            t_elev_norm = elev_norm[lat_idx, lon_idx]
            
            t_x, t_y = isometric_transform(t_lon_norm, t_lat_norm, t_elev_norm * 0.2 + 0.03,
                                           angle=45, scale_z=0.3, depth_offset=depth)
            
            # Get event counts for this tower/month
            tower_data = month_data[month_data['tower'] == tower]
            
            # Base tower marker (small dot)
            ax.scatter([t_x], [t_y], s=25, c='gray', marker='o',
                      edgecolors='white', linewidth=0.5, alpha=0.7,
                      zorder=actual_idx + 50)
            
            # Plot each event type as stacked/offset markers
            event_offset = 0
            for event_col, event_info in EVENT_TYPES.items():
                event_count = tower_data[event_col].sum() if len(tower_data) > 0 else 0
                
                if event_count > 0:
                    # Size based on count (log scale for visibility)
                    marker_size = 40 + np.log1p(event_count) * 25
                    
                    # Offset vertically for multiple event types
                    offset_y = event_offset * 0.012
                    
                    ax.scatter([t_x], [t_y + offset_y], s=marker_size,
                              c=event_info['color'], marker=event_info['marker'],
                              edgecolors='white', linewidth=1,
                              alpha=0.85, zorder=actual_idx + 100 + event_offset)
                    
                    event_offset += 1
            
            # Tower labels on first layer only
            if slice_idx == n_slices - 1:
                ax.text(t_x + 0.015, t_y + 0.015, tower,
                       fontsize=7, fontweight='bold',
                       bbox=dict(facecolor='white', alpha=0.85,
                                edgecolor=coords['color'], boxstyle='round,pad=0.15'),
                       zorder=1001)
    
    # Time axis arrow
    arrow_start_x, arrow_start_y = 0.05, 0.15
    arrow_end_x = arrow_start_x + (n_slices - 1) * depth_spacing * 0.7
    arrow_end_y = arrow_start_y + (n_slices - 1) * depth_spacing * 0.5
    
    ax.annotate('', xy=(arrow_end_x, arrow_end_y), xytext=(arrow_start_x, arrow_start_y),
               arrowprops=dict(arrowstyle='->', color='darkred', lw=2.5),
               zorder=1002)
    ax.text((arrow_start_x + arrow_end_x) / 2 - 0.08,
            (arrow_start_y + arrow_end_y) / 2 + 0.04,
           'TIME →', fontsize=13, fontweight='bold', color='darkred',
           rotation=35, zorder=1002)
    
    # Legend for event types
    legend_elements = []
    for event_col, event_info in EVENT_TYPES.items():
        legend_elements.append(
            Line2D([0], [0], marker=event_info['marker'], color='w',
                  markerfacecolor=event_info['color'], markersize=12,
                  label=event_info['name'], markeredgecolor='white', markeredgewidth=1)
        )
    
    # Add size legend
    legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor='gray', markersize=6,
                                  label='Small = few events'))
    legend_elements.append(Line2D([0], [0], marker='o', color='w',
                                  markerfacecolor='gray', markersize=14,
                                  label='Large = many events'))
    
    ax.legend(handles=legend_elements, loc='upper left', fontsize=10,
             framealpha=0.95, title='Event Types', title_fontsize=11)
    
    ax.set_title(
        f'4D Isometric View: Extreme Weather Events\n'
        f'{n_slices} Monthly Layers | Each marker = event occurrence at tower\n'
        f'Marker size ∝ event count | Multiple markers = multiple event types',
        fontsize=16, fontweight='bold', pad=20
    )
    
    ax.set_xlim(-0.3, 3.5)
    ax.set_ylim(-0.3, 2.8)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
    
    plt.show()
    plt.close()


def create_events_isometric_weekly(df, n_weeks=24, output_path=None):
    """
    Create isometric visualization with weekly granularity.
    More layers = finer temporal resolution for event patterns.
    """
    print(f"\nCreating weekly events isometric visualization ({n_weeks} layers)...")
    
    # Generate terrain
    lon, lat, elevation, bounds = generate_terrain_grid(resolution=25)
    LON, LAT = np.meshgrid(lon, lat)
    
    lon_norm = (LON - lon.min()) / (lon.max() - lon.min())
    lat_norm = (LAT - lat.min()) / (lat.max() - lat.min())
    elev_norm = (elevation - elevation.min()) / (elevation.max() - elevation.min())
    
    # Prepare event data
    event_cols = list(EVENT_TYPES.keys())
    df_events = df[['timestamp', 'tower'] + event_cols].copy()
    df_events['week'] = df_events['timestamp'].dt.to_period('W')
    weeks = sorted(df_events['week'].unique())
    
    if len(weeks) > n_weeks:
        weeks = weeks[-n_weeks:]
    
    n_slices = len(weeks)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(28, 22), facecolor='white')
    ax.set_facecolor('white')
    
    # Tighter spacing for more layers
    depth_spacing = 0.18
    
    ls = LightSource(azdeg=315, altdeg=45)
    terrain_cmap = plt.cm.Greys_r
    
    # Plot layers
    for slice_idx, week in enumerate(reversed(weeks)):
        actual_idx = n_slices - 1 - slice_idx
        depth = slice_idx * depth_spacing
        
        week_label = week.start_time.strftime('%m/%d')
        
        week_data = df_events[df_events['week'] == week]
        
        x_iso, y_iso = isometric_transform(lon_norm, lat_norm, elev_norm * 0.15,
                                           angle=45, scale_z=0.25, depth_offset=depth)
        
        # Simplified terrain drawing for more layers
        shaded = ls.shade(elevation, cmap=terrain_cmap, blend_mode='soft')
        base_color = np.array([0.96, 0.96, 0.96])
        
        # Draw every other cell for speed with many layers
        step = 2
        for i in range(0, len(lat) - step, step):
            for j in range(0, len(lon) - step, step):
                corners_x = [x_iso[i, j], x_iso[i, j+step], x_iso[i+step, j+step], x_iso[i+step, j]]
                corners_y = [y_iso[i, j], y_iso[i, j+step], y_iso[i+step, j+step], y_iso[i+step, j]]
                
                cell_color = shaded[i, j, :3] * 0.25 + base_color * 0.75
                
                poly = Polygon(list(zip(corners_x, corners_y)),
                              facecolor=cell_color, edgecolor='none',
                              alpha=0.88, zorder=actual_idx)
                ax.add_patch(poly)
        
        # Border
        border_x = [x_iso[0, 0], x_iso[0, -1], x_iso[-1, -1], x_iso[-1, 0], x_iso[0, 0]]
        border_y = [y_iso[0, 0], y_iso[0, -1], y_iso[-1, -1], y_iso[-1, 0], y_iso[0, 0]]
        ax.plot(border_x, border_y, 'k-', linewidth=0.8, alpha=0.4, zorder=actual_idx + 0.5)
        
        # Side panels (thinner)
        right_x = [x_iso[0, -1], x_iso[-1, -1],
                   x_iso[-1, -1] + 0.01, x_iso[0, -1] + 0.01]
        right_y = [y_iso[0, -1], y_iso[-1, -1],
                   y_iso[-1, -1] - 0.025, y_iso[0, -1] - 0.025]
        ax.fill(right_x, right_y, color='#666666', alpha=0.4, zorder=actual_idx - 0.1)
        
        # Week label (only every 4th week to avoid clutter)
        if slice_idx % 4 == 0:
            label_x = x_iso[-1, -1] + 0.04
            label_y = y_iso[-1, -1]
            ax.text(label_x, label_y, week_label,
                   fontsize=7, fontweight='bold', ha='left', va='center',
                   bbox=dict(facecolor='white', alpha=0.8, edgecolor='gray',
                            boxstyle='round,pad=0.2'),
                   zorder=1000)
        
        # Plot events
        for tower, coords in TOWER_COORDINATES.items():
            t_lon_norm = (coords['lon'] - lon.min()) / (lon.max() - lon.min())
            t_lat_norm = (coords['lat'] - lat.min()) / (lat.max() - lat.min())
            
            lon_idx = np.argmin(np.abs(lon - coords['lon']))
            lat_idx = np.argmin(np.abs(lat - coords['lat']))
            t_elev_norm = elev_norm[lat_idx, lon_idx]
            
            t_x, t_y = isometric_transform(t_lon_norm, t_lat_norm, t_elev_norm * 0.15 + 0.02,
                                           angle=45, scale_z=0.25, depth_offset=depth)
            
            tower_data = week_data[week_data['tower'] == tower]
            
            # Small base marker
            ax.scatter([t_x], [t_y], s=12, c='lightgray', marker='.',
                      zorder=actual_idx + 50)
            
            # Event markers
            event_offset = 0
            for event_col, event_info in EVENT_TYPES.items():
                event_count = tower_data[event_col].sum() if len(tower_data) > 0 else 0
                
                if event_count > 0:
                    marker_size = 25 + np.log1p(event_count) * 15
                    offset_y = event_offset * 0.008
                    
                    ax.scatter([t_x], [t_y + offset_y], s=marker_size,
                              c=event_info['color'], marker=event_info['marker'],
                              edgecolors='white', linewidth=0.5,
                              alpha=0.8, zorder=actual_idx + 100 + event_offset)
                    
                    event_offset += 1
            
            # Tower label on front layer
            if slice_idx == n_slices - 1:
                ax.text(t_x + 0.01, t_y + 0.01, tower,
                       fontsize=6, fontweight='bold',
                       bbox=dict(facecolor='white', alpha=0.8,
                                edgecolor=coords['color'], boxstyle='round,pad=0.1'),
                       zorder=1001)
    
    # Time arrow
    arrow_start_x, arrow_start_y = 0.02, 0.1
    arrow_end_x = arrow_start_x + (n_slices - 1) * depth_spacing * 0.7
    arrow_end_y = arrow_start_y + (n_slices - 1) * depth_spacing * 0.5
    
    ax.annotate('', xy=(arrow_end_x, arrow_end_y), xytext=(arrow_start_x, arrow_start_y),
               arrowprops=dict(arrowstyle='->', color='darkred', lw=2),
               zorder=1002)
    ax.text((arrow_start_x + arrow_end_x) / 2 - 0.06,
            (arrow_start_y + arrow_end_y) / 2 + 0.03,
           'TIME →', fontsize=12, fontweight='bold', color='darkred',
           rotation=35, zorder=1002)
    
    # Legend
    legend_elements = [
        Line2D([0], [0], marker=info['marker'], color='w',
              markerfacecolor=info['color'], markersize=10,
              label=info['short'], markeredgecolor='white')
        for info in EVENT_TYPES.values()
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9,
             framealpha=0.9, title='Events', title_fontsize=10)
    
    ax.set_title(
        f'4D Isometric: Weekly Extreme Weather Events\n'
        f'{n_slices} Weekly Layers | High-resolution temporal view',
        fontsize=15, fontweight='bold', pad=20
    )
    
    ax.set_xlim(-0.3, 5.0)
    ax.set_ylim(-0.3, 3.5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
    
    plt.show()
    plt.close()


def create_events_heatmap_isometric(df, n_slices=18, output_path=None):
    """
    Create isometric visualization where each layer shows event intensity as a heatmap.
    Color intensity = total event count across all types.
    """
    print(f"\nCreating event heatmap isometric visualization ({n_slices} layers)...")
    
    # Generate terrain
    lon, lat, elevation, bounds = generate_terrain_grid(resolution=30)
    LON, LAT = np.meshgrid(lon, lat)
    
    lon_norm = (LON - lon.min()) / (lon.max() - lon.min())
    lat_norm = (LAT - lat.min()) / (lat.max() - lat.min())
    elev_norm = (elevation - elevation.min()) / (elevation.max() - elevation.min())
    
    # Prepare events
    event_cols = list(EVENT_TYPES.keys())
    df_events = df[['timestamp', 'tower'] + event_cols].copy()
    df_events['period'] = pd.cut(df_events['timestamp'], bins=n_slices, labels=False)
    
    # Get time labels
    time_bins = pd.cut(df_events['timestamp'], bins=n_slices)
    period_labels = {i: f"{cat.left.strftime('%Y-%m')}" for i, cat in 
                     enumerate(time_bins.cat.categories)}
    
    # Create figure
    fig, ax = plt.subplots(figsize=(26, 20), facecolor='white')
    ax.set_facecolor('white')
    
    depth_spacing = 0.22
    
    # Event intensity colormap (white to red)
    event_cmap = plt.cm.YlOrRd
    
    # Calculate max events for normalization
    max_events = 0
    for period in range(n_slices):
        period_data = df_events[df_events['period'] == period]
        for tower in TOWER_COORDINATES:
            tower_data = period_data[period_data['tower'] == tower]
            total = sum(tower_data[col].sum() for col in event_cols)
            max_events = max(max_events, total)
    
    ls = LightSource(azdeg=315, altdeg=45)
    
    # Plot layers
    for slice_idx in range(n_slices - 1, -1, -1):
        period = n_slices - 1 - slice_idx
        depth = slice_idx * depth_spacing
        
        period_label = period_labels.get(period, f'P{period}')
        period_data = df_events[df_events['period'] == period]
        
        x_iso, y_iso = isometric_transform(lon_norm, lat_norm, elev_norm * 0.15,
                                           angle=45, scale_z=0.25, depth_offset=depth)
        
        # Calculate event counts per tower
        tower_events = {}
        for tower in TOWER_COORDINATES:
            tower_data = period_data[period_data['tower'] == tower]
            tower_events[tower] = sum(tower_data[col].sum() for col in event_cols)
        
        # Interpolate event intensity across grid
        tower_points = [(TOWER_COORDINATES[t]['lon'], TOWER_COORDINATES[t]['lat'])
                       for t in TOWER_COORDINATES]
        tower_points_norm = [((p[0] - lon.min()) / (lon.max() - lon.min()),
                              (p[1] - lat.min()) / (lat.max() - lat.min()))
                            for p in tower_points]
        event_vals = [tower_events[t] for t in TOWER_COORDINATES]
        
        if max(event_vals) > 0:
            event_grid = griddata(tower_points_norm, event_vals, (lon_norm, lat_norm),
                                 method='cubic', fill_value=np.mean(event_vals))
            event_grid = np.clip(event_grid, 0, max_events)
        else:
            event_grid = np.zeros_like(lon_norm)
        
        # Color by event intensity
        event_norm = event_grid / max(max_events, 1)
        colors = event_cmap(event_norm * 0.8)  # Scale to avoid pure red
        
        # Blend with terrain hillshade
        shaded = ls.shade(elevation, cmap=plt.cm.gray, blend_mode='soft')
        colors[..., :3] = colors[..., :3] * 0.7 + shaded[..., :3] * 0.3
        
        # Draw terrain
        for i in range(len(lat) - 1):
            for j in range(len(lon) - 1):
                corners_x = [x_iso[i, j], x_iso[i, j+1], x_iso[i+1, j+1], x_iso[i+1, j]]
                corners_y = [y_iso[i, j], y_iso[i, j+1], y_iso[i+1, j+1], y_iso[i+1, j]]
                
                poly = Polygon(list(zip(corners_x, corners_y)),
                              facecolor=colors[i, j], edgecolor='none',
                              alpha=0.9, zorder=period)
                ax.add_patch(poly)
        
        # Border
        border_x = [x_iso[0, 0], x_iso[0, -1], x_iso[-1, -1], x_iso[-1, 0], x_iso[0, 0]]
        border_y = [y_iso[0, 0], y_iso[0, -1], y_iso[-1, -1], y_iso[-1, 0], y_iso[0, 0]]
        ax.plot(border_x, border_y, 'k-', linewidth=1, alpha=0.5, zorder=period + 0.5)
        
        # Side panels
        right_x = [x_iso[0, -1], x_iso[-1, -1],
                   x_iso[-1, -1] + 0.012, x_iso[0, -1] + 0.012]
        right_y = [y_iso[0, -1], y_iso[-1, -1],
                   y_iso[-1, -1] - 0.03, y_iso[0, -1] - 0.03]
        ax.fill(right_x, right_y, color='#555555', alpha=0.5, zorder=period - 0.1)
        
        # Labels every 3rd layer
        if slice_idx % 3 == 0:
            label_x = x_iso[-1, -1] + 0.05
            label_y = y_iso[-1, -1]
            total_events = sum(event_vals)
            ax.text(label_x, label_y, f'{period_label}\n({int(total_events)} events)',
                   fontsize=8, fontweight='bold', ha='left', va='center',
                   bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray',
                            boxstyle='round,pad=0.25'),
                   zorder=1000)
        
        # Tower markers with individual event breakdowns
        for tower, coords in TOWER_COORDINATES.items():
            t_lon_norm = (coords['lon'] - lon.min()) / (lon.max() - lon.min())
            t_lat_norm = (coords['lat'] - lat.min()) / (lat.max() - lat.min())
            
            lon_idx = np.argmin(np.abs(lon - coords['lon']))
            lat_idx = np.argmin(np.abs(lat - coords['lat']))
            t_elev_norm = elev_norm[lat_idx, lon_idx]
            
            t_x, t_y = isometric_transform(t_lon_norm, t_lat_norm, t_elev_norm * 0.15 + 0.025,
                                           angle=45, scale_z=0.25, depth_offset=depth)
            
            tower_data = period_data[period_data['tower'] == tower]
            
            # Stacked event markers
            y_offset = 0
            for event_col, event_info in EVENT_TYPES.items():
                count = tower_data[event_col].sum() if len(tower_data) > 0 else 0
                if count > 0:
                    marker_size = 30 + np.log1p(count) * 18
                    ax.scatter([t_x], [t_y + y_offset], s=marker_size,
                              c=event_info['color'], marker=event_info['marker'],
                              edgecolors='white', linewidth=0.8,
                              alpha=0.85, zorder=period + 100)
                    y_offset += 0.01
            
            # Tower label on front
            if slice_idx == n_slices - 1:
                ax.text(t_x + 0.012, t_y + 0.012, tower,
                       fontsize=7, fontweight='bold',
                       bbox=dict(facecolor='white', alpha=0.85,
                                edgecolor=coords['color'], boxstyle='round,pad=0.12'),
                       zorder=1001)
    
    # Time arrow
    ax.annotate('', xy=(0.02 + (n_slices-1)*depth_spacing*0.7, 0.1 + (n_slices-1)*depth_spacing*0.5),
               xytext=(0.02, 0.1),
               arrowprops=dict(arrowstyle='->', color='darkred', lw=2.5),
               zorder=1002)
    ax.text(0.02 + (n_slices-1)*depth_spacing*0.35 - 0.08,
            0.1 + (n_slices-1)*depth_spacing*0.25 + 0.04,
           'TIME →', fontsize=13, fontweight='bold', color='darkred',
           rotation=35, zorder=1002)
    
    # Legend
    legend_elements = [
        Line2D([0], [0], marker=info['marker'], color='w',
              markerfacecolor=info['color'], markersize=11,
              label=info['name'], markeredgecolor='white')
        for info in EVENT_TYPES.values()
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9,
             framealpha=0.95, title='Event Types', title_fontsize=10)
    
    # Colorbar for heatmap
    sm = plt.cm.ScalarMappable(cmap=event_cmap, norm=Normalize(0, max_events))
    sm.set_array([])
    cbar_ax = fig.add_axes([0.88, 0.25, 0.02, 0.4])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label('Total Event Count', fontsize=11)
    
    ax.set_title(
        f'4D Isometric Heatmap: Extreme Weather Event Intensity\n'
        f'{n_slices} Time Periods | Color = total event count | Markers = event types',
        fontsize=15, fontweight='bold', pad=20
    )
    
    ax.set_xlim(-0.3, 4.5)
    ax.set_ylim(-0.3, 3.2)
    ax.set_aspect('equal')
    ax.axis('off')
    
    plt.tight_layout(rect=[0, 0, 0.86, 1])
    
    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
    
    plt.show()
    plt.close()


def create_events_by_type_isometric(df, n_slices=12, output_path=None):
    """
    Create separate isometric stacks for each event type.
    Side-by-side comparison of different extreme weather patterns.
    """
    print(f"\nCreating event-type comparison isometric visualization...")
    
    # Generate terrain
    lon, lat, elevation, bounds = generate_terrain_grid(resolution=25)
    LON, LAT = np.meshgrid(lon, lat)
    
    lon_norm = (LON - lon.min()) / (lon.max() - lon.min())
    lat_norm = (LAT - lat.min()) / (lat.max() - lat.min())
    elev_norm = (elevation - elevation.min()) / (elevation.max() - elevation.min())
    
    # Prepare events
    event_cols = list(EVENT_TYPES.keys())
    df_events = df[['timestamp', 'tower'] + event_cols].copy()
    df_events['month'] = df_events['timestamp'].dt.to_period('M')
    months = sorted(df_events['month'].unique())
    
    if len(months) > n_slices:
        months = months[-n_slices:]
    
    n_event_types = len(EVENT_TYPES)
    
    # Create figure with subplots for each event type
    fig, axes = plt.subplots(1, n_event_types, figsize=(28, 12), facecolor='white')
    
    ls = LightSource(azdeg=315, altdeg=45)
    depth_spacing = 0.25
    
    for event_idx, (event_col, event_info) in enumerate(EVENT_TYPES.items()):
        ax = axes[event_idx]
        ax.set_facecolor('white')
        
        # Get colormap for this event type
        if 'Temp' in event_col:
            event_cmap = plt.cm.Blues
        elif 'High' in event_col:
            event_cmap = plt.cm.Reds
        else:
            event_cmap = plt.cm.Greens
        
        # Calculate max for this event type
        max_count = max(df_events.groupby(['month', 'tower'])[event_col].sum().max(), 1)
        
        for slice_idx, month in enumerate(reversed(months)):
            actual_idx = len(months) - 1 - slice_idx
            depth = slice_idx * depth_spacing
            
            month_data = df_events[df_events['month'] == month]
            
            x_iso, y_iso = isometric_transform(lon_norm, lat_norm, elev_norm * 0.12,
                                               angle=45, scale_z=0.2, depth_offset=depth)
            
            # Get event counts per tower
            tower_counts = {}
            for tower in TOWER_COORDINATES:
                tower_data = month_data[month_data['tower'] == tower]
                tower_counts[tower] = tower_data[event_col].sum()
            
            # Interpolate for heatmap
            tower_points_norm = [((TOWER_COORDINATES[t]['lon'] - lon.min()) / (lon.max() - lon.min()),
                                  (TOWER_COORDINATES[t]['lat'] - lat.min()) / (lat.max() - lat.min()))
                                for t in TOWER_COORDINATES]
            event_vals = [tower_counts[t] for t in TOWER_COORDINATES]
            
            if max(event_vals) > 0:
                event_grid = griddata(tower_points_norm, event_vals, (lon_norm, lat_norm),
                                     method='cubic', fill_value=0)
                event_grid = np.clip(event_grid, 0, max_count)
            else:
                event_grid = np.zeros_like(lon_norm)
            
            # Color by event intensity
            colors = event_cmap(event_grid / max_count * 0.8 + 0.1)
            
            # Blend with terrain
            shaded = ls.shade(elevation, cmap=plt.cm.gray, blend_mode='soft')
            colors[..., :3] = colors[..., :3] * 0.7 + shaded[..., :3] * 0.3
            
            # Draw terrain (simplified)
            step = 2
            for i in range(0, len(lat) - step, step):
                for j in range(0, len(lon) - step, step):
                    corners_x = [x_iso[i, j], x_iso[i, j+step], x_iso[i+step, j+step], x_iso[i+step, j]]
                    corners_y = [y_iso[i, j], y_iso[i, j+step], y_iso[i+step, j+step], y_iso[i+step, j]]
                    
                    poly = Polygon(list(zip(corners_x, corners_y)),
                                  facecolor=colors[i, j], edgecolor='none',
                                  alpha=0.88, zorder=actual_idx)
                    ax.add_patch(poly)
            
            # Border
            border_x = [x_iso[0, 0], x_iso[0, -1], x_iso[-1, -1], x_iso[-1, 0], x_iso[0, 0]]
            border_y = [y_iso[0, 0], y_iso[0, -1], y_iso[-1, -1], y_iso[-1, 0], y_iso[0, 0]]
            ax.plot(border_x, border_y, color=event_info['color'], linewidth=0.8, 
                   alpha=0.6, zorder=actual_idx + 0.5)
            
            # Tower markers
            for tower, coords in TOWER_COORDINATES.items():
                t_lon_norm = (coords['lon'] - lon.min()) / (lon.max() - lon.min())
                t_lat_norm = (coords['lat'] - lat.min()) / (lat.max() - lat.min())
                
                lon_idx = np.argmin(np.abs(lon - coords['lon']))
                lat_idx = np.argmin(np.abs(lat - coords['lat']))
                t_elev_norm = elev_norm[lat_idx, lon_idx]
                
                t_x, t_y = isometric_transform(t_lon_norm, t_lat_norm, t_elev_norm * 0.12 + 0.02,
                                               angle=45, scale_z=0.2, depth_offset=depth)
                
                count = tower_counts[tower]
                if count > 0:
                    marker_size = 30 + np.log1p(count) * 20
                    ax.scatter([t_x], [t_y], s=marker_size,
                              c=event_info['color'], marker=event_info['marker'],
                              edgecolors='white', linewidth=0.8,
                              alpha=0.85, zorder=actual_idx + 100)
                else:
                    ax.scatter([t_x], [t_y], s=15, c='lightgray', marker='.',
                              zorder=actual_idx + 50)
                
                # Labels on front
                if slice_idx == len(months) - 1:
                    ax.text(t_x + 0.008, t_y + 0.008, tower,
                           fontsize=6, fontweight='bold',
                           bbox=dict(facecolor='white', alpha=0.75,
                                    edgecolor=coords['color'], boxstyle='round,pad=0.08'),
                           zorder=1001)
        
        ax.set_title(f'{event_info["name"]}', fontsize=12, fontweight='bold',
                    color=event_info['color'])
        ax.set_xlim(-0.2, 3.5)
        ax.set_ylim(-0.2, 2.5)
        ax.set_aspect('equal')
        ax.axis('off')
    
    fig.suptitle(
        f'4D Isometric Comparison: Event Types Side-by-Side\n'
        f'{n_slices} Monthly Layers per Event Type | Color intensity = event count',
        fontsize=16, fontweight='bold', y=1.02
    )
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
    
    plt.show()
    plt.close()


def create_events_interactive_plotly(df, n_slices=15, output_path=None):
    """
    Interactive Plotly isometric visualization of extreme weather events.
    """
    if not HAS_PLOTLY:
        print("Plotly required")
        return None
    
    print(f"\nCreating interactive Plotly events visualization ({n_slices} layers)...")
    
    # Generate terrain
    lon, lat, elevation, bounds = generate_terrain_grid(resolution=20)
    LON, LAT = np.meshgrid(lon, lat)
    
    lon_norm = (lon - lon.min()) / (lon.max() - lon.min())
    lat_norm = (lat - lat.min()) / (lat.max() - lat.min())
    
    # Prepare events
    event_cols = list(EVENT_TYPES.keys())
    df_events = df[['timestamp', 'tower'] + event_cols].copy()
    df_events['period'] = pd.cut(df_events['timestamp'], bins=n_slices, labels=False)
    
    time_bins = pd.cut(df_events['timestamp'], bins=n_slices)
    period_labels = {i: f"{cat.left.strftime('%Y-%m')}" for i, cat in 
                     enumerate(time_bins.cat.categories)}
    
    fig = go.Figure()
    
    z_spacing = 0.25
    
    # Calculate max events
    max_events = 1
    for period in range(n_slices):
        period_data = df_events[df_events['period'] == period]
        for tower in TOWER_COORDINATES:
            tower_data = period_data[period_data['tower'] == tower]
            total = sum(tower_data[col].sum() for col in event_cols)
            max_events = max(max_events, total)
    
    for period in range(n_slices):
        z_base = period * z_spacing
        period_label = period_labels.get(period, f'P{period}')
        period_data = df_events[df_events['period'] == period]
        
        # Calculate event intensity for coloring
        tower_events = {}
        for tower in TOWER_COORDINATES:
            tower_data = period_data[period_data['tower'] == tower]
            tower_events[tower] = sum(tower_data[col].sum() for col in event_cols)
        
        # Interpolate
        tower_points_norm = [((TOWER_COORDINATES[t]['lon'] - lon.min()) / (lon.max() - lon.min()),
                              (TOWER_COORDINATES[t]['lat'] - lat.min()) / (lat.max() - lat.min()))
                            for t in TOWER_COORDINATES]
        event_vals = [tower_events[t] for t in TOWER_COORDINATES]
        
        if max(event_vals) > 0:
            event_grid = griddata(tower_points_norm, event_vals, 
                                 (np.meshgrid(lon_norm, lat_norm)[0], 
                                  np.meshgrid(lon_norm, lat_norm)[1]),
                                 method='cubic', fill_value=0)
        else:
            event_grid = np.zeros((len(lat_norm), len(lon_norm)))
        
        # Add surface layer
        fig.add_trace(go.Surface(
            x=lon_norm, y=lat_norm,
            z=np.full((len(lat_norm), len(lon_norm)), z_base),
            surfacecolor=event_grid,
            colorscale='YlOrRd',
            cmin=0, cmax=max_events,
            opacity=0.8,
            showscale=(period == 0),
            colorbar=dict(title='Event Count', x=1.02) if period == 0 else None,
            name=f'{period_label}',
            hovertemplate=f'Time: {period_label}<br>Events: %{{surfacecolor:.0f}}<extra></extra>'
        ))
        
        # Add tower markers with event breakdown
        for tower, coords in TOWER_COORDINATES.items():
            t_x = (coords['lon'] - lon.min()) / (lon.max() - lon.min())
            t_y = (coords['lat'] - lat.min()) / (lat.max() - lat.min())
            
            tower_data = period_data[period_data['tower'] == tower]
            
            # Add marker for each event type that occurred
            z_offset = 0
            for event_col, event_info in EVENT_TYPES.items():
                count = tower_data[event_col].sum() if len(tower_data) > 0 else 0
                if count > 0:
                    marker_size = 6 + np.log1p(count) * 3
                    
                    fig.add_trace(go.Scatter3d(
                        x=[t_x], y=[t_y], z=[z_base + 0.02 + z_offset * 0.03],
                        mode='markers',
                        marker=dict(size=marker_size, color=event_info['color'],
                                   symbol='diamond', line=dict(width=1, color='white')),
                        name=f'{tower} {event_info["short"]}',
                        showlegend=False,
                        hovertemplate=f'{tower}<br>{event_info["name"]}: {int(count)}<extra></extra>'
                    ))
                    z_offset += 1
    
    fig.update_layout(
        title=dict(
            text=f'4D Interactive: Extreme Weather Events<br>'
                 f'<sub>{n_slices} Time Layers | Color = event intensity | Markers = event types</sub>',
            x=0.5
        ),
        scene=dict(
            xaxis_title='Longitude (norm)',
            yaxis_title='Latitude (norm)',
            zaxis_title='Time Layer',
            camera=dict(eye=dict(x=1.8, y=1.8, z=1.2)),
            aspectratio=dict(x=1, y=1, z=n_slices * z_spacing * 0.8)
        ),
        width=1200,
        height=900
    )
    
    if output_path:
        fig.write_html(output_path)
        print(f"Saved: {output_path}")
    
    return fig


# =============================================================================
# MAIN
# =============================================================================

def main():
    print("=" * 60)
    print("   4D ISOMETRIC EXTREME WEATHER EVENTS VISUALIZATION")
    print("=" * 60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Load full data for event analysis
    df = load_data(sample_size=None)  # Use all data
    
    print("\n" + "-" * 60)
    print("Generating event visualizations...")
    print("-" * 60)
    
    # 1. Monthly events (12 layers)
    print("\n[1/5] Monthly events isometric (12 layers)...")
    try:
        create_events_isometric_monthly(
            df, n_months=12,
            output_path=os.path.join(OUTPUT_DIR, '4d_events_monthly_12.png')
        )
    except Exception as e:
        print(f"  Error: {e}")
        import traceback; traceback.print_exc()
    
    # 2. Weekly events (24 layers - more detail)
    print("\n[2/5] Weekly events isometric (24 layers)...")
    try:
        create_events_isometric_weekly(
            df, n_weeks=24,
            output_path=os.path.join(OUTPUT_DIR, '4d_events_weekly_24.png')
        )
    except Exception as e:
        print(f"  Error: {e}")
        import traceback; traceback.print_exc()
    
    # 3. Event heatmap (18 layers)
    print("\n[3/5] Event heatmap isometric (18 layers)...")
    try:
        create_events_heatmap_isometric(
            df, n_slices=18,
            output_path=os.path.join(OUTPUT_DIR, '4d_events_heatmap_18.png')
        )
    except Exception as e:
        print(f"  Error: {e}")
        import traceback; traceback.print_exc()
    
    # 4. Event type comparison
    print("\n[4/5] Event type comparison isometric...")
    try:
        create_events_by_type_isometric(
            df, n_slices=12,
            output_path=os.path.join(OUTPUT_DIR, '4d_events_by_type.png')
        )
    except Exception as e:
        print(f"  Error: {e}")
        import traceback; traceback.print_exc()
    
    # 5. Interactive Plotly
    print("\n[5/5] Interactive Plotly events (15 layers)...")
    try:
        create_events_interactive_plotly(
            df, n_slices=15,
            output_path=os.path.join(OUTPUT_DIR, '4d_events_interactive.html')
        )
    except Exception as e:
        print(f"  Error: {e}")
        import traceback; traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("   COMPLETE!")
    print(f"   Output: {OUTPUT_DIR}/")
    print("=" * 60)
    
    for f in sorted(os.listdir(OUTPUT_DIR)):
        size = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / 1024
        print(f"  • {f} ({size:.1f} KB)")


if __name__ == '__main__':
    main()

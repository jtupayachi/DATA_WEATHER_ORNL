#!/usr/bin/env python3
"""
4D Isometric Perspective Visualization
======================================
Creates neural-network-style isometric projections of 3D terrain slices
arranged in depth to show the 4th dimension (time).

Visual style: Like architecture diagrams in ML papers showing layer stacks
- Partial overlaps
- 45° rotation for isometric feel
- Depth perspective showing time progression
- Clean, publication-ready graphics

Author: Auto-generated
Date: December 2024
"""

import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import Normalize, LightSource
from matplotlib.patches import Patch, FancyBboxPatch, Rectangle, Polygon
from matplotlib.collections import PolyCollection
import matplotlib.transforms as mtransforms
from scipy.interpolate import griddata
from scipy.ndimage import gaussian_filter
from datetime import datetime
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

OUTPUT_DIR = 'viz_4d_isometric'
DATA_FILE = '/home/jose/DATA_WEATHER_ORNL/fully_labeled_weather_data_with_events.csv'

def load_tower_metadata(json_path='tower_metadata_generated.json'):
    """Load tower coordinates and metadata from JSON file."""
    # Try multiple possible locations for the JSON file
    possible_paths = [
        json_path,
        os.path.join(os.path.dirname(__file__), json_path),
        os.path.join(os.getcwd(), json_path),
    ]
    
    for path in possible_paths:
        if os.path.exists(path):
            print(f"Loading tower metadata from: {path}")
            with open(path, 'r') as f:
                data = json.load(f)
            return data['tower_coordinates']
    
    raise FileNotFoundError(
        f"Could not find {json_path}. Tried locations:\n" + 
        "\n".join(f"  - {p}" for p in possible_paths)
    )

# Load tower coordinates from JSON
TOWER_COORDINATES = load_tower_metadata()

WEATHER_VARS = {
    'TempC_015m': {'name': 'Temperature', 'unit': '°C', 'cmap': 'RdYlBu_r'},
    'WSpdMph_015m': {'name': 'Wind Speed', 'unit': 'mph', 'cmap': 'YlOrRd'},
    'RelHum_015m': {'name': 'Humidity', 'unit': '%', 'cmap': 'YlGnBu'},
}

EVENT_TYPES = {
    'event_E3_LowTemp_lt0': {'name': 'Low Temp', 'color': '#1f77b4', 'marker': 'v'},
    'event_E4_HighWind_Peak_gt25': {'name': 'High Wind', 'color': '#d62728', 'marker': '^'},
    'event_E5_LowWind_lt2': {'name': 'Low Wind', 'color': '#2ca02c', 'marker': 's'},
}


def load_data(filepath=DATA_FILE, sample_size=50000):
    """Load weather data."""
    print(f"Loading data from: {filepath}")
    df = pd.read_csv(filepath, parse_dates=['timestamp'])
    print(f"  Total rows: {len(df):,}")
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
    """
    Transform 3D coordinates to 2D isometric projection.
    
    Parameters:
        x, y, z: 3D coordinates
        angle: Rotation angle in degrees (45° for true isometric)
        scale_z: How much to scale the Z (vertical) axis
        depth_offset: Offset along the depth axis (for stacking layers)
    
    Returns:
        x_iso, y_iso: 2D isometric coordinates
    """
    angle_rad = np.radians(angle)
    
    # Rotate in XY plane
    x_rot = x * np.cos(angle_rad) - y * np.sin(angle_rad)
    y_rot = x * np.sin(angle_rad) + y * np.cos(angle_rad)
    
    # Project to 2D isometric
    # X_iso = x_rotated
    # Y_iso = y_rotated * cos(30°) + z * scale  (typical isometric)
    x_iso = x_rot + depth_offset * 0.7  # Shift right for depth
    y_iso = y_rot * 0.5 + z * scale_z + depth_offset * 0.5  # Shift up for depth
    
    return x_iso, y_iso


def create_isometric_time_slices(df, variable='TempC_015m', n_slices=8, output_path=None):
    """
    Create isometric projection of terrain time slices.
    Like neural network architecture diagrams - partial overlaps in perspective.
    """
    print(f"\nCreating isometric 4D visualization for {variable}...")
    
    var_info = WEATHER_VARS.get(variable, {'name': variable, 'unit': '', 'cmap': 'viridis'})
    
    # Generate terrain
    lon, lat, elevation, bounds = generate_terrain_grid(resolution=35)
    LON, LAT = np.meshgrid(lon, lat)
    
    # Normalize coordinates to [0, 1] for cleaner projection
    lon_norm = (LON - lon.min()) / (lon.max() - lon.min())
    lat_norm = (LAT - lat.min()) / (lat.max() - lat.min())
    elev_norm = (elevation - elevation.min()) / (elevation.max() - elevation.min())
    
    # Prepare time slices
    df_var = df[['timestamp', 'tower', variable, 'has_any_event']].dropna()
    time_min, time_max = df_var['timestamp'].min(), df_var['timestamp'].max()
    time_edges = pd.date_range(start=time_min, end=time_max, periods=n_slices + 1)
    
    # Global color normalization
    global_min = df_var[variable].quantile(0.05)
    global_max = df_var[variable].quantile(0.95)
    norm = Normalize(vmin=global_min, vmax=global_max)
    cmap = plt.get_cmap(var_info['cmap'])
    
    # Create figure with white background
    fig, ax = plt.subplots(figsize=(20, 16), facecolor='white')
    ax.set_facecolor('white')
    
    # Depth spacing between layers (controls overlap)
    depth_spacing = 0.35  # Partial overlap
    
    # Light source for shading
    ls = LightSource(azdeg=315, altdeg=45)
    
    # Plot from back to front (oldest to newest time)
    for slice_idx in range(n_slices - 1, -1, -1):
        t_start, t_end = time_edges[slice_idx], time_edges[slice_idx + 1]
        t_label = t_start.strftime('%Y-%m')
        
        depth = (n_slices - 1 - slice_idx) * depth_spacing
        
        # Get data for this time slice
        mask = (df_var['timestamp'] >= t_start) & (df_var['timestamp'] < t_end)
        df_slice = df_var[mask]
        
        # Interpolate weather values
        if len(df_slice) > 0:
            tower_values = df_slice.groupby('tower')[variable].mean()
            tower_points = [(TOWER_COORDINATES[t]['lon'], TOWER_COORDINATES[t]['lat']) 
                           for t in tower_values.index if t in TOWER_COORDINATES]
            weather_vals = [tower_values[t] for t in tower_values.index if t in TOWER_COORDINATES]
            
            tower_points_norm = [((p[0] - lon.min()) / (lon.max() - lon.min()),
                                  (p[1] - lat.min()) / (lat.max() - lat.min())) 
                                for p in tower_points]
            
            if len(tower_points) >= 3:
                weather_grid = griddata(tower_points_norm, weather_vals, 
                                       (lon_norm, lat_norm),
                                       method='cubic', fill_value=np.mean(weather_vals))
            else:
                weather_grid = np.full_like(elevation, np.mean(weather_vals) if weather_vals else global_min)
        else:
            weather_grid = np.full_like(elevation, global_min)
        
        # Transform to isometric coordinates
        x_iso, y_iso = isometric_transform(lon_norm, lat_norm, elev_norm * 0.3, 
                                           angle=45, scale_z=0.4, depth_offset=depth)
        
        # Create color array
        colors = cmap(norm(weather_grid))
        
        # Apply hillshade
        shaded = ls.shade(elevation, cmap=cm.gray, blend_mode='soft')
        colors[..., :3] = colors[..., :3] * 0.75 + shaded[..., :3] * 0.25
        
        # Draw the surface as filled contours or mesh
        # Use pcolormesh-style rendering
        for i in range(len(lat) - 1):
            for j in range(len(lon) - 1):
                # Get corners of this cell
                corners_x = [x_iso[i, j], x_iso[i, j+1], x_iso[i+1, j+1], x_iso[i+1, j]]
                corners_y = [y_iso[i, j], y_iso[i, j+1], y_iso[i+1, j+1], y_iso[i+1, j]]
                
                # Average color for this cell
                cell_color = colors[i, j]
                
                poly = Polygon(list(zip(corners_x, corners_y)), 
                              facecolor=cell_color, edgecolor='none',
                              alpha=0.95, zorder=slice_idx)
                ax.add_patch(poly)
        
        # Draw border around this layer
        border_x = [x_iso[0, 0], x_iso[0, -1], x_iso[-1, -1], x_iso[-1, 0], x_iso[0, 0]]
        border_y = [y_iso[0, 0], y_iso[0, -1], y_iso[-1, -1], y_iso[-1, 0], y_iso[0, 0]]
        ax.plot(border_x, border_y, 'k-', linewidth=1.5, alpha=0.7, zorder=slice_idx + 0.5)
        
        # Add side panels (3D effect)
        # Right side
        right_x = [x_iso[0, -1], x_iso[-1, -1], 
                   x_iso[-1, -1] + 0.02, x_iso[0, -1] + 0.02, x_iso[0, -1]]
        right_y = [y_iso[0, -1], y_iso[-1, -1],
                   y_iso[-1, -1] - 0.05, y_iso[0, -1] - 0.05, y_iso[0, -1]]
        ax.fill(right_x, right_y, color='#444444', alpha=0.6, zorder=slice_idx - 0.1)
        
        # Bottom side
        bottom_x = [x_iso[-1, 0], x_iso[-1, -1],
                    x_iso[-1, -1] + 0.02, x_iso[-1, 0] + 0.02, x_iso[-1, 0]]
        bottom_y = [y_iso[-1, 0], y_iso[-1, -1],
                    y_iso[-1, -1] - 0.05, y_iso[-1, 0] - 0.05, y_iso[-1, 0]]
        ax.fill(bottom_x, bottom_y, color='#666666', alpha=0.6, zorder=slice_idx - 0.1)
        
        # Add time label
        label_x = x_iso[-1, -1] + 0.08
        label_y = y_iso[-1, -1] - 0.02
        ax.text(label_x, label_y, f'T{slice_idx}: {t_label}',
               fontsize=10, fontweight='bold', ha='left', va='center',
               bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray',
                        boxstyle='round,pad=0.3'),
               zorder=1000)
        
        # Plot tower markers on this layer
        tower_events = df_slice.groupby('tower')['has_any_event'].sum() if len(df_slice) > 0 else {}
        
        for tower, coords in TOWER_COORDINATES.items():
            t_lon_norm = (coords['lon'] - lon.min()) / (lon.max() - lon.min())
            t_lat_norm = (coords['lat'] - lat.min()) / (lat.max() - lat.min())
            
            # Get elevation at tower location
            lon_idx = np.argmin(np.abs(lon - coords['lon']))
            lat_idx = np.argmin(np.abs(lat - coords['lat']))
            t_elev_norm = elev_norm[lat_idx, lon_idx]
            
            t_x, t_y = isometric_transform(t_lon_norm, t_lat_norm, t_elev_norm * 0.3 + 0.05,
                                           angle=45, scale_z=0.4, depth_offset=depth)
            
            has_event = tower_events.get(tower, 0) > 0 if isinstance(tower_events, dict) else (
                tower_events.get(tower, 0) > 0 if tower in tower_events.index else False
            )
            
            marker_size = 80 if has_event else 50
            edge_color = 'red' if has_event else 'white'
            edge_width = 2.5 if has_event else 1.5
            
            ax.scatter([t_x], [t_y], s=marker_size, c=coords['color'],
                      marker='^', edgecolors=edge_color, linewidth=edge_width,
                      zorder=slice_idx + 100)
            
            # Label only on front slice
            if slice_idx == 0:
                ax.text(t_x + 0.02, t_y + 0.02, tower, fontsize=8, fontweight='bold',
                       zorder=1001,
                       bbox=dict(facecolor='white', alpha=0.8, edgecolor=coords['color'],
                                boxstyle='round,pad=0.15'))
    
    # Add depth/time axis arrow
    arrow_start_x, arrow_start_y = 0.1, 0.2
    arrow_end_x = arrow_start_x + (n_slices - 1) * depth_spacing * 0.7
    arrow_end_y = arrow_start_y + (n_slices - 1) * depth_spacing * 0.5
    
    ax.annotate('', xy=(arrow_end_x, arrow_end_y), xytext=(arrow_start_x, arrow_start_y),
               arrowprops=dict(arrowstyle='->', color='darkred', lw=2.5),
               zorder=1002)
    ax.text((arrow_start_x + arrow_end_x) / 2 - 0.1, 
            (arrow_start_y + arrow_end_y) / 2 + 0.05,
           'TIME →', fontsize=14, fontweight='bold', color='darkred',
           rotation=35, zorder=1002)
    
    # Title
    ax.set_title(
        f'4D Isometric Projection: {var_info["name"]} ({var_info["unit"]})\n'
        f'{n_slices} Time Slices in Perspective View\n'
        f'(Like Neural Network Architecture Diagrams)',
        fontsize=16, fontweight='bold', pad=20
    )
    
    ax.set_xlim(-0.3, 2.5)
    ax.set_ylim(-0.3, 2.0)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.85, 0.15, 0.02, 0.5])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(f'{var_info["name"]} ({var_info["unit"]})', fontsize=12)
    
    plt.tight_layout(rect=[0, 0, 0.83, 1])
    
    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
    
    plt.show()
    plt.close()
    
    return fig


def create_isometric_simplified(df, variable='TempC_015m', n_slices=6, output_path=None):
    """
    Create a cleaner isometric view using imshow for each slice.
    More similar to neural network paper diagrams.
    """
    print(f"\nCreating simplified isometric view...")
    
    var_info = WEATHER_VARS.get(variable, {'name': variable, 'unit': '', 'cmap': 'viridis'})
    
    # Generate terrain data
    lon, lat, elevation, bounds = generate_terrain_grid(resolution=50)
    LON, LAT = np.meshgrid(lon, lat)
    
    # Prepare time slices
    df_var = df[['timestamp', 'tower', variable, 'has_any_event']].dropna()
    time_min, time_max = df_var['timestamp'].min(), df_var['timestamp'].max()
    time_edges = pd.date_range(start=time_min, end=time_max, periods=n_slices + 1)
    
    global_min = df_var[variable].quantile(0.05)
    global_max = df_var[variable].quantile(0.95)
    norm = Normalize(vmin=global_min, vmax=global_max)
    cmap = plt.get_cmap(var_info['cmap'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=(18, 14), facecolor='white')
    ax.set_facecolor('white')
    
    # Spacing and sizing
    slice_width = 1.0
    slice_height = 0.8
    x_offset = 0.4  # Horizontal offset per slice (perspective)
    y_offset = 0.35  # Vertical offset per slice (perspective)
    
    ls = LightSource(azdeg=315, altdeg=45)
    
    # Plot from back to front
    for slice_idx in range(n_slices - 1, -1, -1):
        t_start, t_end = time_edges[slice_idx], time_edges[slice_idx + 1]
        t_label = t_start.strftime('%Y-%m')
        
        # Position for this slice
        x_base = (n_slices - 1 - slice_idx) * x_offset
        y_base = (n_slices - 1 - slice_idx) * y_offset
        
        # Get weather data
        mask = (df_var['timestamp'] >= t_start) & (df_var['timestamp'] < t_end)
        df_slice = df_var[mask]
        
        if len(df_slice) > 0:
            tower_values = df_slice.groupby('tower')[variable].mean()
            tower_points = [(TOWER_COORDINATES[t]['lon'], TOWER_COORDINATES[t]['lat']) 
                           for t in tower_values.index if t in TOWER_COORDINATES]
            weather_vals = [tower_values[t] for t in tower_values.index if t in TOWER_COORDINATES]
            
            if len(tower_points) >= 3:
                weather_grid = griddata(tower_points, weather_vals, (LON, LAT),
                                       method='cubic', fill_value=np.mean(weather_vals))
            else:
                weather_grid = np.full_like(elevation, np.mean(weather_vals) if weather_vals else global_min)
        else:
            weather_grid = np.full_like(elevation, global_min)
        
        # Blend with terrain hillshade
        shaded = ls.shade(elevation, cmap=cm.gray, blend_mode='soft')
        rgb_weather = cmap(norm(weather_grid))[..., :3]
        blended = rgb_weather * 0.7 + shaded[..., :3] * 0.3
        
        # Create the slice image with perspective transform
        # Draw as a skewed rectangle
        corners = np.array([
            [x_base, y_base],  # bottom-left
            [x_base + slice_width, y_base + slice_height * 0.15],  # bottom-right (skewed up)
            [x_base + slice_width + 0.1, y_base + slice_height + 0.15],  # top-right
            [x_base + 0.1, y_base + slice_height],  # top-left
        ])
        
        # Use imshow with extent, then add perspective with transform
        extent = [x_base, x_base + slice_width, y_base, y_base + slice_height]
        
        # Draw a filled background for 3D effect
        # Side panel (right)
        side_right = np.array([
            corners[1],  # bottom-right
            corners[2],  # top-right
            [corners[2][0] + 0.08, corners[2][1] - 0.12],  # top-right-back
            [corners[1][0] + 0.08, corners[1][1] - 0.12],  # bottom-right-back
        ])
        ax.fill(side_right[:, 0], side_right[:, 1], color='#555555', alpha=0.8, zorder=slice_idx * 2)
        
        # Side panel (bottom)
        side_bottom = np.array([
            corners[0],  # bottom-left
            corners[1],  # bottom-right
            [corners[1][0] + 0.08, corners[1][1] - 0.12],  # bottom-right-back
            [corners[0][0] + 0.08, corners[0][1] - 0.12],  # bottom-left-back
        ])
        ax.fill(side_bottom[:, 0], side_bottom[:, 1], color='#777777', alpha=0.8, zorder=slice_idx * 2)
        
        # Draw the main face (terrain+weather image)
        # For simplicity, draw as colored rectangle with weather pattern overlay
        face_poly = plt.Polygon(corners, facecolor='none', edgecolor='black', 
                               linewidth=2, zorder=slice_idx * 2 + 1)
        ax.add_patch(face_poly)
        
        # Draw terrain as contour-like visualization on the face
        # Sample the image to a coarser grid and draw as rectangles
        sample_step = 5
        for i in range(0, len(lat) - sample_step, sample_step):
            for j in range(0, len(lon) - sample_step, sample_step):
                # Map grid cell to face coordinates
                u = j / len(lon)  # 0 to 1 horizontal
                v = i / len(lat)  # 0 to 1 vertical
                
                # Bilinear interpolation of corner positions
                def lerp(a, b, t):
                    return a + (b - a) * t
                
                bottom = lerp(corners[0], corners[1], u)
                top = lerp(corners[3], corners[2], u)
                pos = lerp(bottom, top, v)
                
                # Cell size in face coordinates
                cell_w = slice_width / (len(lon) / sample_step) * 0.9
                cell_h = slice_height / (len(lat) / sample_step) * 0.9
                
                # Get color
                color = blended[i, j]
                
                rect = plt.Rectangle(pos, cell_w, cell_h, 
                                    facecolor=color, edgecolor='none',
                                    alpha=0.95, zorder=slice_idx * 2 + 0.5)
                ax.add_patch(rect)
        
        # Border
        ax.plot(corners[[0,1,2,3,0], 0], corners[[0,1,2,3,0], 1], 
               'k-', linewidth=2, zorder=slice_idx * 2 + 1.5)
        
        # Time label
        label_x = corners[2][0] + 0.15
        label_y = corners[2][1]
        ax.text(label_x, label_y, f'T{slice_idx}\n{t_label}',
               fontsize=11, fontweight='bold', ha='left', va='center',
               bbox=dict(facecolor='lightyellow', alpha=0.95, edgecolor='orange',
                        boxstyle='round,pad=0.4'),
               zorder=1000)
        
        # Plot towers
        tower_events = df_slice.groupby('tower')['has_any_event'].sum() if len(df_slice) > 0 else {}
        
        for tower, coords in TOWER_COORDINATES.items():
            # Map tower to face coordinates
            u = (coords['lon'] - lon.min()) / (lon.max() - lon.min())
            v = (coords['lat'] - lat.min()) / (lat.max() - lat.min())
            
            def lerp(a, b, t):
                return a + (b - a) * t
            
            bottom = lerp(corners[0], corners[1], u)
            top = lerp(corners[3], corners[2], u)
            pos = lerp(bottom, top, v)
            
            has_event = (hasattr(tower_events, 'get') and tower_events.get(tower, 0) > 0) or \
                       (hasattr(tower_events, 'index') and tower in tower_events.index and tower_events[tower] > 0)
            
            marker_size = 100 if has_event else 60
            edge_color = 'red' if has_event else 'white'
            
            ax.scatter([pos[0]], [pos[1]], s=marker_size, c=coords['color'],
                      marker='^', edgecolors=edge_color, linewidth=2,
                      zorder=slice_idx * 2 + 100)
            
            # Labels on front slice only
            if slice_idx == 0:
                ax.text(pos[0] + 0.05, pos[1] + 0.03, tower,
                       fontsize=8, fontweight='bold', zorder=1001,
                       bbox=dict(facecolor='white', alpha=0.9, edgecolor=coords['color'],
                                boxstyle='round,pad=0.2'))
    
    # Time axis arrow
    ax.annotate('', xy=(x_offset * n_slices + 0.3, y_offset * n_slices + 0.2),
               xytext=(0.2, 0.1),
               arrowprops=dict(arrowstyle='->', color='darkred', lw=3),
               zorder=1002)
    ax.text(x_offset * n_slices / 2, y_offset * n_slices / 2 - 0.15,
           'TIME DIMENSION →', fontsize=14, fontweight='bold', color='darkred',
           rotation=40, zorder=1002)
    
    ax.set_title(
        f'4D Isometric View: {var_info["name"]} Over Time\n'
        f'Neural Network Architecture Style - {n_slices} Temporal Layers\n'
        f'Each "card" = 3D terrain snapshot at time t',
        fontsize=16, fontweight='bold', pad=20
    )
    
    ax.set_xlim(-0.5, x_offset * n_slices + slice_width + 1)
    ax.set_ylim(-0.5, y_offset * n_slices + slice_height + 0.5)
    ax.set_aspect('equal')
    ax.axis('off')
    
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar_ax = fig.add_axes([0.88, 0.2, 0.02, 0.4])
    cbar = fig.colorbar(sm, cax=cbar_ax)
    cbar.set_label(f'{var_info["name"]} ({var_info["unit"]})', fontsize=12)
    
    plt.tight_layout(rect=[0, 0, 0.86, 1])
    
    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
    
    plt.show()
    plt.close()


def create_neural_network_style(df, variable='TempC_015m', n_slices=8, output_path=None):
    """
    Create a clean neural-network-paper-style 4D visualization.
    Uses 3D matplotlib with custom view for isometric effect.
    """
    print(f"\nCreating neural network style 4D visualization...")
    
    var_info = WEATHER_VARS.get(variable, {'name': variable, 'unit': '', 'cmap': 'viridis'})
    
    # Generate terrain
    lon, lat, elevation, bounds = generate_terrain_grid(resolution=30)
    LON, LAT = np.meshgrid(lon, lat)
    
    # Normalize
    lon_norm = (LON - lon.min()) / (lon.max() - lon.min())
    lat_norm = (LAT - lat.min()) / (lat.max() - lat.min())
    
    # Prepare time slices
    df_var = df[['timestamp', 'tower', variable, 'has_any_event']].dropna()
    time_min, time_max = df_var['timestamp'].min(), df_var['timestamp'].max()
    time_edges = pd.date_range(start=time_min, end=time_max, periods=n_slices + 1)
    
    global_min = df_var[variable].quantile(0.05)
    global_max = df_var[variable].quantile(0.95)
    norm = Normalize(vmin=global_min, vmax=global_max)
    cmap = plt.get_cmap(var_info['cmap'])
    
    # Create 3D figure
    fig = plt.figure(figsize=(22, 16), facecolor='white')
    ax = fig.add_subplot(111, projection='3d', facecolor='white')
    
    ls = LightSource(azdeg=315, altdeg=45)
    
    # Layer spacing along Z (time axis)
    z_spacing = 0.4
    
    for slice_idx in range(n_slices):
        t_start, t_end = time_edges[slice_idx], time_edges[slice_idx + 1]
        t_label = t_start.strftime('%Y-%m')
        
        z_base = slice_idx * z_spacing
        
        # Get weather data
        mask = (df_var['timestamp'] >= t_start) & (df_var['timestamp'] < t_end)
        df_slice = df_var[mask]
        
        if len(df_slice) > 0:
            tower_values = df_slice.groupby('tower')[variable].mean()
            tower_points = [(TOWER_COORDINATES[t]['lon'], TOWER_COORDINATES[t]['lat']) 
                           for t in tower_values.index if t in TOWER_COORDINATES]
            weather_vals = [tower_values[t] for t in tower_values.index if t in TOWER_COORDINATES]
            
            if len(tower_points) >= 3:
                weather_grid = griddata(tower_points, weather_vals, (LON, LAT),
                                       method='cubic', fill_value=np.mean(weather_vals))
            else:
                weather_grid = np.full_like(elevation, np.mean(weather_vals) if weather_vals else global_min)
        else:
            weather_grid = np.full_like(elevation, global_min)
        
        # Create flat surface at z_base (like a layer/card)
        Z_flat = np.full_like(lon_norm, z_base)
        
        # Color by weather
        colors = cmap(norm(weather_grid))
        shaded = ls.shade(elevation, cmap=cm.gray, blend_mode='soft')
        colors[..., :3] = colors[..., :3] * 0.7 + shaded[..., :3] * 0.3
        
        # Plot flat surface
        ax.plot_surface(lon_norm, lat_norm, Z_flat,
                       facecolors=colors,
                       linewidth=0.5,
                       edgecolor='gray',
                       alpha=0.9,
                       shade=False)
        
        # Add border frame around layer
        border_x = [0, 1, 1, 0, 0]
        border_y = [0, 0, 1, 1, 0]
        border_z = [z_base] * 5
        ax.plot(border_x, border_y, border_z, 'k-', linewidth=2, alpha=0.8)
        
        # Add slight 3D thickness (side panels)
        thickness = 0.02
        
        # Right side
        ax.plot_surface(
            np.array([[1, 1], [1, 1]]),
            np.array([[0, 1], [0, 1]]),
            np.array([[z_base, z_base], [z_base - thickness, z_base - thickness]]),
            color='#555555', alpha=0.7
        )
        
        # Front side
        ax.plot_surface(
            np.array([[0, 1], [0, 1]]),
            np.array([[0, 0], [0, 0]]),
            np.array([[z_base, z_base], [z_base - thickness, z_base - thickness]]),
            color='#777777', alpha=0.7
        )
        
        # Time label
        ax.text(1.1, 0.5, z_base, f'T{slice_idx}: {t_label}',
               fontsize=10, fontweight='bold', ha='left', va='center',
               bbox=dict(facecolor='white', alpha=0.9, edgecolor='gray'))
        
        # Plot towers
        tower_events = df_slice.groupby('tower')['has_any_event'].sum() if len(df_slice) > 0 else {}
        
        for tower, coords in TOWER_COORDINATES.items():
            t_x = (coords['lon'] - lon.min()) / (lon.max() - lon.min())
            t_y = (coords['lat'] - lat.min()) / (lat.max() - lat.min())
            
            has_event = (hasattr(tower_events, 'get') and tower_events.get(tower, 0) > 0) or \
                       (hasattr(tower_events, 'index') and tower in tower_events.index and tower_events[tower] > 0)
            
            marker_size = 120 if has_event else 60
            edge_color = 'red' if has_event else 'white'
            
            ax.scatter([t_x], [t_y], [z_base + 0.02],
                      s=marker_size, c=coords['color'], marker='^',
                      edgecolors=edge_color, linewidth=2, zorder=1000)
            
            # Label on first and last slices
            if slice_idx == 0 or slice_idx == n_slices - 1:
                ax.text(t_x, t_y, z_base + 0.05, tower,
                       fontsize=8, fontweight='bold', ha='center',
                       zorder=1001)
    
    # Draw connecting lines between layers (time axis visualization)
    for corner in [(0, 0), (1, 0), (1, 1), (0, 1)]:
        ax.plot([corner[0]] * n_slices, [corner[1]] * n_slices,
               [i * z_spacing for i in range(n_slices)],
               'k--', alpha=0.3, linewidth=1)
    
    # Time axis arrow
    ax.quiver(0.5, -0.2, 0, 0, 0, z_spacing * (n_slices - 1),
             arrow_length_ratio=0.05, color='darkred', linewidth=2)
    ax.text(0.5, -0.3, z_spacing * n_slices / 2,
           'TIME →', fontsize=12, fontweight='bold', color='darkred',
           ha='center', rotation=90)
    
    ax.set_xlabel('Longitude (normalized)', fontsize=11, labelpad=10)
    ax.set_ylabel('Latitude (normalized)', fontsize=11, labelpad=10)
    ax.set_zlabel('Time Layer', fontsize=11, labelpad=10)
    
    ax.set_title(
        f'4D Neural Network Style: {var_info["name"]} Over Time\n'
        f'{n_slices} Temporal Layers | Each layer = 3D terrain snapshot\n'
        f'Publication-ready isometric projection',
        fontsize=15, fontweight='bold', pad=20
    )
    
    # Set isometric-like view
    ax.view_init(elev=25, azim=45)
    ax.set_box_aspect([1, 1, n_slices * z_spacing * 0.8])
    
    # Colorbar
    sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax, shrink=0.4, aspect=25, pad=0.1)
    cbar.set_label(f'{var_info["name"]} ({var_info["unit"]})', fontsize=11)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=200, bbox_inches='tight', facecolor='white')
        print(f"Saved: {output_path}")
    
    plt.show()
    plt.close()
    
    return fig


def create_4d_plotly_isometric(df, variable='TempC_015m', n_slices=10, output_path=None):
    """
    Create interactive Plotly isometric 4D visualization.
    Layers are shown as flat planes with partial overlap in 3D space.
    """
    if not HAS_PLOTLY:
        print("Plotly required")
        return None
    
    print(f"\nCreating interactive Plotly isometric 4D visualization...")
    
    var_info = WEATHER_VARS.get(variable, {'name': variable, 'unit': '', 'cmap': 'viridis'})
    
    # Generate terrain
    lon, lat, elevation, bounds = generate_terrain_grid(resolution=25)
    LON, LAT = np.meshgrid(lon, lat)
    
    # Normalize coordinates
    lon_norm = (lon - lon.min()) / (lon.max() - lon.min())
    lat_norm = (lat - lat.min()) / (lat.max() - lat.min())
    
    # Prepare time slices
    df_var = df[['timestamp', 'tower', variable, 'has_any_event']].dropna()
    time_min, time_max = df_var['timestamp'].min(), df_var['timestamp'].max()
    time_edges = pd.date_range(start=time_min, end=time_max, periods=n_slices + 1)
    
    global_min = df_var[variable].quantile(0.05)
    global_max = df_var[variable].quantile(0.95)
    
    fig = go.Figure()
    
    z_spacing = 0.3
    
    for slice_idx in range(n_slices):
        t_start, t_end = time_edges[slice_idx], time_edges[slice_idx + 1]
        t_label = t_start.strftime('%Y-%m')
        
        z_base = slice_idx * z_spacing
        
        # Get weather data
        mask = (df_var['timestamp'] >= t_start) & (df_var['timestamp'] < t_end)
        df_slice = df_var[mask]
        
        if len(df_slice) > 0:
            tower_values = df_slice.groupby('tower')[variable].mean()
            tower_points = [(TOWER_COORDINATES[t]['lon'], TOWER_COORDINATES[t]['lat']) 
                           for t in tower_values.index if t in TOWER_COORDINATES]
            weather_vals = [tower_values[t] for t in tower_values.index if t in TOWER_COORDINATES]
            
            if len(tower_points) >= 3:
                weather_grid = griddata(tower_points, weather_vals, (LON, LAT),
                                       method='cubic', fill_value=np.mean(weather_vals))
            else:
                weather_grid = np.full_like(elevation, np.mean(weather_vals) if weather_vals else global_min)
        else:
            weather_grid = np.full_like(elevation, global_min)
        
        # Add flat surface at z_base
        fig.add_trace(go.Surface(
            x=lon_norm, y=lat_norm,
            z=np.full_like(LON, z_base),
            surfacecolor=weather_grid,
            colorscale='RdYlBu_r',
            cmin=global_min, cmax=global_max,
            opacity=0.85,
            showscale=(slice_idx == 0),
            colorbar=dict(title=f'{var_info["name"]}<br>({var_info["unit"]})', x=1.02) if slice_idx == 0 else None,
            name=f'T{slice_idx}: {t_label}',
            hovertemplate=f'Time: {t_label}<br>' +
                         f'{var_info["name"]}: ' + '%{surfacecolor:.1f}' + f' {var_info["unit"]}<br>' +
                         '<extra></extra>'
        ))
        
        # Add tower markers
        tower_events = df_slice.groupby('tower')['has_any_event'].sum() if len(df_slice) > 0 else {}
        
        tower_x, tower_y, tower_z = [], [], []
        tower_colors, tower_sizes, tower_names = [], [], []
        
        for tower, coords in TOWER_COORDINATES.items():
            t_x = (coords['lon'] - lon.min()) / (lon.max() - lon.min())
            t_y = (coords['lat'] - lat.min()) / (lat.max() - lat.min())
            
            tower_x.append(t_x)
            tower_y.append(t_y)
            tower_z.append(z_base + 0.02)
            tower_colors.append(coords['color'])
            tower_names.append(tower)
            
            has_event = (hasattr(tower_events, 'get') and tower_events.get(tower, 0) > 0) or \
                       (hasattr(tower_events, 'index') and tower in tower_events.index and tower_events[tower] > 0)
            tower_sizes.append(12 if has_event else 8)
        
        fig.add_trace(go.Scatter3d(
            x=tower_x, y=tower_y, z=tower_z,
            mode='markers+text',
            marker=dict(size=tower_sizes, color=tower_colors, symbol='diamond',
                       line=dict(width=1, color='white')),
            text=tower_names if slice_idx == 0 or slice_idx == n_slices - 1 else [''] * len(tower_names),
            textposition='top center',
            textfont=dict(size=10),
            name=f'Towers T{slice_idx}',
            showlegend=False,
            hovertemplate='%{text}<br>Time Layer: ' + str(slice_idx) + '<extra></extra>'
        ))
    
    fig.update_layout(
        title=dict(
            text=f'4D Isometric Visualization: {var_info["name"]}<br>'
                 f'<sub>{n_slices} Time Layers | Neural Network Style Projection</sub>',
            x=0.5
        ),
        scene=dict(
            xaxis_title='Longitude (norm)',
            yaxis_title='Latitude (norm)',
            zaxis_title='Time Layer',
            camera=dict(
                eye=dict(x=1.8, y=1.8, z=1.2),  # Isometric-like view
                up=dict(x=0, y=0, z=1)
            ),
            aspectratio=dict(x=1, y=1, z=n_slices * z_spacing)
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
    print("   4D ISOMETRIC PERSPECTIVE VISUALIZATION")
    print("   (Neural Network Architecture Style)")
    print("=" * 60)
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    df = load_data(sample_size=80000)
    
    print("\n" + "-" * 60)
    print("Generating isometric 4D visualizations...")
    print("-" * 60)
    
    # 1. Isometric time slices (2D projection)
    print("\n[1/4] Isometric time slices (2D)...")
    try:
        create_isometric_time_slices(
            df, variable='TempC_015m', n_slices=6,
            output_path=os.path.join(OUTPUT_DIR, '4d_isometric_slices.png')
        )
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
    
    # 2. Simplified isometric view
    print("\n[2/4] Simplified isometric view...")
    try:
        create_isometric_simplified(
            df, variable='TempC_015m', n_slices=6,
            output_path=os.path.join(OUTPUT_DIR, '4d_isometric_simplified.png')
        )
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
    
    # 3. Neural network style (3D matplotlib)
    print("\n[3/4] Neural network style 3D...")
    try:
        create_neural_network_style(
            df, variable='TempC_015m', n_slices=8,
            output_path=os.path.join(OUTPUT_DIR, '4d_neural_network_style.png')
        )
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
    
    # 4. Interactive Plotly isometric
    print("\n[4/4] Interactive Plotly isometric...")
    try:
        create_4d_plotly_isometric(
            df, variable='TempC_015m', n_slices=10,
            output_path=os.path.join(OUTPUT_DIR, '4d_isometric_interactive.html')
        )
    except Exception as e:
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "=" * 60)
    print("   COMPLETE!")
    print(f"   Output: {OUTPUT_DIR}/")
    print("=" * 60)
    
    for f in sorted(os.listdir(OUTPUT_DIR)):
        size = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / 1024
        print(f"  • {f} ({size:.1f} KB)")


if __name__ == '__main__':
    main()

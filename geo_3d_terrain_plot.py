#!/usr/bin/env python3
"""
High-Resolution 3D Geographic Terrain Visualization
====================================================
Creates detailed 3D terrain plots of the ORNL weather tower region
using elevation data from various sources.

Usage:
    python geo_3d_terrain_plot.py

Requirements:
    pip install numpy matplotlib scipy requests rasterio elevation
    
For highest resolution, you may also want:
    pip install cartopy pyproj geopandas
    
Elevation Data Sources:
    1. SRTM (Shuttle Radar Topography Mission) - 30m resolution
    2. USGS 3DEP (3D Elevation Program) - up to 1m resolution
    3. OpenTopography API - Various resolutions

Author: Auto-generated
Date: December 2024
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import LightSource, Normalize
from mpl_toolkits.mplot3d import Axes3D
from scipy.interpolate import griddata, RectBivariateSpline
from scipy.ndimage import gaussian_filter
import os
import json
from pathlib import Path

# Optional imports - will use fallbacks if not available
try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    print("Warning: 'requests' not installed. Install with: pip install requests")

try:
    import rasterio
    from rasterio.merge import merge
    HAS_RASTERIO = True
except ImportError:
    HAS_RASTERIO = False
    print("Warning: 'rasterio' not installed. Using synthetic terrain. Install with: pip install rasterio")

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    print("Warning: 'cartopy' not installed. Some map features unavailable. Install with: pip install cartopy")

# =============================================================================
# TOWER COORDINATES (ORNL Weather Towers) - Full Details
# =============================================================================
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
        'color': '#e41a1c',  # Red
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
        'color': '#377eb8',  # Blue
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
        'color': '#4daf4a',  # Green
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
        'color': '#984ea3',  # Purple
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
        'color': '#ff7f00',  # Orange
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
        'color': '#a65628',  # Brown
    },
}

# Region bounds (with padding)
REGION_INFO = {
    'name': 'Oak Ridge National Laboratory (ORNL)',
    'state': 'Tennessee, USA',
    'center_lat': 35.931,
    'center_lon': -84.310,
    'terrain': 'Ridge and Valley Province, Appalachian Region',
}

# =============================================================================
# ELEVATION DATA FUNCTIONS
# =============================================================================

def get_region_bounds(padding_km=1.0):
    """Calculate region bounds with padding around towers."""
    lats = [t['lat'] for t in TOWER_COORDINATES.values()]
    lons = [t['lon'] for t in TOWER_COORDINATES.values()]
    
    # Convert km to degrees (approximate)
    lat_padding = padding_km / 111.0
    lon_padding = padding_km / (111.0 * np.cos(np.radians(np.mean(lats))))
    
    return {
        'min_lat': min(lats) - lat_padding,
        'max_lat': max(lats) + lat_padding,
        'min_lon': min(lons) - lon_padding,
        'max_lon': max(lons) + lon_padding,
    }


# =============================================================================
# FREE OPEN-SOURCE ELEVATION DATA SOURCES
# =============================================================================
# 
# 1. OPEN-ELEVATION API (Easiest - No registration required)
#    - Resolution: ~30m (SRTM-based)
#    - URL: https://open-elevation.com/
#    - Limit: Free, but rate-limited
#
# 2. OPENTOPOGRAPHY (Best quality - Free registration)
#    - Resolution: 30m SRTM, 30m Copernicus, 30m ALOS
#    - URL: https://opentopography.org/
#    - Get API key: https://portal.opentopography.org/requestService?service=api
#
# 3. NASA EARTHDATA (SRTM source - Free registration)
#    - Resolution: 30m (SRTM GL1) or 90m (SRTM GL3)
#    - URL: https://urs.earthdata.nasa.gov/
#
# 4. COPERNICUS DEM (European Space Agency - Free registration)
#    - Resolution: 30m or 90m global
#    - URL: https://spacedata.copernicus.eu/
#
# 5. USGS 3DEP (USA only - Highest resolution)
#    - Resolution: 1m LiDAR in many areas
#    - URL: https://apps.nationalmap.gov/downloader/
# =============================================================================


def fetch_open_elevation_grid(bounds, resolution=50):
    """
    Fetch elevation data from Open-Elevation API (FREE, no registration).
    
    This is the EASIEST option - works out of the box!
    
    API: https://open-elevation.com/
    
    Parameters:
        bounds: dict with min_lat, max_lat, min_lon, max_lon
        resolution: number of points per dimension (e.g., 50 = 50x50 grid)
    
    Returns:
        lon, lat, elevation arrays
    """
    if not HAS_REQUESTS:
        print("ERROR: 'requests' library required. Install with: pip install requests")
        return None, None, None
    
    print(f"Fetching elevation from Open-Elevation API ({resolution}x{resolution} grid)...")
    print("  Source: https://open-elevation.com/ (FREE, no registration)")
    
    # Create grid of points
    lats = np.linspace(bounds['min_lat'], bounds['max_lat'], resolution)
    lons = np.linspace(bounds['min_lon'], bounds['max_lon'], resolution)
    
    # Prepare locations for API request
    locations = []
    for lat in lats:
        for lon in lons:
            locations.append({"latitude": lat, "longitude": lon})
    
    # API endpoint
    url = "https://api.open-elevation.com/api/v1/lookup"
    
    # Split into chunks (API limit ~1000 points per request)
    chunk_size = 500
    all_elevations = []
    
    for i in range(0, len(locations), chunk_size):
        chunk = locations[i:i+chunk_size]
        
        try:
            response = requests.post(
                url,
                json={"locations": chunk},
                headers={"Content-Type": "application/json"},
                timeout=60
            )
            
            if response.status_code == 200:
                results = response.json()['results']
                elevations = [r['elevation'] for r in results]
                all_elevations.extend(elevations)
                print(f"  Downloaded {len(all_elevations)}/{len(locations)} points...")
            else:
                print(f"  API Error: {response.status_code}")
                # Fill with NaN for failed chunk
                all_elevations.extend([np.nan] * len(chunk))
                
        except Exception as e:
            print(f"  Request failed: {e}")
            all_elevations.extend([np.nan] * len(chunk))
    
    # Reshape to 2D grid
    elevation = np.array(all_elevations).reshape(resolution, resolution)
    
    # Handle any NaN values by interpolation
    if np.any(np.isnan(elevation)):
        from scipy.ndimage import generic_filter
        elevation = np.nan_to_num(elevation, nan=np.nanmean(elevation))
    
    print(f"  SUCCESS! Elevation range: {elevation.min():.0f}m - {elevation.max():.0f}m")
    
    return lons, lats, elevation


def fetch_opentopography_dem(bounds, api_key=None, output_dir='elevation_data', dem_type='SRTMGL1'):
    """
    Fetch DEM from OpenTopography API (FREE with registration).
    
    Get your FREE API key at: https://portal.opentopography.org/requestService?service=api
    
    Available DEM types:
        - 'SRTMGL1': SRTM GL1 30m (recommended)
        - 'SRTMGL3': SRTM GL3 90m
        - 'AW3D30': ALOS World 3D 30m
        - 'COP30': Copernicus DEM 30m (best quality)
        - 'COP90': Copernicus DEM 90m
    
    Parameters:
        bounds: dict with min_lat, max_lat, min_lon, max_lon
        api_key: OpenTopography API key (required)
        output_dir: Directory to save downloaded data
        dem_type: Type of DEM to download
    
    Returns:
        Path to downloaded GeoTIFF file
    """
    if not HAS_REQUESTS:
        print("ERROR: 'requests' library required")
        return None
    
    if not api_key:
        print("\n" + "="*60)
        print("OPENTOPOGRAPHY API KEY REQUIRED")
        print("="*60)
        print("Get your FREE API key at:")
        print("  https://portal.opentopography.org/requestService?service=api")
        print("\nThen either:")
        print("  1. Set environment variable: export OPENTOPO_API_KEY='your_key'")
        print("  2. Pass api_key parameter to this function")
        print("="*60 + "\n")
        return None
    
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Fetching {dem_type} from OpenTopography...")
    print(f"  Bounds: {bounds['min_lat']:.4f}N to {bounds['max_lat']:.4f}N")
    print(f"          {bounds['min_lon']:.4f}W to {bounds['max_lon']:.4f}W")
    
    # OpenTopography Global DEM API endpoint
    base_url = "https://portal.opentopography.org/API/globaldem"
    
    params = {
        'demtype': dem_type,
        'south': bounds['min_lat'],
        'north': bounds['max_lat'],
        'west': bounds['min_lon'],
        'east': bounds['max_lon'],
        'outputFormat': 'GTiff',
        'API_Key': api_key,
    }
    
    try:
        response = requests.get(base_url, params=params, timeout=120)
        
        if response.status_code == 200:
            output_path = os.path.join(output_dir, f'opentopo_{dem_type}.tif')
            with open(output_path, 'wb') as f:
                f.write(response.content)
            print(f"  SUCCESS! Saved to: {output_path}")
            return output_path
        else:
            print(f"  API Error {response.status_code}: {response.text[:200]}")
            return None
            
    except Exception as e:
        print(f"  Error: {e}")
        return None


def fetch_nasa_srtm(bounds, output_dir='elevation_data'):
    """
    Download SRTM data from NASA Earthdata.
    
    Requires registration at: https://urs.earthdata.nasa.gov/
    
    This function provides instructions for manual download.
    """
    print("\n" + "="*60)
    print("NASA EARTHDATA SRTM DOWNLOAD")
    print("="*60)
    print("\nFor NASA SRTM data (30m resolution):")
    print("\n1. Register at: https://urs.earthdata.nasa.gov/")
    print("\n2. Use EarthExplorer: https://earthexplorer.usgs.gov/")
    print("   - Search for 'SRTM 1 Arc-Second Global'")
    print(f"   - Set coordinates: {bounds['min_lat']:.4f}, {bounds['min_lon']:.4f}")
    print(f"                  to: {bounds['max_lat']:.4f}, {bounds['max_lon']:.4f}")
    print("\n3. Or use direct tile download:")
    
    # Calculate SRTM tile names
    lat_tile = int(bounds['min_lat'])
    lon_tile = int(bounds['min_lon'])
    lat_letter = 'N' if lat_tile >= 0 else 'S'
    lon_letter = 'E' if lon_tile >= 0 else 'W'
    tile_name = f"{lat_letter}{abs(lat_tile):02d}{lon_letter}{abs(lon_tile):03d}"
    
    print(f"   SRTM Tile: {tile_name}")
    print(f"   URL: https://e4ftl01.cr.usgs.gov/MEASURES/SRTMGL1.003/2000.02.11/{tile_name}.SRTMGL1.hgt.zip")
    print("="*60 + "\n")
    
    return None


def download_srtm_elevation(bounds, output_dir='elevation_data'):
    """
    Download SRTM elevation data using the 'elevation' package.
    
    Install with: pip install elevation
    
    This package automatically handles NASA SRTM download and caching.
    """
    try:
        import elevation
        
        os.makedirs(output_dir, exist_ok=True)
        output_path = os.path.join(output_dir, 'srtm_ornl.tif')
        
        print("Downloading SRTM data using 'elevation' package...")
        
        # Download SRTM 30m data
        elevation.clip(
            bounds=(bounds['min_lon'], bounds['min_lat'], 
                   bounds['max_lon'], bounds['max_lat']),
            output=output_path,
            product='SRTM1'  # 30m resolution
        )
        
        print(f"  SUCCESS! Saved to: {output_path}")
        return output_path
        
    except ImportError:
        print("\nPackage 'elevation' not installed.")
        print("Install with: pip install elevation")
        print("This package auto-downloads SRTM data from NASA.\n")
        return None
    except Exception as e:
        print(f"Error downloading SRTM: {e}")
        return None


def load_geotiff_elevation(filepath):
    """Load elevation data from a GeoTIFF file."""
    if not HAS_RASTERIO:
        print("ERROR: 'rasterio' required to load GeoTIFF files")
        print("Install with: pip install rasterio")
        return None, None, None
    
    print(f"Loading GeoTIFF: {filepath}")
    
    with rasterio.open(filepath) as src:
        elevation = src.read(1)
        bounds = src.bounds
        
        # Create coordinate arrays
        rows, cols = elevation.shape
        lon = np.linspace(bounds.left, bounds.right, cols)
        lat = np.linspace(bounds.top, bounds.bottom, rows)
        
        print(f"  Shape: {elevation.shape}")
        print(f"  Elevation range: {elevation.min():.0f}m - {elevation.max():.0f}m")
        
    return lon, lat, elevation


def generate_synthetic_terrain(bounds, resolution=100):
    """
    Generate synthetic but realistic terrain for the Ridge and Valley region.
    Uses actual tower elevations as control points.
    
    This creates terrain that mimics the Ridge and Valley Province characteristics:
    - SW to NE oriented ridges
    - Parallel valleys
    - Gradual elevation changes
    """
    print("Generating synthetic terrain based on tower elevations...")
    
    # Create coordinate grid
    lon = np.linspace(bounds['min_lon'], bounds['max_lon'], resolution)
    lat = np.linspace(bounds['min_lat'], bounds['max_lat'], resolution)
    LON, LAT = np.meshgrid(lon, lat)
    
    # Base elevation (interpolated from tower data)
    tower_points = []
    tower_elevations = []
    for tower, data in TOWER_COORDINATES.items():
        tower_points.append([data['lon'], data['lat']])
        tower_elevations.append(data['elevation_m'])
    
    tower_points = np.array(tower_points)
    tower_elevations = np.array(tower_elevations)
    
    # Interpolate base elevation
    base_elevation = griddata(
        tower_points, 
        tower_elevations, 
        (LON, LAT), 
        method='cubic',
        fill_value=np.mean(tower_elevations)
    )
    
    # Add Ridge and Valley topography pattern
    # Ridges run approximately SW to NE (about 45 degrees from E-W)
    ridge_angle = np.radians(45)
    
    # Rotate coordinates to align with ridge direction
    lon_center = np.mean(lon)
    lat_center = np.mean(lat)
    
    # Convert to approximate meters for wave calculation
    lon_m = (LON - lon_center) * 111000 * np.cos(np.radians(lat_center))
    lat_m = (LAT - lat_center) * 111000
    
    # Rotated coordinates
    x_rot = lon_m * np.cos(ridge_angle) + lat_m * np.sin(ridge_angle)
    y_rot = -lon_m * np.sin(ridge_angle) + lat_m * np.cos(ridge_angle)
    
    # Create ridge pattern (perpendicular to ridge direction)
    # Ridge spacing approximately 1-2 km in this region
    ridge_wavelength = 1500  # meters
    ridge_amplitude = 30  # meters
    
    ridge_pattern = ridge_amplitude * np.sin(2 * np.pi * y_rot / ridge_wavelength)
    
    # Add smaller-scale variations
    fine_variation = 10 * np.sin(2 * np.pi * x_rot / 500) * np.cos(2 * np.pi * y_rot / 400)
    
    # Add some random roughness
    np.random.seed(42)  # For reproducibility
    roughness = np.random.randn(resolution, resolution) * 3
    roughness = gaussian_filter(roughness, sigma=3)
    
    # Combine all components
    elevation = base_elevation + ridge_pattern + fine_variation + roughness
    
    # Ensure elevations are realistic (between 200m and 400m for this region)
    elevation = np.clip(elevation, 220, 380)
    
    # Smooth the result
    elevation = gaussian_filter(elevation, sigma=1.5)
    
    return lon, lat, elevation


# =============================================================================
# 3D VISUALIZATION FUNCTIONS
# =============================================================================

def plot_3d_terrain_basic(lon, lat, elevation, save_path=None):
    """
    Create a basic 3D terrain visualization.
    """
    LON, LAT = np.meshgrid(lon, lat)
    
    fig = plt.figure(figsize=(16, 12))
    ax = fig.add_subplot(111, projection='3d')
    
    # Create light source for shading
    ls = LightSource(azdeg=315, altdeg=45)
    
    # Normalize elevation for colormap
    norm = Normalize(vmin=elevation.min(), vmax=elevation.max())
    
    # Create shaded surface
    colors = cm.terrain(norm(elevation))
    shaded = ls.shade(elevation, cmap=cm.terrain, blend_mode='soft')
    
    # Plot surface
    surf = ax.plot_surface(
        LON, LAT, elevation,
        facecolors=shaded,
        linewidth=0,
        antialiased=True,
        shade=False
    )
    
    # Plot tower locations - ALWAYS ON TOP
    for tower, data in TOWER_COORDINATES.items():
        # Find interpolated terrain height at tower location
        lon_idx = np.argmin(np.abs(lon - data['lon']))
        lat_idx = np.argmin(np.abs(lat - data['lat']))
        terrain_z = elevation[lat_idx, lon_idx]
        
        # Plot tower marker (elevated above terrain) - HIGH ZORDER
        tower_height = 30  # meters above ground
        ax.scatter(
            [data['lon']], [data['lat']], [terrain_z + tower_height],
            s=250, c=data['color'], marker='^', 
            edgecolors='white', linewidth=3, zorder=9999,
            label=f"{tower}: {data['name']}"
        )
        
        # Draw tower "pole"
        ax.plot(
            [data['lon'], data['lon']],
            [data['lat'], data['lat']],
            [terrain_z, terrain_z + tower_height],
            color=data['color'], linewidth=3, alpha=0.9, zorder=9998
        )
        
        # Add label - ALWAYS VISIBLE with high contrast background
        ax.text(
            data['lon'], data['lat'], terrain_z + tower_height + 12,
            tower, fontsize=12, fontweight='bold',
            ha='center', va='bottom', zorder=10000,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                     edgecolor=data['color'], linewidth=2, alpha=0.95)
        )
    
    # Labels and title
    ax.set_xlabel('Longitude (°W)', fontsize=12, labelpad=10)
    ax.set_ylabel('Latitude (°N)', fontsize=12, labelpad=10)
    ax.set_zlabel('Elevation (m)', fontsize=12, labelpad=10)
    
    ax.set_title(
        f'3D Terrain Visualization\n{REGION_INFO["name"]}\n{REGION_INFO["terrain"]}',
        fontsize=14, fontweight='bold'
    )
    
    # Adjust view angle
    ax.view_init(elev=35, azim=225)
    
    # Add legend
    ax.legend(loc='upper left', fontsize=8, framealpha=0.9)
    
    # Add colorbar
    mappable = cm.ScalarMappable(norm=norm, cmap=cm.terrain)
    mappable.set_array(elevation)
    cbar = plt.colorbar(mappable, ax=ax, shrink=0.5, aspect=20, pad=0.1)
    cbar.set_label('Elevation (m)', fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    plt.close()


def plot_3d_terrain_detailed(lon, lat, elevation, save_path=None):
    """
    Create a detailed 3D terrain visualization with multiple view angles.
    """
    LON, LAT = np.meshgrid(lon, lat)
    
    # Create figure with multiple subplots for different views
    fig = plt.figure(figsize=(20, 16))
    
    # Light source for hillshade effect
    ls = LightSource(azdeg=315, altdeg=45)
    norm = Normalize(vmin=elevation.min(), vmax=elevation.max())
    shaded = ls.shade(elevation, cmap=cm.terrain, blend_mode='soft')
    
    # View angles: (elevation_angle, azimuth_angle, title)
    views = [
        (35, 225, 'Southwest View'),
        (35, 135, 'Southeast View'),
        (35, 315, 'Northwest View'),
        (60, 270, 'West View (High Angle)'),
    ]
    
    for idx, (elev_angle, azim, title) in enumerate(views, 1):
        ax = fig.add_subplot(2, 2, idx, projection='3d')
        
        # Plot surface
        ax.plot_surface(
            LON, LAT, elevation,
            facecolors=shaded,
            linewidth=0,
            antialiased=True,
            shade=False
        )
        
        # Plot towers - ALWAYS ON TOP
        for tower, data in TOWER_COORDINATES.items():
            lon_idx = np.argmin(np.abs(lon - data['lon']))
            lat_idx = np.argmin(np.abs(lat - data['lat']))
            terrain_z = elevation[lat_idx, lon_idx]
            tower_height = 25
            
            ax.scatter(
                [data['lon']], [data['lat']], [terrain_z + tower_height],
                s=180, c=data['color'], marker='^',
                edgecolors='white', linewidth=2, zorder=9999
            )
            ax.plot(
                [data['lon'], data['lon']],
                [data['lat'], data['lat']],
                [terrain_z, terrain_z + tower_height],
                color=data['color'], linewidth=2.5, alpha=0.9, zorder=9998
            )
            
            # Labels on ALL subplots - ALWAYS VISIBLE
            ax.text(
                data['lon'], data['lat'], terrain_z + tower_height + 10,
                tower, fontsize=10, fontweight='bold', ha='center', zorder=10000,
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white', 
                         edgecolor=data['color'], linewidth=1.5, alpha=0.95)
            )
        
        ax.set_xlabel('Lon', fontsize=9)
        ax.set_ylabel('Lat', fontsize=9)
        ax.set_zlabel('Elev (m)', fontsize=9)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.view_init(elev=elev_angle, azim=azim)
    
    fig.suptitle(
        f'3D Terrain - Multiple Views\n{REGION_INFO["name"]}',
        fontsize=14, fontweight='bold', y=1.02
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    plt.close()


def plot_3d_terrain_with_contours(lon, lat, elevation, save_path=None):
    """
    3D terrain with contour lines projected onto the surface.
    """
    LON, LAT = np.meshgrid(lon, lat)
    
    fig = plt.figure(figsize=(18, 14))
    ax = fig.add_subplot(111, projection='3d')
    
    # Light source
    ls = LightSource(azdeg=315, altdeg=45)
    norm = Normalize(vmin=elevation.min(), vmax=elevation.max())
    shaded = ls.shade(elevation, cmap=cm.terrain, blend_mode='soft')
    
    # Plot surface
    ax.plot_surface(
        LON, LAT, elevation,
        facecolors=shaded,
        linewidth=0,
        antialiased=True,
        shade=False,
        alpha=0.9
    )
    
    # Add contour lines at base
    z_base = elevation.min() - 20
    contour_levels = np.arange(
        np.floor(elevation.min() / 20) * 20,
        np.ceil(elevation.max() / 20) * 20 + 1,
        20
    )
    
    # Contours on XY plane (base)
    ax.contour(
        LON, LAT, elevation,
        levels=contour_levels,
        zdir='z', offset=z_base,
        cmap='gray_r', alpha=0.5, linewidths=0.5
    )
    
    # Contours on the surface
    ax.contour(
        LON, LAT, elevation,
        levels=contour_levels,
        colors='black', alpha=0.3, linewidths=0.3
    )
    
    # Plot towers with detailed info - ALWAYS ON TOP
    for tower, data in TOWER_COORDINATES.items():
        lon_idx = np.argmin(np.abs(lon - data['lon']))
        lat_idx = np.argmin(np.abs(lat - data['lat']))
        terrain_z = elevation[lat_idx, lon_idx]
        tower_height = 30
        
        # Tower marker - HIGH ZORDER
        ax.scatter(
            [data['lon']], [data['lat']], [terrain_z + tower_height],
            s=250, c=data['color'], marker='^',
            edgecolors='white', linewidth=3, zorder=9999
        )
        
        # Tower pole
        ax.plot(
            [data['lon'], data['lon']],
            [data['lat'], data['lat']],
            [terrain_z, terrain_z + tower_height],
            color=data['color'], linewidth=3, alpha=0.9, zorder=9998
        )
        
        # Projection line to base
        ax.plot(
            [data['lon'], data['lon']],
            [data['lat'], data['lat']],
            [z_base, terrain_z],
            color='gray', linewidth=1, linestyle='--', alpha=0.5
        )
        
        # Base marker
        ax.scatter(
            [data['lon']], [data['lat']], [z_base],
            s=50, c='gray', marker='o', alpha=0.5
        )
        
        # Label - ALWAYS VISIBLE ON TOP
        ax.text(
            data['lon'], data['lat'], terrain_z + tower_height + 15,
            f"{tower}\n{data['terrain_type']}\n{terrain_z:.0f}m",
            fontsize=10, fontweight='bold', ha='center', zorder=10000,
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                     edgecolor=data['color'], linewidth=2, alpha=0.95)
        )
    
    ax.set_xlabel('Longitude (°W)', fontsize=12, labelpad=10)
    ax.set_ylabel('Latitude (°N)', fontsize=12, labelpad=10)
    ax.set_zlabel('Elevation (m)', fontsize=12, labelpad=10)
    
    ax.set_title(
        f'3D Terrain with Contours\n{REGION_INFO["name"]}\n{REGION_INFO["terrain"]}',
        fontsize=14, fontweight='bold'
    )
    
    ax.view_init(elev=30, azim=220)
    ax.set_zlim(z_base, elevation.max() + 50)
    
    # Colorbar
    mappable = cm.ScalarMappable(norm=norm, cmap=cm.terrain)
    mappable.set_array(elevation)
    cbar = plt.colorbar(mappable, ax=ax, shrink=0.5, aspect=20, pad=0.08)
    cbar.set_label('Elevation (m)', fontsize=11)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    plt.close()


def plot_2d_hillshade_map(lon, lat, elevation, save_path=None):
    """
    Create a 2D hillshade map with tower locations.
    """
    LON, LAT = np.meshgrid(lon, lat)
    
    fig, ax = plt.subplots(figsize=(14, 12))
    
    # Create hillshade
    ls = LightSource(azdeg=315, altdeg=45)
    hillshade = ls.hillshade(elevation, vert_exag=2)
    
    # Plot hillshade
    ax.imshow(
        hillshade, extent=[lon.min(), lon.max(), lat.min(), lat.max()],
        cmap='gray', origin='lower', alpha=0.5
    )
    
    # Overlay elevation colors
    im = ax.imshow(
        elevation, extent=[lon.min(), lon.max(), lat.min(), lat.max()],
        cmap='terrain', origin='lower', alpha=0.6
    )
    
    # Add contours
    contour_levels = np.arange(
        np.floor(elevation.min() / 10) * 10,
        np.ceil(elevation.max() / 10) * 10 + 1,
        10
    )
    cs = ax.contour(
        LON, LAT, elevation,
        levels=contour_levels,
        colors='black', alpha=0.4, linewidths=0.5
    )
    ax.clabel(cs, inline=True, fontsize=7, fmt='%d m')
    
    # Plot towers - ALWAYS ON TOP
    for tower, data in TOWER_COORDINATES.items():
        ax.scatter(
            data['lon'], data['lat'],
            s=250, c=data['color'], marker='^',
            edgecolors='white', linewidth=3, zorder=9999,
            label=f"{tower}: {data['name']}"
        )
        ax.annotate(
            f"{tower}\n({data['elevation_m']}m)",
            (data['lon'], data['lat']),
            xytext=(12, 12), textcoords='offset points',
            fontsize=11, fontweight='bold', zorder=10000,
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                     edgecolor=data['color'], linewidth=2, alpha=0.95)
        )
    
    ax.set_xlabel('Longitude (°W)', fontsize=12)
    ax.set_ylabel('Latitude (°N)', fontsize=12)
    ax.set_title(
        f'Hillshade Map with Tower Locations\n{REGION_INFO["name"]}',
        fontsize=14, fontweight='bold'
    )
    
    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8, aspect=30)
    cbar.set_label('Elevation (m)', fontsize=11)
    
    # Legend
    ax.legend(loc='lower right', fontsize=8, framealpha=0.9)
    
    # Add scale bar (approximate)
    scale_lon = lon.min() + 0.002
    scale_lat = lat.min() + 0.001
    scale_length = 0.009  # ~1 km at this latitude
    ax.plot([scale_lon, scale_lon + scale_length], [scale_lat, scale_lat], 
            'k-', linewidth=3)
    ax.text(scale_lon + scale_length/2, scale_lat + 0.0005, '~1 km',
            ha='center', fontsize=9, fontweight='bold')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    plt.close()


def plot_terrain_cross_sections(lon, lat, elevation, save_path=None):
    """
    Plot terrain cross-sections through the study area.
    """
    LON, LAT = np.meshgrid(lon, lat)
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    # Cross-section 1: West to East (through center)
    ax1 = axes[0, 0]
    center_lat_idx = len(lat) // 2
    ax1.fill_between(lon, elevation[center_lat_idx, :], alpha=0.6, color='forestgreen')
    ax1.plot(lon, elevation[center_lat_idx, :], 'k-', linewidth=2)
    
    # Mark towers on this cross-section
    for tower, data in TOWER_COORDINATES.items():
        if abs(data['lat'] - lat[center_lat_idx]) < 0.002:
            lon_idx = np.argmin(np.abs(lon - data['lon']))
            ax1.scatter([data['lon']], [elevation[center_lat_idx, lon_idx]],
                       s=150, c=data['color'], marker='^', edgecolors='black',
                       linewidth=2, zorder=10, label=tower)
    
    ax1.set_xlabel('Longitude (°W)', fontsize=11)
    ax1.set_ylabel('Elevation (m)', fontsize=11)
    ax1.set_title(f'W-E Cross Section (Lat ≈ {lat[center_lat_idx]:.4f}°N)', fontweight='bold')
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8)
    
    # Cross-section 2: South to North (through center)
    ax2 = axes[0, 1]
    center_lon_idx = len(lon) // 2
    ax2.fill_between(lat, elevation[:, center_lon_idx], alpha=0.6, color='saddlebrown')
    ax2.plot(lat, elevation[:, center_lon_idx], 'k-', linewidth=2)
    
    for tower, data in TOWER_COORDINATES.items():
        if abs(data['lon'] - lon[center_lon_idx]) < 0.002:
            lat_idx = np.argmin(np.abs(lat - data['lat']))
            ax2.scatter([data['lat']], [elevation[lat_idx, center_lon_idx]],
                       s=150, c=data['color'], marker='^', edgecolors='black',
                       linewidth=2, zorder=10, label=tower)
    
    ax2.set_xlabel('Latitude (°N)', fontsize=11)
    ax2.set_ylabel('Elevation (m)', fontsize=11)
    ax2.set_title(f'S-N Cross Section (Lon ≈ {lon[center_lon_idx]:.4f}°W)', fontweight='bold')
    ax2.grid(True, alpha=0.3)
    ax2.legend(fontsize=8)
    
    # Cross-section 3: SW to NE diagonal (along ridge direction)
    ax3 = axes[1, 0]
    diag_indices = [(i, i) for i in range(min(len(lat), len(lon)))]
    diag_dist = np.linspace(0, 1, len(diag_indices))
    diag_elev = [elevation[i, j] for i, j in diag_indices]
    
    ax3.fill_between(diag_dist, diag_elev, alpha=0.6, color='steelblue')
    ax3.plot(diag_dist, diag_elev, 'k-', linewidth=2)
    ax3.set_xlabel('Relative Distance (SW → NE)', fontsize=11)
    ax3.set_ylabel('Elevation (m)', fontsize=11)
    ax3.set_title('SW-NE Cross Section (Ridge Direction)', fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # Elevation histogram
    ax4 = axes[1, 1]
    ax4.hist(elevation.flatten(), bins=50, color='olivedrab', edgecolor='black', alpha=0.7)
    
    # Mark tower elevations
    for tower, data in TOWER_COORDINATES.items():
        ax4.axvline(data['elevation_m'], color=data['color'], linewidth=2,
                   linestyle='--', label=f"{tower}: {data['elevation_m']}m")
    
    ax4.set_xlabel('Elevation (m)', fontsize=11)
    ax4.set_ylabel('Frequency', fontsize=11)
    ax4.set_title('Elevation Distribution', fontweight='bold')
    ax4.legend(fontsize=8, loc='upper right')
    ax4.grid(True, alpha=0.3)
    
    fig.suptitle(
        f'Terrain Cross-Sections - {REGION_INFO["name"]}',
        fontsize=14, fontweight='bold', y=1.02
    )
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    plt.close()


def create_interactive_html_plot(lon, lat, elevation, output_path='terrain_3d.html'):
    """
    Create an interactive 3D terrain plot using Plotly (if available).
    """
    try:
        import plotly.graph_objects as go
        
        LON, LAT = np.meshgrid(lon, lat)
        
        # Downsample for performance if needed
        step = max(1, len(lon) // 100)
        
        fig = go.Figure()
        
        # Add terrain surface
        fig.add_trace(go.Surface(
            x=LON[::step, ::step],
            y=LAT[::step, ::step],
            z=elevation[::step, ::step],
            colorscale='earth',
            name='Terrain',
            showscale=True,
            colorbar=dict(title='Elevation (m)')
        ))
        
        # Add tower markers
        for tower, data in TOWER_COORDINATES.items():
            lon_idx = np.argmin(np.abs(lon - data['lon']))
            lat_idx = np.argmin(np.abs(lat - data['lat']))
            terrain_z = elevation[lat_idx, lon_idx]
            
            fig.add_trace(go.Scatter3d(
                x=[data['lon']],
                y=[data['lat']],
                z=[terrain_z + 20],
                mode='markers+text',
                marker=dict(size=10, color=data['color'], symbol='diamond'),
                text=[f"{tower}<br>{data['terrain_type']}<br>{terrain_z:.0f}m"],
                textposition='top center',
                name=tower
            ))
        
        fig.update_layout(
            title=f'Interactive 3D Terrain - {REGION_INFO["name"]}',
            scene=dict(
                xaxis_title='Longitude (°W)',
                yaxis_title='Latitude (°N)',
                zaxis_title='Elevation (m)',
                aspectmode='manual',
                aspectratio=dict(x=1.5, y=1.5, z=0.5)
            ),
            width=1200,
            height=800
        )
        
        fig.write_html(output_path)
        print(f"Interactive plot saved: {output_path}")
        
        return fig
        
    except ImportError:
        print("Plotly not installed. Install with: pip install plotly")
        return None


def fetch_satellite_image(bounds, zoom=14, output_dir='elevation_data'):
    """
    Fetch satellite/aerial imagery from free tile servers.
    
    Uses OpenStreetMap tiles or ESRI World Imagery (free for non-commercial use).
    """
    if not HAS_REQUESTS:
        print("ERROR: 'requests' library required for satellite imagery")
        return None
    
    import io
    from PIL import Image
    
    print("Fetching satellite imagery...")
    
    # Calculate tile coordinates
    def lat_lon_to_tile(lat, lon, zoom):
        n = 2 ** zoom
        x = int((lon + 180) / 360 * n)
        y = int((1 - np.log(np.tan(np.radians(lat)) + 1/np.cos(np.radians(lat))) / np.pi) / 2 * n)
        return x, y
    
    # Get tile range for bounds
    x_min, y_max = lat_lon_to_tile(bounds['min_lat'], bounds['min_lon'], zoom)
    x_max, y_min = lat_lon_to_tile(bounds['max_lat'], bounds['max_lon'], zoom)
    
    # ESRI World Imagery (high quality satellite)
    # tile_url = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    
    # OpenStreetMap (fallback)
    tile_url = "https://tile.openstreetmap.org/{z}/{x}/{y}.png"
    
    # Alternative: ESRI Satellite
    esri_url = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
    
    tiles = []
    tile_size = 256
    
    print(f"  Downloading {(x_max - x_min + 1) * (y_max - y_min + 1)} tiles...")
    
    for y in range(y_min, y_max + 1):
        row = []
        for x in range(x_min, x_max + 1):
            url = esri_url.format(z=zoom, x=x, y=y)
            try:
                response = requests.get(url, headers={'User-Agent': 'Mozilla/5.0'}, timeout=10)
                if response.status_code == 200:
                    img = Image.open(io.BytesIO(response.content))
                    row.append(np.array(img))
                else:
                    # White placeholder
                    row.append(np.ones((tile_size, tile_size, 3), dtype=np.uint8) * 255)
            except Exception as e:
                print(f"  Tile error: {e}")
                row.append(np.ones((tile_size, tile_size, 3), dtype=np.uint8) * 255)
        tiles.append(row)
    
    # Stitch tiles together
    rows = [np.hstack(row) for row in tiles]
    satellite_img = np.vstack(rows)
    
    print(f"  SUCCESS! Image size: {satellite_img.shape}")
    
    # Calculate geographic extent of the stitched image
    def tile_to_lat_lon(x, y, zoom):
        n = 2 ** zoom
        lon = x / n * 360 - 180
        lat = np.degrees(np.arctan(np.sinh(np.pi * (1 - 2 * y / n))))
        return lat, lon
    
    lat_max, lon_min = tile_to_lat_lon(x_min, y_min, zoom)
    lat_min, lon_max = tile_to_lat_lon(x_max + 1, y_max + 1, zoom)
    
    extent = [lon_min, lon_max, lat_min, lat_max]
    
    return satellite_img, extent


def plot_satellite_with_towers(bounds, save_path=None):
    """
    Create a satellite image map with tower locations overlaid.
    """
    try:
        from PIL import Image
    except ImportError:
        print("PIL/Pillow required. Install with: pip install Pillow")
        return
    
    # Fetch satellite imagery
    satellite_img, extent = fetch_satellite_image(bounds, zoom=15)
    
    if satellite_img is None:
        print("Could not fetch satellite imagery")
        return
    
    fig, ax = plt.subplots(figsize=(16, 14))
    
    # Plot satellite image
    ax.imshow(satellite_img, extent=extent, origin='upper', aspect='auto')
    
    # Plot towers with detailed labels - ALWAYS ON TOP
    for tower, data in TOWER_COORDINATES.items():
        # Main marker - HIGH ZORDER
        ax.scatter(
            data['lon'], data['lat'],
            s=350, c=data['color'], marker='^',
            edgecolors='white', linewidth=4, zorder=9999
        )
        
        # Tower label with details - ALWAYS VISIBLE
        label = f"{tower}\n{data['name'].split(' - ')[1]}\n{data['elevation_m']}m"
        ax.annotate(
            label,
            (data['lon'], data['lat']),
            xytext=(18, 18), textcoords='offset points',
            fontsize=11, fontweight='bold',
            color='white', zorder=10000,
            bbox=dict(boxstyle='round,pad=0.5', facecolor=data['color'], 
                     edgecolor='white', linewidth=3, alpha=0.98),
            arrowprops=dict(arrowstyle='->', color='white', lw=3)
        )
    
    # Add title and labels
    ax.set_xlabel('Longitude (°W)', fontsize=12, color='white')
    ax.set_ylabel('Latitude (°N)', fontsize=12, color='white')
    ax.set_title(
        f'Satellite View - ORNL Weather Tower Network\n'
        f'{REGION_INFO["name"]}, {REGION_INFO["state"]}',
        fontsize=14, fontweight='bold', color='white',
        bbox=dict(facecolor='black', alpha=0.7, edgecolor='none', pad=10)
    )
    
    # Style the axes
    ax.tick_params(colors='white')
    for spine in ax.spines.values():
        spine.set_color('white')
    
    # Add legend with tower info
    legend_text = "TOWER NETWORK:\n" + "-"*30 + "\n"
    for tower, data in sorted(TOWER_COORDINATES.items()):
        legend_text += f"▲ {tower}: {data['terrain_type']}, {data['elevation_m']}m\n"
        legend_text += f"    {data.get('land_use', 'N/A')}, {data.get('canopy_cover_pct', 'N/A')}% canopy\n"
    
    ax.text(0.02, 0.02, legend_text, transform=ax.transAxes, fontsize=8,
            verticalalignment='bottom', fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='black', alpha=0.8),
            color='white')
    
    # Add scale bar
    scale_lat = bounds['min_lat'] + 0.001
    scale_lon = bounds['min_lon'] + 0.002
    scale_km = 0.5  # 500m
    scale_deg = scale_km / (111 * np.cos(np.radians(scale_lat)))
    
    ax.plot([scale_lon, scale_lon + scale_deg], [scale_lat, scale_lat], 
            'w-', linewidth=4)
    ax.plot([scale_lon, scale_lon + scale_deg], [scale_lat, scale_lat], 
            'k-', linewidth=2)
    ax.text(scale_lon + scale_deg/2, scale_lat + 0.0008, f'{scale_km} km',
            ha='center', fontsize=10, fontweight='bold', color='white',
            bbox=dict(facecolor='black', alpha=0.7, edgecolor='none'))
    
    # Add north arrow
    arrow_lat = bounds['max_lat'] - 0.002
    arrow_lon = bounds['max_lon'] - 0.003
    ax.annotate('N', xy=(arrow_lon, arrow_lat), xytext=(arrow_lon, arrow_lat - 0.003),
                fontsize=14, fontweight='bold', color='white', ha='center',
                arrowprops=dict(arrowstyle='->', color='white', lw=2))
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight', facecolor='black')
        print(f"Saved: {save_path}")
    
    plt.show()
    plt.close()


def plot_terrain_with_satellite_overlay(lon, lat, elevation, bounds, save_path=None):
    """
    Create a combined plot with terrain elevation and satellite imagery side by side.
    """
    try:
        from PIL import Image
    except ImportError:
        print("PIL/Pillow required. Install with: pip install Pillow")
        return
    
    # Fetch satellite imagery
    satellite_img, sat_extent = fetch_satellite_image(bounds, zoom=14)
    
    if satellite_img is None:
        print("Could not fetch satellite imagery")
        return
    
    LON, LAT = np.meshgrid(lon, lat)
    
    fig = plt.figure(figsize=(20, 10))
    
    # Left: Satellite image
    ax1 = fig.add_subplot(1, 2, 1)
    ax1.imshow(satellite_img, extent=sat_extent, origin='upper', aspect='auto')
    
    # Plot towers on satellite - ALWAYS ON TOP
    for tower, data in TOWER_COORDINATES.items():
        ax1.scatter(data['lon'], data['lat'], s=250, c=data['color'], marker='^',
                   edgecolors='white', linewidth=3, zorder=9999)
        ax1.annotate(tower, (data['lon'], data['lat']), xytext=(10, 10),
                    textcoords='offset points', fontsize=12, fontweight='bold',
                    color='white', zorder=10000,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor=data['color'], 
                             edgecolor='white', linewidth=2, alpha=0.95))
    
    ax1.set_xlabel('Longitude (°W)', fontsize=11)
    ax1.set_ylabel('Latitude (°N)', fontsize=11)
    ax1.set_title('Satellite Imagery (ESRI World Imagery)', fontsize=12, fontweight='bold')
    
    # Right: Terrain elevation with hillshade
    ax2 = fig.add_subplot(1, 2, 2)
    
    ls = LightSource(azdeg=315, altdeg=45)
    hillshade = ls.hillshade(elevation, vert_exag=2)
    
    ax2.imshow(hillshade, extent=[lon.min(), lon.max(), lat.min(), lat.max()],
               cmap='gray', origin='lower', alpha=0.4)
    im = ax2.imshow(elevation, extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                    cmap='terrain', origin='lower', alpha=0.7)
    
    # Contours
    contour_levels = np.arange(np.floor(elevation.min()/20)*20, 
                               np.ceil(elevation.max()/20)*20 + 1, 20)
    cs = ax2.contour(LON, LAT, elevation, levels=contour_levels,
                     colors='black', alpha=0.5, linewidths=0.5)
    ax2.clabel(cs, inline=True, fontsize=7, fmt='%d m')
    
    # Plot towers on terrain - ALWAYS ON TOP
    for tower, data in TOWER_COORDINATES.items():
        ax2.scatter(data['lon'], data['lat'], s=250, c=data['color'], marker='^',
                   edgecolors='white', linewidth=3, zorder=9999)
        ax2.annotate(f"{tower}\n{data['elevation_m']}m", (data['lon'], data['lat']),
                    xytext=(10, 10), textcoords='offset points', fontsize=11, fontweight='bold',
                    zorder=10000,
                    bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                             edgecolor=data['color'], linewidth=2, alpha=0.95))
    
    ax2.set_xlabel('Longitude (°W)', fontsize=11)
    ax2.set_ylabel('Latitude (°N)', fontsize=11)
    ax2.set_title('Terrain Elevation (Open-Elevation API)', fontsize=12, fontweight='bold')
    
    cbar = plt.colorbar(im, ax=ax2, shrink=0.8)
    cbar.set_label('Elevation (m)', fontsize=10)
    
    fig.suptitle(f'{REGION_INFO["name"]} - Satellite vs Terrain\n{REGION_INFO["terrain"]}',
                 fontsize=14, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    plt.close()


def plot_tower_characteristics(save_path=None):
    """
    Create a comprehensive summary plot of all tower characteristics.
    """
    towers = list(TOWER_COORDINATES.keys())
    n_towers = len(towers)
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    # Extract data
    elevations = [TOWER_COORDINATES[t]['elevation_m'] for t in towers]
    slopes = [TOWER_COORDINATES[t].get('slope_deg', 0) for t in towers]
    canopies = [TOWER_COORDINATES[t].get('canopy_cover_pct', 0) for t in towers]
    ridge_dist = [TOWER_COORDINATES[t].get('dist_to_ridge_m', 0) for t in towers]
    colors = [TOWER_COORDINATES[t]['color'] for t in towers]
    
    # 1. Elevation bar chart
    ax1 = axes[0, 0]
    bars = ax1.bar(towers, elevations, color=colors, edgecolor='black', linewidth=2)
    ax1.set_ylabel('Elevation (m)', fontsize=11)
    ax1.set_title('Tower Elevations', fontsize=12, fontweight='bold')
    ax1.set_ylim(250, 360)
    for bar, elev in zip(bars, elevations):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                f'{elev}m', ha='center', fontsize=10, fontweight='bold')
    ax1.grid(axis='y', alpha=0.3)
    
    # 2. Slope comparison
    ax2 = axes[0, 1]
    bars = ax2.bar(towers, slopes, color=colors, edgecolor='black', linewidth=2)
    ax2.set_ylabel('Terrain Slope (°)', fontsize=11)
    ax2.set_title('Terrain Slopes', fontsize=12, fontweight='bold')
    for bar, slope in zip(bars, slopes):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.2, 
                f'{slope}°', ha='center', fontsize=10, fontweight='bold')
    ax2.grid(axis='y', alpha=0.3)
    
    # 3. Canopy cover
    ax3 = axes[0, 2]
    bars = ax3.bar(towers, canopies, color=colors, edgecolor='black', linewidth=2)
    ax3.set_ylabel('Canopy Cover (%)', fontsize=11)
    ax3.set_title('Vegetation Canopy Coverage', fontsize=12, fontweight='bold')
    for bar, can in zip(bars, canopies):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1, 
                f'{can}%', ha='center', fontsize=10, fontweight='bold')
    ax3.grid(axis='y', alpha=0.3)
    
    # 4. Distance to ridge
    ax4 = axes[1, 0]
    bars = ax4.bar(towers, ridge_dist, color=colors, edgecolor='black', linewidth=2)
    ax4.set_ylabel('Distance to Ridge (m)', fontsize=11)
    ax4.set_title('Distance to Nearest Ridge', fontsize=12, fontweight='bold')
    for bar, dist in zip(bars, ridge_dist):
        ax4.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 10, 
                f'{dist}m', ha='center', fontsize=10, fontweight='bold')
    ax4.grid(axis='y', alpha=0.3)
    
    # 5. Terrain type pie chart
    ax5 = axes[1, 1]
    terrain_types = [TOWER_COORDINATES[t]['terrain_type'] for t in towers]
    terrain_counts = {}
    terrain_colors = {}
    for t, terrain in zip(towers, terrain_types):
        if terrain not in terrain_counts:
            terrain_counts[terrain] = 0
            terrain_colors[terrain] = TOWER_COORDINATES[t]['color']
        terrain_counts[terrain] += 1
    
    wedges, texts, autotexts = ax5.pie(
        terrain_counts.values(), 
        labels=terrain_counts.keys(),
        colors=[terrain_colors[t] for t in terrain_counts.keys()],
        autopct='%1.0f%%',
        explode=[0.05]*len(terrain_counts),
        shadow=True
    )
    ax5.set_title('Terrain Type Distribution', fontsize=12, fontweight='bold')
    
    # 6. Land use summary table
    ax6 = axes[1, 2]
    ax6.axis('off')
    
    table_data = []
    for tower in towers:
        data = TOWER_COORDINATES[tower]
        table_data.append([
            tower,
            data['terrain_type'],
            data.get('land_use', 'N/A'),
            data.get('soil_type', 'N/A'),
            f"{data.get('aspect_deg', 0)}°"
        ])
    
    table = ax6.table(
        cellText=table_data,
        colLabels=['Tower', 'Terrain', 'Land Use', 'Soil Type', 'Aspect'],
        cellLoc='center',
        loc='center',
        colColours=['lightgray']*5
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    
    # Color code tower cells
    for i, tower in enumerate(towers):
        table[(i+1, 0)].set_facecolor(TOWER_COORDINATES[tower]['color'])
        table[(i+1, 0)].set_text_props(color='white', fontweight='bold')
    
    ax6.set_title('Tower Characteristics Summary', fontsize=12, fontweight='bold', pad=20)
    
    fig.suptitle('ORNL Weather Tower Network - Site Characteristics',
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    plt.close()


# =============================================================================
# ELEVATION ZONE CLASSIFICATION (Basin, Plain, Hill)
# =============================================================================

def classify_elevation_zones(elevation, method='percentile'):
    """
    Classify terrain into elevation zones: Basin, Plain, Hill (and optionally more).
    
    Parameters:
        elevation: 2D numpy array of elevation data
        method: Classification method
            - 'percentile': Use data percentiles (adapts to local terrain)
            - 'absolute': Use absolute elevation thresholds
            - 'relative': Use relative height from min elevation
    
    Returns:
        zones: 2D array with zone classifications (0=Basin, 1=Plain, 2=Hill, etc.)
        zone_info: Dictionary with zone metadata
    """
    elev_min = elevation.min()
    elev_max = elevation.max()
    elev_range = elev_max - elev_min
    
    if method == 'percentile':
        # Adaptive classification based on data distribution
        p20 = np.percentile(elevation, 20)
        p40 = np.percentile(elevation, 40)
        p60 = np.percentile(elevation, 60)
        p80 = np.percentile(elevation, 80)
        
        zone_info = {
            'Valley/Basin': {'range': (elev_min, p20), 'color': '#1a5276', 'description': 'Lowest 20% - Valley floors and basins'},
            'Lower Plain': {'range': (p20, p40), 'color': '#2e86ab', 'description': '20-40% - Lower plains and gentle slopes'},
            'Upper Plain': {'range': (p40, p60), 'color': '#a8d5ba', 'description': '40-60% - Upper plains and moderate terrain'},
            'Foothill': {'range': (p60, p80), 'color': '#f4a261', 'description': '60-80% - Foothills and elevated areas'},
            'Ridge/Hill': {'range': (p80, elev_max), 'color': '#8b4513', 'description': 'Top 20% - Ridges and hilltops'},
        }
        
        zones = np.zeros_like(elevation, dtype=int)
        zones[elevation <= p20] = 0
        zones[(elevation > p20) & (elevation <= p40)] = 1
        zones[(elevation > p40) & (elevation <= p60)] = 2
        zones[(elevation > p60) & (elevation <= p80)] = 3
        zones[elevation > p80] = 4
        
    elif method == 'relative':
        # Classification based on relative height
        thresholds = [0.15, 0.35, 0.55, 0.75]  # Relative thresholds
        
        rel_elev = (elevation - elev_min) / elev_range
        
        zone_info = {
            'Valley/Basin': {'range': (elev_min, elev_min + 0.15*elev_range), 'color': '#1a5276', 
                           'description': f'0-15% relative height ({elev_min:.0f}-{elev_min + 0.15*elev_range:.0f}m)'},
            'Lower Plain': {'range': (elev_min + 0.15*elev_range, elev_min + 0.35*elev_range), 'color': '#2e86ab',
                          'description': f'15-35% relative height'},
            'Upper Plain': {'range': (elev_min + 0.35*elev_range, elev_min + 0.55*elev_range), 'color': '#a8d5ba',
                          'description': f'35-55% relative height'},
            'Foothill': {'range': (elev_min + 0.55*elev_range, elev_min + 0.75*elev_range), 'color': '#f4a261',
                        'description': f'55-75% relative height'},
            'Ridge/Hill': {'range': (elev_min + 0.75*elev_range, elev_max), 'color': '#8b4513',
                         'description': f'75-100% relative height ({elev_min + 0.75*elev_range:.0f}-{elev_max:.0f}m)'},
        }
        
        zones = np.zeros_like(elevation, dtype=int)
        zones[rel_elev <= 0.15] = 0
        zones[(rel_elev > 0.15) & (rel_elev <= 0.35)] = 1
        zones[(rel_elev > 0.35) & (rel_elev <= 0.55)] = 2
        zones[(rel_elev > 0.55) & (rel_elev <= 0.75)] = 3
        zones[rel_elev > 0.75] = 4
        
    else:  # absolute method - specific to this region
        # For ORNL region (elevation ~237-363m), use appropriate thresholds
        zone_info = {
            'Valley/Basin': {'range': (0, 260), 'color': '#1a5276', 'description': 'Below 260m - Valley floors'},
            'Lower Plain': {'range': (260, 290), 'color': '#2e86ab', 'description': '260-290m - Lower terrain'},
            'Upper Plain': {'range': (290, 320), 'color': '#a8d5ba', 'description': '290-320m - Mid-elevation'},
            'Foothill': {'range': (320, 350), 'color': '#f4a261', 'description': '320-350m - Elevated terrain'},
            'Ridge/Hill': {'range': (350, 500), 'color': '#8b4513', 'description': 'Above 350m - Ridge tops'},
        }
        
        zones = np.zeros_like(elevation, dtype=int)
        zones[elevation < 260] = 0
        zones[(elevation >= 260) & (elevation < 290)] = 1
        zones[(elevation >= 290) & (elevation < 320)] = 2
        zones[(elevation >= 320) & (elevation < 350)] = 3
        zones[elevation >= 350] = 4
    
    return zones, zone_info


def plot_elevation_zones_2d(lon, lat, elevation, method='percentile', save_path=None):
    """
    Create a 2D map showing elevation zones (Basin, Plain, Hill, etc.).
    """
    LON, LAT = np.meshgrid(lon, lat)
    
    # Classify elevation zones
    zones, zone_info = classify_elevation_zones(elevation, method=method)
    zone_names = list(zone_info.keys())
    zone_colors = [zone_info[name]['color'] for name in zone_names]
    
    # Create custom colormap
    from matplotlib.colors import ListedColormap, BoundaryNorm
    cmap = ListedColormap(zone_colors)
    bounds = np.arange(-0.5, len(zone_names) + 0.5, 1)
    norm = BoundaryNorm(bounds, cmap.N)
    
    fig, axes = plt.subplots(1, 2, figsize=(20, 10))
    
    # Left: Zone classification map
    ax1 = axes[0]
    
    # Add hillshade for depth
    ls = LightSource(azdeg=315, altdeg=45)
    hillshade = ls.hillshade(elevation, vert_exag=2)
    ax1.imshow(hillshade, extent=[lon.min(), lon.max(), lat.min(), lat.max()],
               cmap='gray', origin='lower', alpha=0.3)
    
    # Plot zones
    im = ax1.imshow(zones, extent=[lon.min(), lon.max(), lat.min(), lat.max()],
                    cmap=cmap, norm=norm, origin='lower', alpha=0.8)
    
    # Add zone boundaries (contours)
    for i, name in enumerate(zone_names):
        elev_range = zone_info[name]['range']
        if i < len(zone_names) - 1:
            ax1.contour(LON, LAT, elevation, levels=[elev_range[1]], 
                       colors='white', linewidths=1.5, linestyles='-', alpha=0.8)
    
    # Plot towers - ALWAYS ON TOP
    for tower, data in TOWER_COORDINATES.items():
        ax1.scatter(data['lon'], data['lat'], s=280, c=data['color'], marker='^',
                   edgecolors='white', linewidth=3, zorder=9999)
        
        # Determine tower's zone
        lon_idx = np.argmin(np.abs(lon - data['lon']))
        lat_idx = np.argmin(np.abs(lat - data['lat']))
        tower_zone = zones[lat_idx, lon_idx]
        tower_zone_name = zone_names[tower_zone]
        
        ax1.annotate(f"{tower}\n{tower_zone_name}", (data['lon'], data['lat']),
                    xytext=(12, 12), textcoords='offset points', fontsize=11, fontweight='bold',
                    zorder=10000,
                    bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.98, 
                             edgecolor=data['color'], linewidth=2.5))
    
    ax1.set_xlabel('Longitude (°W)', fontsize=12)
    ax1.set_ylabel('Latitude (°N)', fontsize=12)
    ax1.set_title(f'Elevation Zone Classification\n{REGION_INFO["name"]}', 
                  fontsize=14, fontweight='bold')
    
    # Custom colorbar with zone labels
    cbar = plt.colorbar(im, ax=ax1, shrink=0.8, aspect=30, ticks=range(len(zone_names)))
    cbar.ax.set_yticklabels(zone_names, fontsize=10)
    cbar.set_label('Terrain Zone', fontsize=11)
    
    # Right: Zone statistics and legend
    ax2 = axes[1]
    ax2.axis('off')
    
    # Calculate zone statistics
    zone_stats = []
    total_cells = zones.size
    
    for i, name in enumerate(zone_names):
        zone_mask = zones == i
        zone_area_pct = np.sum(zone_mask) / total_cells * 100
        zone_elev = elevation[zone_mask]
        
        if len(zone_elev) > 0:
            zone_stats.append({
                'name': name,
                'area_pct': zone_area_pct,
                'elev_min': zone_elev.min(),
                'elev_max': zone_elev.max(),
                'elev_mean': zone_elev.mean(),
                'color': zone_info[name]['color'],
                'description': zone_info[name]['description']
            })
    
    # Create zone legend with details
    y_pos = 0.95
    ax2.text(0.5, y_pos, 'ELEVATION ZONES', fontsize=16, fontweight='bold', 
             ha='center', transform=ax2.transAxes)
    y_pos -= 0.05
    ax2.text(0.5, y_pos, f'Classification Method: {method.upper()}', fontsize=11, 
             ha='center', transform=ax2.transAxes, style='italic')
    y_pos -= 0.08
    
    for stat in zone_stats:
        # Zone color box
        rect = plt.Rectangle((0.05, y_pos - 0.02), 0.08, 0.04, 
                             facecolor=stat['color'], edgecolor='black', linewidth=2,
                             transform=ax2.transAxes)
        ax2.add_patch(rect)
        
        # Zone name and stats
        ax2.text(0.16, y_pos, stat['name'], fontsize=12, fontweight='bold',
                transform=ax2.transAxes, va='center')
        ax2.text(0.16, y_pos - 0.035, 
                f"  Area: {stat['area_pct']:.1f}% | Elev: {stat['elev_min']:.0f}-{stat['elev_max']:.0f}m (avg: {stat['elev_mean']:.0f}m)",
                fontsize=10, transform=ax2.transAxes, va='center')
        ax2.text(0.16, y_pos - 0.06, f"  {stat['description']}", fontsize=9, 
                transform=ax2.transAxes, va='center', style='italic', color='gray')
        
        y_pos -= 0.12
    
    # Add tower zone assignments
    y_pos -= 0.05
    ax2.text(0.5, y_pos, 'TOWER ZONE ASSIGNMENTS', fontsize=14, fontweight='bold',
             ha='center', transform=ax2.transAxes)
    y_pos -= 0.05
    
    for tower, data in TOWER_COORDINATES.items():
        lon_idx = np.argmin(np.abs(lon - data['lon']))
        lat_idx = np.argmin(np.abs(lat - data['lat']))
        tower_zone = zones[lat_idx, lon_idx]
        tower_zone_name = zone_names[tower_zone]
        tower_elev = elevation[lat_idx, lon_idx]
        
        ax2.text(0.1, y_pos, f"▲ {tower}:", fontsize=11, fontweight='bold',
                transform=ax2.transAxes, color=data['color'])
        ax2.text(0.25, y_pos, f"{tower_zone_name} ({tower_elev:.0f}m)", fontsize=11,
                transform=ax2.transAxes)
        y_pos -= 0.04
    
    fig.suptitle(f'Terrain Zone Analysis - Basin, Plain, Hill Classification',
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    plt.close()


def plot_elevation_zones_3d(lon, lat, elevation, method='percentile', save_path=None):
    """
    Create a 3D visualization showing elevation zones as layered planes.
    """
    LON, LAT = np.meshgrid(lon, lat)
    
    # Classify elevation zones
    zones, zone_info = classify_elevation_zones(elevation, method=method)
    zone_names = list(zone_info.keys())
    
    fig = plt.figure(figsize=(20, 16))
    
    # Main 3D plot
    ax = fig.add_subplot(111, projection='3d')
    
    # Light source for terrain
    ls = LightSource(azdeg=315, altdeg=45)
    
    # Plot each zone with different colors
    for i, name in enumerate(zone_names):
        zone_mask = zones == i
        zone_color = zone_info[name]['color']
        
        # Create masked elevation for this zone
        zone_elev = np.where(zone_mask, elevation, np.nan)
        
        # Plot zone surface
        ax.plot_surface(
            LON, LAT, zone_elev,
            color=zone_color,
            alpha=0.8,
            linewidth=0,
            antialiased=True,
            shade=True,
            label=name
        )
    
    # Add horizontal reference planes at zone boundaries
    z_base = elevation.min() - 30
    for i, name in enumerate(zone_names):
        elev_range = zone_info[name]['range']
        plane_z = elev_range[1] if i < len(zone_names) - 1 else elev_range[1]
        
        if plane_z > elevation.min() and plane_z < elevation.max():
            # Create semi-transparent plane
            plane_lon = np.array([[lon.min(), lon.max()], [lon.min(), lon.max()]])
            plane_lat = np.array([[lat.min(), lat.min()], [lat.max(), lat.max()]])
            plane_elev = np.full_like(plane_lon, plane_z)
            
            ax.plot_surface(plane_lon, plane_lat, plane_elev,
                           color=zone_info[name]['color'], alpha=0.15, 
                           linewidth=0, shade=False)
            
            # Add boundary label
            ax.text(lon.max() + 0.002, lat.mean(), plane_z,
                   f'{plane_z:.0f}m\n{name}', fontsize=8, ha='left',
                   color=zone_info[name]['color'], fontweight='bold')
    
    # Plot towers - ALWAYS ON TOP
    for tower, data in TOWER_COORDINATES.items():
        lon_idx = np.argmin(np.abs(lon - data['lon']))
        lat_idx = np.argmin(np.abs(lat - data['lat']))
        terrain_z = elevation[lat_idx, lon_idx]
        tower_height = 30
        
        # Tower marker - HIGH ZORDER
        ax.scatter([data['lon']], [data['lat']], [terrain_z + tower_height],
                  s=280, c=data['color'], marker='^',
                  edgecolors='white', linewidth=3, zorder=9999)
        
        # Tower pole
        ax.plot([data['lon'], data['lon']], [data['lat'], data['lat']],
               [terrain_z, terrain_z + tower_height],
               color=data['color'], linewidth=3.5, alpha=0.9, zorder=9998)
        
        # Vertical line to base
        ax.plot([data['lon'], data['lon']], [data['lat'], data['lat']],
               [z_base, terrain_z], color='gray', linewidth=1, 
               linestyle='--', alpha=0.4)
        
        # Label - ALWAYS VISIBLE ON TOP
        tower_zone = zones[lat_idx, lon_idx]
        ax.text(data['lon'], data['lat'], terrain_z + tower_height + 18,
               f"{tower}\n{zone_names[tower_zone]}", fontsize=11, fontweight='bold',
               ha='center', zorder=10000,
               bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                        edgecolor=data['color'], linewidth=2, alpha=0.98))
    
    ax.set_xlabel('Longitude (°W)', fontsize=12, labelpad=10)
    ax.set_ylabel('Latitude (°N)', fontsize=12, labelpad=10)
    ax.set_zlabel('Elevation (m)', fontsize=12, labelpad=10)
    
    ax.set_title(
        f'3D Elevation Zones - Basin, Plain, Hill Classification\n'
        f'{REGION_INFO["name"]}\nMethod: {method.upper()}',
        fontsize=14, fontweight='bold'
    )
    
    ax.view_init(elev=25, azim=225)
    ax.set_zlim(z_base, elevation.max() + 60)
    
    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=zone_info[name]['color'], 
                            edgecolor='black', label=name) 
                      for name in zone_names]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9, framealpha=0.9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    plt.close()


def plot_elevation_zones_stacked(lon, lat, elevation, save_path=None):
    """
    Create a stacked/exploded view showing each elevation zone as a separate layer.
    """
    LON, LAT = np.meshgrid(lon, lat)
    
    # Classify zones using percentile method
    zones, zone_info = classify_elevation_zones(elevation, method='percentile')
    zone_names = list(zone_info.keys())
    n_zones = len(zone_names)
    
    fig = plt.figure(figsize=(22, 14))
    
    # Calculate vertical offset for exploded view
    elev_range = elevation.max() - elevation.min()
    layer_offset = elev_range * 0.4  # Spacing between layers
    
    ax = fig.add_subplot(111, projection='3d')
    
    # Plot each zone as a separate elevated layer
    for i, name in enumerate(zone_names):
        zone_mask = zones == i
        zone_color = zone_info[name]['color']
        
        # Create masked and offset elevation
        zone_elev = np.where(zone_mask, elevation + i * layer_offset, np.nan)
        
        # Plot zone surface
        surf = ax.plot_surface(
            LON, LAT, zone_elev,
            color=zone_color,
            alpha=0.85,
            linewidth=0,
            antialiased=True,
            shade=True
        )
        
        # Add zone label
        zone_center_lat = lat.mean()
        zone_center_lon = lon.mean()
        zone_center_elev = elevation.mean() + i * layer_offset
        
        ax.text(lon.max() + 0.005, lat.mean(), zone_center_elev,
               f"{name}\n({zone_info[name]['range'][0]:.0f}-{zone_info[name]['range'][1]:.0f}m)",
               fontsize=10, fontweight='bold', color=zone_color,
               ha='left', va='center',
               bbox=dict(facecolor='white', alpha=0.8, edgecolor=zone_color))
    
    # Plot towers on appropriate layers - ALWAYS ON TOP
    for tower, data in TOWER_COORDINATES.items():
        lon_idx = np.argmin(np.abs(lon - data['lon']))
        lat_idx = np.argmin(np.abs(lat - data['lat']))
        terrain_z = elevation[lat_idx, lon_idx]
        tower_zone = zones[lat_idx, lon_idx]
        
        # Offset tower to its zone layer
        offset_z = terrain_z + tower_zone * layer_offset
        
        # Tower marker - HIGH ZORDER
        ax.scatter([data['lon']], [data['lat']], [offset_z + 20],
                  s=280, c=data['color'], marker='^',
                  edgecolors='white', linewidth=3, zorder=9999)
        
        # Tower pole within layer
        ax.plot([data['lon'], data['lon']], [data['lat'], data['lat']],
               [offset_z, offset_z + 20],
               color=data['color'], linewidth=3.5, zorder=9998)
        
        # Label - ALWAYS VISIBLE ON TOP
        ax.text(data['lon'], data['lat'], offset_z + 35,
               tower, fontsize=12, fontweight='bold', ha='center', zorder=10000,
               bbox=dict(boxstyle='round,pad=0.4', facecolor='white', 
                        edgecolor=data['color'], linewidth=2, alpha=0.98))
    
    # Draw connecting lines between layers (showing layer stacking)
    corner_lons = [lon.min(), lon.max(), lon.max(), lon.min()]
    corner_lats = [lat.min(), lat.min(), lat.max(), lat.max()]
    
    for clat, clon in zip(corner_lats, corner_lons):
        z_values = [elevation.min() + i * layer_offset for i in range(n_zones)]
        ax.plot([clon]*n_zones, [clat]*n_zones, z_values,
               'k--', alpha=0.2, linewidth=1)
    
    ax.set_xlabel('Longitude (°W)', fontsize=12, labelpad=10)
    ax.set_ylabel('Latitude (°N)', fontsize=12, labelpad=10)
    ax.set_zlabel('Elevation (m) - Stacked View', fontsize=12, labelpad=10)
    
    ax.set_title(
        f'Exploded/Stacked Elevation Zones View\n'
        f'{REGION_INFO["name"]}\n'
        f'Each zone layer is vertically offset for clarity',
        fontsize=14, fontweight='bold'
    )
    
    ax.view_init(elev=20, azim=230)
    
    # Create custom legend
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=zone_info[name]['color'], 
                            edgecolor='black', label=f"{name}: {zone_info[name]['range'][0]:.0f}-{zone_info[name]['range'][1]:.0f}m") 
                      for name in zone_names]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9, framealpha=0.9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    plt.close()


def plot_elevation_profile_zones(lon, lat, elevation, save_path=None):
    """
    Create elevation profiles showing zone classifications along transects.
    """
    LON, LAT = np.meshgrid(lon, lat)
    
    # Classify zones
    zones, zone_info = classify_elevation_zones(elevation, method='percentile')
    zone_names = list(zone_info.keys())
    zone_colors = [zone_info[name]['color'] for name in zone_names]
    
    fig, axes = plt.subplots(2, 2, figsize=(18, 14))
    
    # Profile 1: West to East through center
    ax1 = axes[0, 0]
    center_lat_idx = len(lat) // 2
    profile_elev = elevation[center_lat_idx, :]
    profile_zones = zones[center_lat_idx, :]
    
    # Fill zones with colors
    for i, name in enumerate(zone_names):
        zone_mask = profile_zones == i
        if np.any(zone_mask):
            ax1.fill_between(lon, elevation.min() - 10, profile_elev,
                           where=zone_mask, color=zone_colors[i], alpha=0.7,
                           label=name)
    
    ax1.plot(lon, profile_elev, 'k-', linewidth=2)
    
    # Mark zone boundaries
    for name in zone_names[:-1]:
        boundary = zone_info[name]['range'][1]
        ax1.axhline(y=boundary, color='white', linestyle='--', linewidth=1.5, alpha=0.8)
        ax1.text(lon.max(), boundary, f' {boundary:.0f}m', fontsize=8, va='center')
    
    ax1.set_xlabel('Longitude (°W)', fontsize=11)
    ax1.set_ylabel('Elevation (m)', fontsize=11)
    ax1.set_title(f'W-E Profile (Lat ≈ {lat[center_lat_idx]:.4f}°N)', fontsize=12, fontweight='bold')
    ax1.legend(loc='upper right', fontsize=8)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(elevation.min() - 10, elevation.max() + 20)
    
    # Profile 2: South to North through center
    ax2 = axes[0, 1]
    center_lon_idx = len(lon) // 2
    profile_elev = elevation[:, center_lon_idx]
    profile_zones = zones[:, center_lon_idx]
    
    for i, name in enumerate(zone_names):
        zone_mask = profile_zones == i
        if np.any(zone_mask):
            ax2.fill_between(lat, elevation.min() - 10, profile_elev,
                           where=zone_mask, color=zone_colors[i], alpha=0.7,
                           label=name)
    
    ax2.plot(lat, profile_elev, 'k-', linewidth=2)
    
    for name in zone_names[:-1]:
        boundary = zone_info[name]['range'][1]
        ax2.axhline(y=boundary, color='white', linestyle='--', linewidth=1.5, alpha=0.8)
    
    ax2.set_xlabel('Latitude (°N)', fontsize=11)
    ax2.set_ylabel('Elevation (m)', fontsize=11)
    ax2.set_title(f'S-N Profile (Lon ≈ {lon[center_lon_idx]:.4f}°W)', fontsize=12, fontweight='bold')
    ax2.legend(loc='upper right', fontsize=8)
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(elevation.min() - 10, elevation.max() + 20)
    
    # Zone area distribution
    ax3 = axes[1, 0]
    zone_areas = []
    for i in range(len(zone_names)):
        zone_areas.append(np.sum(zones == i) / zones.size * 100)
    
    bars = ax3.barh(zone_names, zone_areas, color=zone_colors, edgecolor='black', linewidth=2)
    ax3.set_xlabel('Area Coverage (%)', fontsize=11)
    ax3.set_title('Zone Area Distribution', fontsize=12, fontweight='bold')
    
    for bar, area in zip(bars, zone_areas):
        ax3.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{area:.1f}%', va='center', fontsize=10, fontweight='bold')
    
    ax3.set_xlim(0, max(zone_areas) * 1.2)
    ax3.grid(axis='x', alpha=0.3)
    
    # Tower positions by zone
    ax4 = axes[1, 1]
    
    tower_data = []
    for tower, data in TOWER_COORDINATES.items():
        lon_idx = np.argmin(np.abs(lon - data['lon']))
        lat_idx = np.argmin(np.abs(lat - data['lat']))
        tower_zone = zones[lat_idx, lon_idx]
        tower_elev = elevation[lat_idx, lon_idx]
        tower_data.append((tower, zone_names[tower_zone], tower_elev, data['color']))
    
    # Sort by elevation
    tower_data.sort(key=lambda x: x[2])
    
    y_positions = np.arange(len(tower_data))
    elevations = [t[2] for t in tower_data]
    colors = [t[3] for t in tower_data]
    
    bars = ax4.barh(y_positions, elevations, color=colors, edgecolor='black', linewidth=2)
    ax4.set_yticks(y_positions)
    ax4.set_yticklabels([f"{t[0]} ({t[1]})" for t in tower_data])
    ax4.set_xlabel('Elevation (m)', fontsize=11)
    ax4.set_title('Towers by Elevation Zone', fontsize=12, fontweight='bold')
    
    # Add zone boundary lines
    for name in zone_names[:-1]:
        boundary = zone_info[name]['range'][1]
        ax4.axvline(x=boundary, color='gray', linestyle='--', linewidth=1.5, alpha=0.5)
        ax4.text(boundary, len(tower_data) - 0.5, f'{boundary:.0f}m', 
                fontsize=8, ha='center', rotation=90)
    
    for bar, (tower, zone, elev, color) in zip(bars, tower_data):
        ax4.text(bar.get_width() + 2, bar.get_y() + bar.get_height()/2,
                f'{elev:.0f}m', va='center', fontsize=10, fontweight='bold')
    
    ax4.grid(axis='x', alpha=0.3)
    
    fig.suptitle(f'Elevation Zone Profiles - {REGION_INFO["name"]}\nBasin → Plain → Hill Classification',
                 fontsize=16, fontweight='bold', y=1.02)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=200, bbox_inches='tight')
        print(f"Saved: {save_path}")
    
    plt.show()
    plt.close()


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    print("=" * 60)
    print("   HIGH-RESOLUTION 3D TERRAIN VISUALIZATION")
    print("   ORNL Weather Tower Region")
    print("=" * 60)
    
    # Create output directory
    output_dir = 'viz_terrain_3d'
    os.makedirs(output_dir, exist_ok=True)
    
    # Get region bounds
    bounds = get_region_bounds(padding_km=0.8)
    print(f"\nRegion bounds:")
    print(f"  Latitude:  {bounds['min_lat']:.4f} to {bounds['max_lat']:.4f}")
    print(f"  Longitude: {bounds['min_lon']:.4f} to {bounds['max_lon']:.4f}")
    
    # Try to load real elevation data
    lon, lat, elevation = None, None, None
    
    # Option 1: Try to load existing GeoTIFF (if previously downloaded)
    for existing_tiff in ['elevation_data/srtm_ornl.tif', 
                          'elevation_data/opentopo_SRTMGL1.tif',
                          'elevation_data/opentopo_COP30.tif']:
        if os.path.exists(existing_tiff) and HAS_RASTERIO:
            print(f"\n[Option 1] Loading existing GeoTIFF: {existing_tiff}")
            lon, lat, elevation = load_geotiff_elevation(existing_tiff)
            if elevation is not None:
                break
    
    # Option 2: Try OpenTopography API (if API key provided)
    if elevation is None:
        api_key = os.environ.get('OPENTOPO_API_KEY')
        if api_key and HAS_REQUESTS:
            print("\n[Option 2] Trying OpenTopography API...")
            tiff_path = fetch_opentopography_dem(bounds, api_key, dem_type='COP30')
            if tiff_path and HAS_RASTERIO:
                lon, lat, elevation = load_geotiff_elevation(tiff_path)
    
    # Option 3: Try Open-Elevation API (FREE, no registration!)
    if elevation is None and HAS_REQUESTS:
        print("\n[Option 3] Using Open-Elevation API (FREE, no registration)...")
        lon, lat, elevation = fetch_open_elevation_grid(bounds, resolution=60)
    
    # Option 4: Use synthetic terrain as fallback
    if elevation is None:
        print("\n[Option 4] Using synthetic terrain generation (fallback)...")
        lon, lat, elevation = generate_synthetic_terrain(bounds, resolution=150)
    
    print(f"\nTerrain data shape: {elevation.shape}")
    print(f"Elevation range: {elevation.min():.1f}m to {elevation.max():.1f}m")
    
    # Generate all visualizations
    print("\n" + "-" * 60)
    print("Generating visualizations...")
    print("-" * 60)
    
    # 1. Basic 3D terrain
    print("\n[1/8] Basic 3D terrain plot...")
    plot_3d_terrain_basic(
        lon, lat, elevation,
        save_path=os.path.join(output_dir, '3d_terrain_basic.png')
    )
    
    # 2. Detailed multi-view
    print("\n[2/8] Multi-view 3D terrain...")
    plot_3d_terrain_detailed(
        lon, lat, elevation,
        save_path=os.path.join(output_dir, '3d_terrain_multiview.png')
    )
    
    # 3. With contours
    print("\n[3/8] 3D terrain with contours...")
    plot_3d_terrain_with_contours(
        lon, lat, elevation,
        save_path=os.path.join(output_dir, '3d_terrain_contours.png')
    )
    
    # 4. 2D hillshade map
    print("\n[4/8] 2D hillshade map...")
    plot_2d_hillshade_map(
        lon, lat, elevation,
        save_path=os.path.join(output_dir, '2d_hillshade_map.png')
    )
    
    # 5. Cross-sections
    print("\n[5/8] Terrain cross-sections...")
    plot_terrain_cross_sections(
        lon, lat, elevation,
        save_path=os.path.join(output_dir, 'terrain_cross_sections.png')
    )
    
    # 6. Tower characteristics summary
    print("\n[6/8] Tower characteristics summary...")
    plot_tower_characteristics(
        save_path=os.path.join(output_dir, 'tower_characteristics.png')
    )
    
    # 7. Satellite image with towers
    print("\n[7/8] Satellite image with tower locations...")
    try:
        plot_satellite_with_towers(
            bounds,
            save_path=os.path.join(output_dir, 'satellite_towers.png')
        )
    except Exception as e:
        print(f"  Satellite plot skipped: {e}")
    
    # 8. Combined satellite + terrain
    print("\n[8/11] Combined satellite and terrain view...")
    try:
        plot_terrain_with_satellite_overlay(
            lon, lat, elevation, bounds,
            save_path=os.path.join(output_dir, 'satellite_terrain_combined.png')
        )
    except Exception as e:
        print(f"  Combined plot skipped: {e}")
    
    # 9. Elevation zones 2D map (Basin, Plain, Hill)
    print("\n[9/11] Elevation zones 2D map (Basin/Plain/Hill)...")
    plot_elevation_zones_2d(
        lon, lat, elevation, method='percentile',
        save_path=os.path.join(output_dir, 'elevation_zones_2d.png')
    )
    
    # 10. Elevation zones 3D visualization
    print("\n[10/11] Elevation zones 3D with reference planes...")
    plot_elevation_zones_3d(
        lon, lat, elevation, method='percentile',
        save_path=os.path.join(output_dir, 'elevation_zones_3d.png')
    )
    
    # 11. Elevation zone profiles
    print("\n[11/11] Elevation zone profiles...")
    plot_elevation_profile_zones(
        lon, lat, elevation,
        save_path=os.path.join(output_dir, 'elevation_zone_profiles.png')
    )
    
    # Bonus: Stacked/exploded zone view
    print("\n[Bonus 1] Stacked elevation zones view...")
    plot_elevation_zones_stacked(
        lon, lat, elevation,
        save_path=os.path.join(output_dir, 'elevation_zones_stacked.png')
    )
    
    # Bonus: Interactive plot
    print("\n[Bonus 2] Attempting interactive HTML plot...")
    create_interactive_html_plot(
        lon, lat, elevation,
        output_path=os.path.join(output_dir, 'terrain_interactive.html')
    )
    
    print("\n" + "=" * 60)
    print("   VISUALIZATION COMPLETE!")
    print(f"   Output saved to: {output_dir}/")
    print("=" * 60)
    
    # List generated files
    print("\nGenerated files:")
    for f in sorted(os.listdir(output_dir)):
        fpath = os.path.join(output_dir, f)
        size_kb = os.path.getsize(fpath) / 1024
        print(f"  • {f} ({size_kb:.1f} KB)")


if __name__ == '__main__':
    main()

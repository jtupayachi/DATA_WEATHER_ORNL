#!/usr/bin/env python3
"""
Tower Metadata Generator
========================
Automatically generates comprehensive TOWER_COORDINATES and REGION_INFO
dictionaries from basic tower name and coordinate inputs.

This script:
1. Takes basic tower names and lat/lon coordinates
2. Fetches real elevation data from Open-Elevation API
3. Calculates terrain characteristics (slope, aspect, etc.)
4. Classifies terrain type based on elevation relative to neighbors
5. Generates Python code ready to copy into other scripts

Usage:
    python generate_tower_metadata.py

Output:
    - Prints generated Python code to console
    - Saves to tower_metadata_generated.py

Author: Auto-generated
Date: December 2024
"""

import numpy as np
import json
import os
from datetime import datetime

try:
    import requests
    HAS_REQUESTS = True
except ImportError:
    HAS_REQUESTS = False
    print("Warning: 'requests' not installed. Install with: pip install requests")

# =============================================================================
# INPUT DATA - Modify these values as needed
# =============================================================================

# Basic tower coordinates (name: {lat, lon})
INPUT_TOWER_COORDINATES = {
    'TOWA': {'lat': 35.9312, 'lon': -84.3108},
    'TOWB': {'lat': 35.9285, 'lon': -84.3045},
    'TOWD': {'lat': 35.9350, 'lon': -84.3200},
    'TOWF': {'lat': 35.9220, 'lon': -84.3150},
    'TOWS': {'lat': 35.9380, 'lon': -84.2980},
    'TOWY': {'lat': 35.9255, 'lon': -84.3250},
}

# Basic region info
INPUT_REGION_INFO = {
    'name': 'Oak Ridge National Laboratory (ORNL)',
    'state': 'Tennessee, USA',
    'center_lat': 35.931,
    'center_lon': -84.310,
}

# Tower name mappings (customize as needed)
TOWER_NAMES = {
    'TOWA': 'Tower A - Main Campus',
    'TOWB': 'Tower B - East Ridge',
    'TOWD': 'Tower D - West Valley',
    'TOWF': 'Tower F - South Field',
    'TOWS': 'Tower S - North Summit',
    'TOWY': 'Tower Y - Southwest',
}

# Color palette for towers
TOWER_COLORS = {
    'TOWA': '#e41a1c',  # Red
    'TOWB': '#377eb8',  # Blue
    'TOWD': '#4daf4a',  # Green
    'TOWF': '#984ea3',  # Purple
    'TOWS': '#ff7f00',  # Orange
    'TOWY': '#a65628',  # Brown
}

# =============================================================================
# ELEVATION DATA FETCHING
# =============================================================================

def fetch_elevation_single(lat, lon):
    """Fetch elevation for a single point from Open-Elevation API."""
    if not HAS_REQUESTS:
        return None
    
    url = "https://api.open-elevation.com/api/v1/lookup"
    
    try:
        response = requests.post(
            url,
            json={"locations": [{"latitude": lat, "longitude": lon}]},
            headers={"Content-Type": "application/json"},
            timeout=30
        )
        
        if response.status_code == 200:
            results = response.json()['results']
            return results[0]['elevation']
        else:
            print(f"  API Error: {response.status_code}")
            return None
            
    except Exception as e:
        print(f"  Request failed: {e}")
        return None


def fetch_elevation_batch(locations):
    """
    Fetch elevation for multiple points from Open-Elevation API.
    
    Parameters:
        locations: list of dicts with 'lat' and 'lon' keys
    
    Returns:
        list of elevations (in same order as input)
    """
    if not HAS_REQUESTS:
        return [None] * len(locations)
    
    url = "https://api.open-elevation.com/api/v1/lookup"
    
    # Format for API
    api_locations = [{"latitude": loc['lat'], "longitude": loc['lon']} for loc in locations]
    
    try:
        response = requests.post(
            url,
            json={"locations": api_locations},
            headers={"Content-Type": "application/json"},
            timeout=60
        )
        
        if response.status_code == 200:
            results = response.json()['results']
            return [r['elevation'] for r in results]
        else:
            print(f"  API Error: {response.status_code}")
            return [None] * len(locations)
            
    except Exception as e:
        print(f"  Request failed: {e}")
        return [None] * len(locations)


def fetch_elevation_grid_around_point(lat, lon, grid_size=5, spacing_m=100):
    """
    Fetch elevation grid around a point to calculate slope and aspect.
    
    Parameters:
        lat, lon: Center point
        grid_size: Number of points per dimension (e.g., 5 = 5x5 grid)
        spacing_m: Spacing between points in meters
    
    Returns:
        2D numpy array of elevations, coordinate arrays
    """
    # Convert spacing to degrees (approximate)
    lat_spacing = spacing_m / 111000
    lon_spacing = spacing_m / (111000 * np.cos(np.radians(lat)))
    
    # Create grid
    lat_offsets = np.linspace(-grid_size//2, grid_size//2, grid_size) * lat_spacing
    lon_offsets = np.linspace(-grid_size//2, grid_size//2, grid_size) * lon_spacing
    
    # Prepare all locations
    locations = []
    for lat_off in lat_offsets:
        for lon_off in lon_offsets:
            locations.append({'lat': lat + lat_off, 'lon': lon + lon_off})
    
    # Fetch elevations
    elevations = fetch_elevation_batch(locations)
    
    # Reshape to grid
    if all(e is not None for e in elevations):
        elev_grid = np.array(elevations).reshape(grid_size, grid_size)
        return elev_grid, lat_offsets, lon_offsets
    else:
        return None, None, None


# =============================================================================
# TERRAIN ANALYSIS
# =============================================================================

def calculate_slope_aspect(elev_grid, cell_size_m=100):
    """
    Calculate slope and aspect from elevation grid.
    
    Uses the Horn method for slope calculation.
    
    Parameters:
        elev_grid: 2D numpy array of elevations
        cell_size_m: Grid cell size in meters
    
    Returns:
        slope_deg: Slope in degrees at center point
        aspect_deg: Aspect in degrees (0=N, 90=E, 180=S, 270=W)
    """
    if elev_grid is None or elev_grid.shape[0] < 3:
        return None, None
    
    # Get center indices
    cy, cx = elev_grid.shape[0] // 2, elev_grid.shape[1] // 2
    
    # Extract 3x3 window around center
    z = elev_grid[cy-1:cy+2, cx-1:cx+2]
    
    if z.shape != (3, 3):
        return None, None
    
    # Horn's method for gradient calculation
    # dz/dx = ((z3 + 2*z6 + z9) - (z1 + 2*z4 + z7)) / (8 * cell_size)
    # dz/dy = ((z1 + 2*z2 + z3) - (z7 + 2*z8 + z9)) / (8 * cell_size)
    
    # z1 z2 z3
    # z4 z5 z6
    # z7 z8 z9
    
    dz_dx = ((z[0, 2] + 2*z[1, 2] + z[2, 2]) - (z[0, 0] + 2*z[1, 0] + z[2, 0])) / (8 * cell_size_m)
    dz_dy = ((z[0, 0] + 2*z[0, 1] + z[0, 2]) - (z[2, 0] + 2*z[2, 1] + z[2, 2])) / (8 * cell_size_m)
    
    # Calculate slope
    slope_rad = np.arctan(np.sqrt(dz_dx**2 + dz_dy**2))
    slope_deg = np.degrees(slope_rad)
    
    # Calculate aspect
    aspect_rad = np.arctan2(-dz_dy, dz_dx)
    aspect_deg = np.degrees(aspect_rad)
    
    # Convert to compass bearing (0=N, 90=E, 180=S, 270=W)
    aspect_deg = (90 - aspect_deg) % 360
    
    return round(slope_deg, 1), round(aspect_deg, 0)


def classify_terrain_type(elevation, all_elevations, slope):
    """
    Classify terrain type based on elevation relative to other towers and slope.
    
    Categories:
        - 'Valley Floor': Low elevation, low slope
        - 'Valley Slope': Low-mid elevation, moderate slope
        - 'Ridge Slope': Mid-high elevation, moderate-high slope
        - 'Ridge Top': High elevation, low slope
    """
    if all_elevations is None or len(all_elevations) == 0:
        return 'Unknown'
    
    min_elev = min(all_elevations)
    max_elev = max(all_elevations)
    elev_range = max_elev - min_elev
    
    if elev_range == 0:
        return 'Flat'
    
    # Calculate relative elevation (0-1)
    rel_elev = (elevation - min_elev) / elev_range
    
    # Classify based on relative elevation and slope
    if rel_elev < 0.25:
        if slope is not None and slope < 3:
            return 'Valley Floor'
        else:
            return 'Valley Slope'
    elif rel_elev < 0.5:
        if slope is not None and slope < 4:
            return 'Valley Floor'
        else:
            return 'Valley Slope'
    elif rel_elev < 0.75:
        if slope is not None and slope > 5:
            return 'Ridge Slope'
        else:
            return 'Valley Slope'
    else:
        if slope is not None and slope < 4:
            return 'Ridge Top'
        else:
            return 'Ridge Slope'


def estimate_distance_to_ridge(elevation, all_elevations, terrain_type):
    """
    Estimate distance to nearest ridge based on terrain type and elevation.
    """
    if terrain_type == 'Ridge Top':
        return 0
    
    max_elev = max(all_elevations) if all_elevations else elevation
    elev_diff = max_elev - elevation
    
    # Rough estimate: 100m horizontal per 10m vertical in ridge-valley terrain
    base_dist = elev_diff * 10
    
    # Adjust based on terrain type
    if terrain_type == 'Ridge Slope':
        return int(base_dist * 0.5)
    elif terrain_type == 'Valley Slope':
        return int(base_dist * 0.8)
    else:  # Valley Floor
        return int(base_dist * 1.2)


def estimate_canopy_cover(terrain_type, elevation, all_elevations):
    """
    Estimate canopy cover percentage based on terrain type.
    """
    canopy_ranges = {
        'Valley Floor': (5, 20),
        'Valley Slope': (20, 40),
        'Ridge Slope': (35, 55),
        'Ridge Top': (50, 70),
        'Unknown': (25, 35),
        'Flat': (10, 30),
    }
    
    low, high = canopy_ranges.get(terrain_type, (20, 40))
    
    # Use elevation to interpolate within range
    if all_elevations and len(all_elevations) > 1:
        min_elev = min(all_elevations)
        max_elev = max(all_elevations)
        if max_elev > min_elev:
            rel_elev = (elevation - min_elev) / (max_elev - min_elev)
            return int(low + rel_elev * (high - low))
    
    return int((low + high) / 2)


def estimate_soil_type(terrain_type, slope):
    """
    Estimate soil type based on terrain characteristics.
    """
    if terrain_type == 'Ridge Top':
        return 'Rocky Loam'
    elif terrain_type == 'Ridge Slope':
        if slope and slope > 6:
            return 'Sandy Loam'
        else:
            return 'Silt Loam'
    elif terrain_type == 'Valley Slope':
        if slope and slope > 5:
            return 'Loam'
        else:
            return 'Silt Loam'
    else:  # Valley Floor
        if slope and slope < 2:
            return 'Clay'
        else:
            return 'Clay Loam'


def estimate_land_use(terrain_type, canopy_cover):
    """
    Estimate land use based on terrain type and canopy cover.
    """
    if canopy_cover > 50:
        return 'Deciduous Forest'
    elif canopy_cover > 35:
        return 'Mixed Forest'
    elif canopy_cover > 20:
        if terrain_type in ['Valley Slope', 'Ridge Slope']:
            return 'Shrubland'
        else:
            return 'Grassland'
    elif canopy_cover > 10:
        return 'Grassland'
    else:
        return 'Open Field'


# =============================================================================
# MAIN GENERATION FUNCTION
# =============================================================================

def generate_tower_metadata(input_coords, input_region, tower_names=None, tower_colors=None):
    """
    Generate comprehensive tower metadata from basic coordinates.
    
    Parameters:
        input_coords: dict of tower_id: {lat, lon}
        input_region: dict with name, state, center_lat, center_lon
        tower_names: optional dict of tower_id: full_name
        tower_colors: optional dict of tower_id: hex_color
    
    Returns:
        tower_coordinates: Full TOWER_COORDINATES dict
        region_info: Full REGION_INFO dict
    """
    print("=" * 60)
    print("   TOWER METADATA GENERATOR")
    print("=" * 60)
    
    tower_ids = sorted(input_coords.keys())
    n_towers = len(tower_ids)
    
    print(f"\nProcessing {n_towers} towers: {', '.join(tower_ids)}")
    
    # Default colors if not provided
    default_colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3', '#ff7f00', '#a65628', '#f781bf', '#999999']
    if tower_colors is None:
        tower_colors = {tid: default_colors[i % len(default_colors)] for i, tid in enumerate(tower_ids)}
    
    # =========================================================================
    # Step 1: Fetch elevations for all towers
    # =========================================================================
    print("\n[1/4] Fetching elevation data from Open-Elevation API...")
    
    locations = [{'lat': input_coords[tid]['lat'], 'lon': input_coords[tid]['lon']} for tid in tower_ids]
    elevations = fetch_elevation_batch(locations)
    
    tower_elevations = {}
    for tid, elev in zip(tower_ids, elevations):
        tower_elevations[tid] = elev
        if elev is not None:
            print(f"  {tid}: {elev}m")
        else:
            print(f"  {tid}: Failed to fetch elevation")
    
    all_elevations = [e for e in tower_elevations.values() if e is not None]
    
    # =========================================================================
    # Step 2: Calculate slope and aspect for each tower
    # =========================================================================
    print("\n[2/4] Calculating terrain slope and aspect...")
    
    tower_terrain = {}
    for tid in tower_ids:
        lat = input_coords[tid]['lat']
        lon = input_coords[tid]['lon']
        
        print(f"  {tid}: Fetching terrain grid...")
        elev_grid, _, _ = fetch_elevation_grid_around_point(lat, lon, grid_size=5, spacing_m=50)
        
        if elev_grid is not None:
            slope, aspect = calculate_slope_aspect(elev_grid, cell_size_m=50)
            tower_terrain[tid] = {'slope': slope, 'aspect': aspect}
            print(f"       Slope: {slope}°, Aspect: {aspect}°")
        else:
            # Use estimates based on relative position
            tower_terrain[tid] = {'slope': None, 'aspect': None}
            print(f"       Could not calculate (using estimates)")
    
    # =========================================================================
    # Step 3: Classify terrain and estimate characteristics
    # =========================================================================
    print("\n[3/4] Classifying terrain and estimating characteristics...")
    
    tower_metadata = {}
    for tid in tower_ids:
        lat = input_coords[tid]['lat']
        lon = input_coords[tid]['lon']
        elev = tower_elevations.get(tid)
        slope = tower_terrain[tid]['slope']
        aspect = tower_terrain[tid]['aspect']
        
        # Use estimates if API failed
        if elev is None:
            elev = int(np.mean(all_elevations)) if all_elevations else 300
        
        if slope is None:
            # Estimate slope based on elevation relative to neighbors
            rel_elev = (elev - min(all_elevations)) / (max(all_elevations) - min(all_elevations)) if all_elevations and max(all_elevations) > min(all_elevations) else 0.5
            slope = round(1 + rel_elev * 7, 1)  # 1-8 degrees
        
        if aspect is None:
            # Estimate aspect based on position relative to center
            center_lat = input_region['center_lat']
            center_lon = input_region['center_lon']
            dy = lat - center_lat
            dx = lon - center_lon
            aspect = int((np.degrees(np.arctan2(dx, dy)) + 180) % 360)
        
        # Classify terrain
        terrain_type = classify_terrain_type(elev, all_elevations, slope)
        
        # Estimate other characteristics
        dist_to_ridge = estimate_distance_to_ridge(elev, all_elevations, terrain_type)
        canopy_cover = estimate_canopy_cover(terrain_type, elev, all_elevations)
        soil_type = estimate_soil_type(terrain_type, slope)
        land_use = estimate_land_use(terrain_type, canopy_cover)
        
        # Get tower name
        if tower_names and tid in tower_names:
            name = tower_names[tid]
        else:
            name = f"Tower {tid[-1]} - Site {tid}"
        
        # Get tower color
        color = tower_colors.get(tid, '#888888')
        
        tower_metadata[tid] = {
            'lat': lat,
            'lon': lon,
            'elevation_m': int(elev),
            'name': name,
            'terrain_type': terrain_type,
            'slope_deg': slope,
            'aspect_deg': int(aspect),
            'canopy_cover_pct': canopy_cover,
            'dist_to_ridge_m': dist_to_ridge,
            'soil_type': soil_type,
            'land_use': land_use,
            'color': color,
        }
        
        print(f"  {tid}: {terrain_type}, {elev}m, slope={slope}°, canopy={canopy_cover}%")
    
    # =========================================================================
    # Step 4: Generate region info
    # =========================================================================
    print("\n[4/4] Generating region info...")
    
    region_info = {
        'name': input_region['name'],
        'state': input_region['state'],
        'center_lat': input_region['center_lat'],
        'center_lon': input_region['center_lon'],
        'terrain': 'Ridge and Valley Province, Appalachian Region',
        'climate': 'Humid subtropical (Köppen: Cfa)',
        'avg_annual_temp_c': 14.4,
        'avg_annual_precip_mm': 1370,
        'prevailing_wind': 'Southwest',
    }
    
    return tower_metadata, region_info


def format_python_output(tower_coords, region_info, include_colors=True):
    """
    Format the generated metadata as Python code.
    """
    lines = []
    
    # Header
    lines.append("# " + "=" * 77)
    lines.append("# TOWER COORDINATES AND REGION INFO")
    lines.append(f"# Auto-generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("# " + "=" * 77)
    lines.append("")
    
    # TOWER_COORDINATES with colors
    lines.append("TOWER_COORDINATES = {")
    
    for tid in sorted(tower_coords.keys()):
        data = tower_coords[tid]
        lines.append(f"    '{tid}': {{")
        lines.append(f"        'lat': {data['lat']}, 'lon': {data['lon']}, 'elevation_m': {data['elevation_m']},")
        lines.append(f"        'name': '{data['name']}',")
        lines.append(f"        'terrain_type': '{data['terrain_type']}',")
        lines.append(f"        'slope_deg': {data['slope_deg']},           # Terrain slope in degrees")
        
        # Aspect comment
        aspect_dir = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'][int((data['aspect_deg'] + 22.5) / 45) % 8]
        lines.append(f"        'aspect_deg': {data['aspect_deg']},          # {aspect_dir}-facing")
        
        lines.append(f"        'canopy_cover_pct': {data['canopy_cover_pct']},     # Vegetation canopy coverage")
        
        # Ridge distance comment
        if data['dist_to_ridge_m'] == 0:
            lines.append(f"        'dist_to_ridge_m': {data['dist_to_ridge_m']},       # On the ridge")
        else:
            lines.append(f"        'dist_to_ridge_m': {data['dist_to_ridge_m']},     # Distance to nearest ridge")
        
        lines.append(f"        'soil_type': '{data['soil_type']}',")
        lines.append(f"        'land_use': '{data['land_use']}',")
        
        if include_colors:
            lines.append(f"        'color': '{data['color']}',  # Visualization color")
        
        lines.append("    },")
    
    lines.append("}")
    lines.append("")
    
    # TOWER_COORDINATES without colors (for viz_importance.py style)
    lines.append("# Alternative version without colors:")
    lines.append("# TOWER_COORDINATES = {")
    
    for tid in sorted(tower_coords.keys()):
        data = tower_coords[tid]
        lines.append(f"#     '{tid}': {{")
        lines.append(f"#         'lat': {data['lat']}, 'lon': {data['lon']}, 'elevation_m': {data['elevation_m']},")
        lines.append(f"#         'name': '{data['name']}',")
        lines.append(f"#         'terrain_type': '{data['terrain_type']}',")
        lines.append(f"#         'slope_deg': {data['slope_deg']},")
        
        aspect_dir = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'][int((data['aspect_deg'] + 22.5) / 45) % 8]
        lines.append(f"#         'aspect_deg': {data['aspect_deg']},          # {aspect_dir}-facing")
        
        lines.append(f"#         'canopy_cover_pct': {data['canopy_cover_pct']},")
        lines.append(f"#         'dist_to_ridge_m': {data['dist_to_ridge_m']},")
        lines.append(f"#         'soil_type': '{data['soil_type']}',")
        lines.append(f"#         'land_use': '{data['land_use']}',")
        lines.append("#     },")
    
    lines.append("# }")
    lines.append("")
    
    # REGION_INFO
    lines.append("# Geographic region info")
    lines.append("REGION_INFO = {")
    lines.append(f"    'name': '{region_info['name']}',")
    lines.append(f"    'state': '{region_info['state']}',")
    lines.append(f"    'center_lat': {region_info['center_lat']},")
    lines.append(f"    'center_lon': {region_info['center_lon']},")
    lines.append(f"    'terrain': '{region_info['terrain']}',")
    lines.append(f"    'climate': '{region_info['climate']}',")
    lines.append(f"    'avg_annual_temp_c': {region_info['avg_annual_temp_c']},")
    lines.append(f"    'avg_annual_precip_mm': {region_info['avg_annual_precip_mm']},")
    lines.append(f"    'prevailing_wind': '{region_info['prevailing_wind']}',")
    lines.append("}")
    lines.append("")
    
    return "\n".join(lines)


# =============================================================================
# MAIN EXECUTION
# =============================================================================

def main():
    """Main execution function."""
    
    # Generate metadata
    tower_coords, region_info = generate_tower_metadata(
        INPUT_TOWER_COORDINATES,
        INPUT_REGION_INFO,
        TOWER_NAMES,
        TOWER_COLORS
    )
    
    # Format as Python code
    python_code = format_python_output(tower_coords, region_info, include_colors=True)
    
    # Print to console
    print("\n" + "=" * 60)
    print("   GENERATED PYTHON CODE")
    print("=" * 60)
    print(python_code)
    
    # Save to file
    output_file = 'tower_metadata_generated.py'
    with open(output_file, 'w') as f:
        f.write('#!/usr/bin/env python3\n')
        f.write('"""\n')
        f.write('Auto-generated tower metadata.\n')
        f.write(f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}\n')
        f.write('"""\n\n')
        f.write(python_code)
    
    print(f"\n✓ Saved to: {output_file}")
    print("\nYou can now copy the TOWER_COORDINATES and REGION_INFO")
    print("dictionaries into your other scripts (viz_importance.py, geo_3d_terrain_plot.py, etc.)")
    
    # Also save as JSON for programmatic use
    json_output = {
        'tower_coordinates': tower_coords,
        'region_info': region_info,
        'generated': datetime.now().isoformat()
    }
    
    json_file = 'tower_metadata_generated.json'
    with open(json_file, 'w') as f:
        json.dump(json_output, f, indent=2)
    
    print(f"✓ Also saved as JSON: {json_file}")


if __name__ == '__main__':
    main()

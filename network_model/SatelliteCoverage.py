import math
import numpy as np


import matplotlib.pyplot as plt
import cartopy.crs as ccrs
import cartopy.feature as cfeature

from shapely.geometry import Polygon
from shapely.ops import transform, unary_union
import pyproj


EARTH_RADIUS_KM = 6371.0


def generate_circle(lat, lon, radius_km, num_points=100):
    # Generate points around the circle
    angles = np.linspace(0, 2 * np.pi, num_points)
    lats = lat + (radius_km / EARTH_RADIUS_KM) * (180 / np.pi) * np.sin(angles)
    lons = lon + (radius_km / EARTH_RADIUS_KM) * (180 / np.pi) * np.cos(angles) / np.cos(np.radians(lat))
    return Polygon(zip(lons, lats))

def convert_area_to_km2(geometry, lat, lon):
    proj_wgs84 = pyproj.Proj(proj="latlong", datum="WGS84")
    proj_aeqd = pyproj.Proj(proj='aeqd', lat_0=lat, lon_0=lon)

    project = lambda x, y: pyproj.transform(proj_wgs84, proj_aeqd, x, y)
    projected_geom = transform(project, geometry)

    area_m2 = projected_geom.area
    area_km2 = area_m2 / 1e6

    return area_km2

def estimate_land_ocean_area(footprint_radius_km, lat, lon):
    # Create a circular footprint centered at (lon, lat) with the given radius
    footprint = generate_circle(lat, lon, footprint_radius_km)
    
    # Load Natural Earth land geometries
    land_geom = list(cfeature.NaturalEarthFeature('physical', 'land', '110m').geometries())

    # Calculate the union of all land geometries (to handle overlapping geometries)
    land_union = unary_union(land_geom)
    
    # Calculate the intersection of the footprint with the land
    land_intersection = footprint.intersection(land_union)
    
    # Estimate the land area within the footprint
    land_area_km2 = convert_area_to_km2(land_intersection, lat, lon)
    footprint_area_km2 = convert_area_to_km2(footprint, lat, lon)

    # Calculate the ocean area as the remainder
    ocean_area_km2 = max(footprint_area_km2 - land_area_km2, 0)

    return land_area_km2, ocean_area_km2, footprint_area_km2

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import numpy as np
    from shapely.geometry import Polygon

    # Define constants
    EARTH_RADIUS_KM = 6371.0  # Radius of the Earth in kilometers
    coverage_radius_km = 2000  # Coverage radius of the satellite in kilometers
    lat, lon = 50, 50  # Latitude and longitude of the satellite's position
    print(estimate_land_ocean_area(coverage_radius_km, lat, lon))
    # Set up the map using Cartopy with the PlateCarree projection
    fig, ax = plt.subplots(figsize=(10, 6), subplot_kw={'projection': ccrs.PlateCarree()})

    # Add basic map features (optional, for better visualization)
    ax.coastlines()
    ax.gridlines(draw_labels=True)

    # Load and plot the land geometries
    land_geoms = cfeature.NaturalEarthFeature('physical', 'land', '110m').geometries()

    # Plot each land geometry
    for geom in land_geoms:
        ax.add_geometries([geom], ccrs.PlateCarree(), edgecolor='black', facecolor='lightgreen')

    # Create and plot the satellite coverage circle
    footprint = generate_circle(lat, lon, coverage_radius_km)
    ax.add_geometries([footprint], ccrs.PlateCarree(), edgecolor='blue', facecolor='none', linewidth=2)

    # Set the title and display the plot
    ax.set_title(f'Satellite Coverage Circle with {coverage_radius_km} km Radius')
    plt.show()
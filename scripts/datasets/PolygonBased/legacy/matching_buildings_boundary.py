import warnings
warnings.filterwarnings('ignore', message="The expected order of coordinates in `bbox` will change in the v2.0.0 release")

import json
import os
import numpy as np
import osmnx as ox
import pyproj
import matplotlib.pyplot as plt

from shapely.ops import unary_union, polygonize, transform
from shapely.geometry import Polygon, MultiPolygon
from tqdm import tqdm

from scripts.token import get_mapbox_token

# Mapbox Access Token
radius = 0.01
is_visualize = False

def get_geojson_files(folder_path):
    all_files = os.listdir(folder_path)
    geojson_files = [os.path.join(folder_path, file) for file in all_files if file.endswith('.geojson')]

    return geojson_files

def get_boundaries(graph):
    edges = ox.graph_to_gdfs(graph, nodes=False, edges=True)
    lines = [line for line in edges.geometry]
    merged_lines = unary_union(lines)
    polygons = list(polygonize(merged_lines))

    return polygons

def plot_boundary_building(building_polygon, boundary_polygon):
    if not is_visualize:
        return

    building_x, building_y = building_polygon.exterior.xy
    boundary_x, boundary_y = boundary_polygon.exterior.xy

    plt.figure(figsize=(8, 8))
    plt.plot(boundary_x, boundary_y, color='blue', label='Boundary Polygon')
    plt.plot(building_x, building_y, color='red', label='Building Polygon')

    plt.fill(boundary_x, boundary_y, color='blue', alpha=0.3)
    plt.fill(building_x, building_y, color='red', alpha=0.5)

    plt.legend()
    plt.title('Building and Boundary Polygon Visualization')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)
    plt.show()

folder_path = 'C:/Users/ttd85/Downloads/US_Building_Footprints'
geojson_files = get_geojson_files(folder_path)[1:]

for geojson_file in geojson_files:
    with open(geojson_file, 'r') as f:
        geojson_data = json.load(f)

    save_boundary_polygons = []
    save_building_index = []

    for building_idx, feature in enumerate(tqdm(geojson_data['features'])):
        print(save_building_index)
        building_coordinate = geojson_data['features'][building_idx]['geometry']['coordinates'][0]
        building_polygon = Polygon(building_coordinate)

        is_intersecting = False
        for boundary_idx, boundary_polygon in enumerate(save_boundary_polygons):
            if boundary_polygon.intersects(building_polygon):
                is_intersecting = True
                break

        if is_intersecting:
            save_building_index[boundary_idx].append(building_idx)
            plot_boundary_building(building_polygon, boundary_polygon)
            continue

        building_center_coordinate = np.mean(building_coordinate, axis=0)

        east, west = building_center_coordinate[0] + radius, building_center_coordinate[0] - radius
        north, south = building_center_coordinate[1] + radius, building_center_coordinate[1] - radius

        bbox = (west, south, east, north)
        try:
            G = ox.graph_from_bbox(bbox=bbox, network_type='all')
        except:
            continue

        boundary_polygons = get_boundaries(G)

        is_intersecting = False
        for boundary_polygon in boundary_polygons:
            if boundary_polygon.intersects(building_polygon):
                is_intersecting = True
                break

        if is_intersecting:
            save_boundary_polygons.append(boundary_polygon)
            save_building_index.append([building_idx])
            plot_boundary_building(building_polygon, boundary_polygon)
            continue


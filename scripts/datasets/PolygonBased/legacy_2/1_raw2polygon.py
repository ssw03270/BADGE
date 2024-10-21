import matplotlib.pyplot as plt
from PIL import Image, ImageDraw
from shapely.geometry import Polygon
import matplotlib.patches as patches
from pyproj import Transformer
import numpy as np
import os
import re
import json
import pickle
from tqdm import tqdm

transformer = Transformer.from_crs("EPSG:4326", "EPSG:3857", always_xy=True)
# 좌표 변환 함수 (위도, 경도를 평면 좌표계로 변환)
def transform_coordinates(coords):
    return [transformer.transform(lon, lat) for lon, lat in coords]

def get_geojson_files(folder_path):
    all_files = os.listdir(folder_path)
    geojson_files = [os.path.join(folder_path, file) for file in all_files if file.endswith('.geojson')]
    return geojson_files

def extract_number_from_filename(filename):
    match = re.search(r'\d+', filename)
    return match.group() if match else None

def get_matching_files(boundary_files, building_files):
    boundary_numbers = {extract_number_from_filename(os.path.basename(f)) for f in boundary_files}
    building_numbers = {extract_number_from_filename(os.path.basename(f)) for f in building_files}

    matching_numbers = boundary_numbers.intersection(building_numbers)

    matching_boundary_files = [f for f in boundary_files if extract_number_from_filename(os.path.basename(f)) in matching_numbers]
    matching_building_files = [f for f in building_files if extract_number_from_filename(os.path.basename(f)) in matching_numbers]

    return matching_boundary_files, matching_building_files

# Helper function to plot a polygon
def plot_polygon(ax, polygon, edge_color='blue', face_color='none'):
    patch = patches.Polygon(list(polygon.exterior.coords), closed=True, edgecolor=edge_color, facecolor=face_color)
    ax.add_patch(patch)

# Function to calculate the centroid of a polygon
def calculate_centroid(polygon):
    return polygon.centroid

# Function to calculate the angle to rotate the longest side of the minimum rotated bounding box to be horizontal
def calculate_rotation_angle(polygon):
    min_rot_rect = polygon.minimum_rotated_rectangle
    exterior_coords = list(min_rot_rect.exterior.coords)

    # Find the longest side
    max_length = 0
    angle = 0
    for i in range(len(exterior_coords) - 1):
        p1 = exterior_coords[i]
        p2 = exterior_coords[i + 1]
        dx = p2[0] - p1[0]
        dy = p2[1] - p1[1]
        length = np.sqrt(dx**2 + dy**2)
        if length > max_length:
            max_length = length
            angle = np.arctan2(dy, dx)  # Angle in radians

    return -angle  # Negative angle to rotate to horizontal

# Function to rotate a polygon
def rotate_polygon(polygon, angle, center_x, center_y):
    coords = list(polygon.exterior.coords)
    rotated_coords = []
    cos_angle = np.cos(angle)
    sin_angle = np.sin(angle)
    for x, y in coords:
        tx = x - center_x
        ty = y - center_y
        rot_x = tx * cos_angle - ty * sin_angle + center_x
        rot_y = tx * sin_angle + ty * cos_angle + center_y
        rotated_coords.append((rot_x, rot_y))
    return Polygon(rotated_coords)

# Function to normalize polygons
def normalize_polygon(polygon, scale_factor, center_x, center_y):
    coords = list(polygon.exterior.coords)
    normalized_coords = []
    for x, y in coords:
        norm_x = (x - center_x) * scale_factor + 0.5  # Shift by 0.5 for center
        norm_y = (y - center_y) * scale_factor + 0.5  # Shift by 0.5 for center
        normalized_coords.append((norm_x, norm_y))
    return Polygon(normalized_coords)

def create_polygon_mask(image_size, polygon_coords):
    polygon_coords = np.round(polygon_coords).astype(np.int32).tolist()
    polygon_coords = [tuple(inner_list) for inner_list in polygon_coords]

    # 빈 마스크 이미지 생성 (0으로 초기화된 배열)
    mask = Image.new('L', image_size, 0)
    empty_array = np.array(mask)

    # Draw 객체 생성
    draw = ImageDraw.Draw(mask)

    # 폴리곤을 채워서 그리기 (1로 채워짐)
    draw.polygon(polygon_coords, outline=1, fill=1)

    # numpy 배열로 변환 (0과 1로 구성된 배열)
    filled_mask_array = np.array(mask)

    # Draw 객체 생성
    draw = ImageDraw.Draw(mask)

    # 폴리곤을 채워서 그리기 (1로 채워짐)
    draw.polygon(polygon_coords, outline=1, fill=0)

    # numpy 배열로 변환 (0과 1로 구성된 배열)
    outline_mask_array = np.array(mask)

    return filled_mask_array, outline_mask_array, empty_array

city_list = [
    "Atlanta", "Boston", "Dallas", "Denver", "Houston", "Lasvegas",
    "Littlerock", "Miami", "NewOrleans", "Philadelphia", "Phoenix",
    "Pittsburgh", "Portland", "Providence", "Richmond", "Saintpaul",
    "Sanfrancisco", "Seattle", "Washington"
]

for city in city_list:
    print(city)
    # Load the pickle file
    boundary_folder_path = f'C:/Users/ttd85/Documents/Resources/{city}/Boundaries'
    building_folder_path = f'C:/Users/ttd85/Documents/Resources/{city}/Buildings'

    # Create directories if they don't exist
    os.makedirs(f'{city}/gt', exist_ok=True)
    os.makedirs(f'{city}/mask', exist_ok=True)
    os.makedirs(f'{city}/polygon', exist_ok=True)

    boundary_files = get_geojson_files(boundary_folder_path)
    building_files = get_geojson_files(building_folder_path)

    matching_boundary_files, matching_building_files = get_matching_files(boundary_files, building_files)

    # Save the lists to variables
    boundary_files = matching_boundary_files
    building_files = matching_building_files

    # Create PNGs for each block
    for idx, (boundary_file, building_file) in enumerate(zip(tqdm(boundary_files), building_files)):
        fig, ax = plt.subplots()
        ax.set_aspect('equal')

        with open(boundary_file, 'r') as f:
            boundary_data = json.load(f)['features'][0]['geometry']['coordinates'][0]
        with open(building_file, 'r') as f:
            building_data = json.load(f)['features']

        boundary_polygon = Polygon(transform_coordinates(boundary_data))
        building_polygons = []
        for building in building_data:
            coords = building['geometry']['coordinates'][0]
            building_polygons.append(Polygon(transform_coordinates(coords)))

        # Calculate the centroid of the block
        centroid = calculate_centroid(boundary_polygon)
        center_x, center_y = centroid.x, centroid.y

        # Calculate the rotation angle based on the longest side of the minimum rotated bounding box
        rotation_angle = calculate_rotation_angle(boundary_polygon)

        # Rotate the block polygon
        rotated_boundary_polycon = rotate_polygon(boundary_polygon, rotation_angle, center_x, center_y)
        min_rot_rect = rotated_boundary_polycon.minimum_rotated_rectangle
        min_x, min_y, max_x, max_y = min_rot_rect.bounds
        rotated_centroid = calculate_centroid(min_rot_rect)
        rotated_center_x, rotated_center_y = rotated_centroid.x, rotated_centroid.y

        # Determine the scale factor to normalize the longest side to 1
        bbox_width = max_x - min_x
        bbox_height = max_y - min_y
        max_length = max(bbox_width, bbox_height)
        scale_factor = 0.9 / max_length

        # Normalize the rotated block polygon
        normalized_boundary_block = normalize_polygon(rotated_boundary_polycon, scale_factor, rotated_center_x, rotated_center_y)
        plot_polygon(ax, normalized_boundary_block, edge_color='black')

        # Normalize and plot each building bounding box
        normalized_buildings = []
        normalized_building_bboxs = []
        for building_polygon in building_polygons:
            rotated_building_polygon = rotate_polygon(building_polygon, rotation_angle, center_x, center_y)
            normalized_building_polygon = normalize_polygon(rotated_building_polygon, scale_factor, rotated_center_x, rotated_center_y)

            building_bbox = building_polygon.minimum_rotated_rectangle
            rotated_building_bbox = rotate_polygon(building_bbox, rotation_angle, center_x, center_y)
            normalized_building_bbox = normalize_polygon(rotated_building_bbox, scale_factor, rotated_center_x, rotated_center_y)

            normalized_buildings.append(normalized_building_polygon)
            normalized_building_bboxs.append(normalized_building_bbox)
            plot_polygon(ax, normalized_building_polygon, edge_color='red')

        # Set limits and labels
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.set_title(f'Block {idx + 1}')
        ax.set_xlabel('Normalized X')
        ax.set_ylabel('Normalized Y')

        # Save the figure
        plt.savefig(f'{city}/gt/block_{idx + 1}.png')
        plt.close()

        # 이미지 크기와 폴리곤 좌표 설정
        image_size = (224, 224)
        polygon_coords = np.array(list(normalized_boundary_block.exterior.coords)) * 224

        # 마스크 생성
        filled_mask_array, outline_mask_array, empty_array = create_polygon_mask(image_size, polygon_coords)

        # # 결과 확인 (Pillow 이미지로 변환)
        # Image.fromarray(filled_mask_array * 255).show()
        # Image.fromarray(outline_mask_array * 255).show()
        # Image.fromarray(empty_array * 255).show()

        mask = [filled_mask_array, outline_mask_array, empty_array]
        with open(f'{city}/mask/block_{idx + 1}.pkl', 'wb') as f:
            pickle.dump(mask, f)

        save_path = f'{city}/polygon/block_{idx + 1}.pkl'

        # 저장할 데이터
        transformed_data = {
            'normalized_block_polygon': normalized_boundary_block,
            'normalized_buildings_polygons': normalized_buildings,
            'normalized_building_bboxs': normalized_building_bboxs,
            'scale_factor': scale_factor,
            'rotation_angle': rotation_angle,
            'boundary_file_path': boundary_file,
            'building_file_path': building_file,
            'raw_boundary_polygon': boundary_polygon,
            'raw_building_polygons': building_polygons,
        }

        # pkl 파일로 저장
        with open(save_path, 'wb') as f:
            pickle.dump(transformed_data, f)
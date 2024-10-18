import os
import pickle
from shapely.geometry import Polygon, MultiPoint
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.cm as cm
import random

city_list = [
    "Atlanta", "Boston", "Dallas", "Denver", "Houston", "Lasvegas",
    "Littlerock", "Miami", "NewOrleans", "Philadelphia", "Phoenix",
    "Pittsburgh", "Portland", "Providence", "Richmond", "Saintpaul",
    "Sanfrancisco", "Seattle", "Washington"
]

UNIT_LENGTH = 10

def remove_overlapping_segments(boundary_coords, unit_length):
    i = 0
    while i < len(boundary_coords):
        x1, y1 = boundary_coords[i]
        x2, y2 = boundary_coords[(i + 1) % len(boundary_coords)]

        distance = np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2]))
        num_segments = int(distance // unit_length)

        if num_segments == 0:
            del boundary_coords[(i + 1) % len(boundary_coords)]
        else:
            i += 1

    return boundary_coords

def divide_boundary_into_segments(boundary_coords, unit_length):
    new_boundary_coords = []

    for i in range(len(boundary_coords)):
        x1, y1 = boundary_coords[i]
        x2, y2 = boundary_coords[(i + 1) % len(boundary_coords)]

        # 두 점 사이의 거리 계산
        distance = np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2]))

        # 두 점 사이의 세그먼트 수 계산
        num_segments = int(distance // unit_length)

        # 시작점 추가
        new_boundary_coords.append([x1, y1])

        # 세그먼트를 단위 길이만큼 추가
        if num_segments > 0:
            vector = np.array([x2 - x1, y2 - y1]) / distance
            for j in range(1, num_segments):
                new_point = np.array([x1, y1]) + vector * unit_length * j
                new_boundary_coords.append(new_point.tolist())

    return new_boundary_coords

def nearest_index_from_origin(boundary_coords):
    boundary_coords = np.array([list(t) for t in boundary_coords])
    distance = boundary_coords[:, 0] ** 2 + boundary_coords[:, 1] ** 2

    return distance.argmin()

def visualize_boundary_with_buildings(boundary_coords, building_polygons, convex_hull,
                                      convex_hull_percentage, polygon_file_path):
    # Create a copy to avoid modifying the original list
    coords_copy = boundary_coords.copy()

    # Boundary를 닫기 위해 첫 좌표를 마지막에 추가
    coords_copy.append(coords_copy[0])

    # 리스트를 numpy 배열로 변환
    coords_array = np.array(coords_copy)

    # 인덱스를 색상으로 매핑하기 위해 색상 맵(cmap)을 사용
    num_points = len(coords_array)
    colors = cm.viridis(np.linspace(0, 1, num_points))  # viridis 색상 맵을 사용

    fig, ax = plt.subplots()

    # 각 좌표를 선으로 연결하는 플롯 (경계 시각화)
    ax.plot(coords_array[:, 0], coords_array[:, 1], marker='o', color='gray', alpha=0.5)

    # 각 좌표를 색상과 함께 플롯 (색상 그라디언트)
    for i in range(num_points):
        ax.scatter(coords_array[i, 0], coords_array[i, 1], color=colors[i], s=100)

    num_buildings = len(building_polygons)
    colors = ['#%06X' % random.randint(0, 0xFFFFFF) for _ in range(num_buildings + 1)]

    # Building polygons를 시각화
    for i, poly in enumerate(building_polygons):
        if isinstance(poly, Polygon):  # 만약 `poly`가 `Polygon` 객체라면
            x, y = poly.exterior.xy  # 외곽선의 좌표를 추출
            ax.fill(x, y, alpha=0.5, fc=colors[i], ec='black')  # 폴리곤을 시각화
            ax.text(poly.centroid.x, poly.centroid.y, str(i+1), fontsize=12, ha='center', va='center', color='black')

    x, y = convex_hull.exterior.xy  # 외곽선의 좌표를 추출
    ax.fill(x, y, alpha=0.5, fc=colors[-1], ec='black')  # 폴리곤을 시각화

    # 시각화 설정
    ax.set_title(f'{polygon_file_path} ({round(convex_hull_percentage, 2)} %)')
    ax.set_aspect('equal')
    plt.show()

# 겹치는 다각형이 있는지 확인하는 함수
def check_overlapping_polygons(polygon_list):
    for i in range(len(polygon_list)):
        for j in range(i + 1, len(polygon_list)):
            if polygon_list[i].intersects(polygon_list[j]):
                return True  # 하나라도 겹치면 True 반환
    return False  # 전부 겹치지 않으면 False 반환


def remove_duplicate_polygons(polygon_list):
    seen = set()
    unique_polygons = []

    for polygon in polygon_list:
        # WKT 표현을 사용하여 폴리곤의 고유성을 체크
        polygon_wkt = polygon.wkt
        if polygon_wkt not in seen:
            seen.add(polygon_wkt)
            unique_polygons.append(polygon)

    return unique_polygons

flag_1 = True
flag_2 = True
for city in city_list:
    print(city)
    folder_path = f'./1_Raw2Polygon/{city}/polygon/'
    all_files = os.listdir(folder_path)

    os.makedirs(f'./2_FilteredData/{city}', exist_ok=True)

    file_idx = 1
    for idx in tqdm(range(len(all_files))):
        polygon_file_path = f'./1_Raw2Polygon/{city}/polygon/block_{idx + 1}.pkl'
        with open(polygon_file_path, 'rb') as f:
            polygon_data = pickle.load(f)

        normalized_boundary = polygon_data['normalized_block_polygon']
        normalized_buildings = polygon_data['normalized_buildings_polygons']
        raw_boundary_file_path = polygon_data['boundary_file_path']
        raw_building_file_path = polygon_data['building_file_path']
        scale_factor = 0.9 / polygon_data['scale_factor']
        unit_length = UNIT_LENGTH / scale_factor

        if len(normalized_buildings) >= 300:
            continue

        normalized_buildings = remove_duplicate_polygons(normalized_buildings)

        simplified_boundary = normalized_boundary.simplify(0.001, preserve_topology=True)
        simplified_buildings = [building.simplify(0.001, preserve_topology=True) for building in normalized_buildings]

        is_valid = True
        for building in simplified_buildings:
            if simplified_boundary.boundary.intersects(building):
                is_valid = False
                break

        if not is_valid:
            continue

        boundary_coords = list(simplified_boundary.exterior.coords)
        boundary_segments = divide_boundary_into_segments(boundary_coords, unit_length)

        building_point_count_list = [len(list(building.exterior.coords)) - 1 for building in simplified_buildings]

        # Store the number of segments for this boundary
        if len(simplified_buildings) == 1:
            flag_1 = not flag_1
            if flag_1:
                continue

        if len(simplified_buildings) == 2:
            flag_2 = not flag_2
            if flag_2:
                continue

        if len(boundary_segments) <= 200 and len(simplified_buildings) > 0 and len(simplified_buildings) <= 100 and not check_overlapping_polygons(simplified_buildings) and max(building_point_count_list) <= 16:
            # 건물 외곽 좌표를 사용
            exterior_coords_list = [list(poly.exterior.coords) for i, poly in enumerate(simplified_buildings)]
            all_coords = np.vstack(exterior_coords_list)  # 리스트의 리스트를 하나의 배열로 결합
            center = np.mean(all_coords, axis=0)  # 중앙점 계산
            angles = np.arctan2(all_coords[:, 1] - center[1], all_coords[:, 0] - center[0])
            sorted_coords = all_coords[np.argsort(angles)]
            multipoint = MultiPoint(sorted_coords)

            # 볼록 껍질 계산
            convex_hull = multipoint.convex_hull
            percentage_convex_hull = (convex_hull.area / simplified_boundary.area) * 100

            if percentage_convex_hull < 20:
                continue

            if percentage_convex_hull > 200:
                continue

            centroid_list = [(i, poly.centroid.x, round(poly.centroid.y, 2)) for i, poly in enumerate(simplified_buildings)]
            sorted_indices = sorted(centroid_list, key=lambda c: (c[2], c[1]))
            sorted_indices = np.array([list(t)[0] for t in sorted_indices])

            simplified_buildings = np.array(simplified_buildings)[sorted_indices]

            nearest_index = nearest_index_from_origin(boundary_segments)
            boundary_segments = boundary_segments[nearest_index:] + boundary_segments[:nearest_index]
            # visualize_boundary_with_buildings(boundary_segments, simplified_buildings, convex_hull, percentage_convex_hull, polygon_file_path)

            save_path = f'./2_FilteredData/{city}/block_{file_idx}.pkl'
            file_idx += 1

            # 저장할 데이터
            transformed_data = {
                'simplified_boundary': simplified_boundary,
                'simplified_boundary_segments': boundary_segments,
                'simplified_building_polygons': simplified_buildings,
                'polygon_file_path': polygon_file_path,
                'raw_boundary_file_path': raw_boundary_file_path,
                'raw_building_file_path': raw_building_file_path,
                'city_name': city
            }

            # pkl 파일로 저장
            with open(save_path, 'wb') as f:
                pickle.dump(transformed_data, f)
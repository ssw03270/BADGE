import os
import pickle
from shapely.geometry import Polygon, MultiPoint
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.cm as cm
import random
import seaborn as sns

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
                                      building_percentage, convex_hull_percentage):
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
    ax.set_title(f'Boundary and Building Polygons ({round(building_percentage, 2)} %, {round(convex_hull_percentage, 2)} %)')
    ax.set_aspect('equal')
    plt.show()

# 겹치는 다각형이 있는지 확인하는 함수
def check_overlapping_polygons(polygon_list):
    for i in range(len(polygon_list)):
        for j in range(i + 1, len(polygon_list)):
            if polygon_list[i].intersects(polygon_list[j]):
                return True  # 하나라도 겹치면 True 반환
    return False  # 전부 겹치지 않으면 False 반환

convex_hull_percentages = []

for city in city_list:
    print(city)
    folder_path = f'./1_Raw2Polygon/{city}/polygon/'
    all_files = os.listdir(folder_path)

    os.makedirs(f'./2_FilteredData/{city}', exist_ok=True)

    file_idx = 1
    for idx in tqdm(range(len(all_files))):
        file_path = f'./1_Raw2Polygon/{city}/polygon/block_{idx + 1}.pkl'
        with open(file_path, 'rb') as f:
            polygon_data = pickle.load(f)

        normalized_boundary = polygon_data['normalized_block_polygon']
        normalized_buildings = polygon_data['normalized_buildings_polygons']
        scale_factor = 0.9 / polygon_data['scale_factor']
        unit_length = UNIT_LENGTH / scale_factor

        simplified_polygon = normalized_boundary.simplify(0.001, preserve_topology=True)
        simplified_buildings = [building.simplify(0.001, preserve_topology=True) for building in normalized_buildings]

        boundary_coords = list(simplified_polygon.exterior.coords)
        modified_boundary_coords = remove_overlapping_segments(boundary_coords, unit_length)
        boundary_segments = divide_boundary_into_segments(modified_boundary_coords, unit_length)
        boundary_segments = list(normalized_boundary.exterior.coords)

        building_point_count_list = [len(list(building.exterior.coords)) - 1 for building in simplified_buildings]

        # Store the number of segments for this boundary
        if len(boundary_segments) <= 200 and len(simplified_buildings) > 2 and len(simplified_buildings) <= 100 and not check_overlapping_polygons(simplified_buildings) and max(building_point_count_list) <= 16:
            area_a = simplified_polygon.area
            area_b_sum = sum(polygon.area for polygon in simplified_buildings)
            percentage_building_area = (area_b_sum / area_a) * 100

            # 건물 외곽 좌표를 사용
            exterior_coords_list = [list(poly.exterior.coords) for i, poly in enumerate(simplified_buildings)]

            # 외곽 좌표의 평균을 중앙점으로 사용
            all_coords = np.vstack(exterior_coords_list)  # 리스트의 리스트를 하나의 배열로 결합
            center = np.mean(all_coords, axis=0)  # 중앙점 계산

            # 각 외곽 좌표가 중앙점에 대해 형성하는 각도 계산
            angles = np.arctan2(all_coords[:, 1] - center[1], all_coords[:, 0] - center[0])

            # 각도에 따라 외곽 좌표 정렬
            sorted_coords = all_coords[np.argsort(angles)]

            # 정렬된 외곽 좌표를 사용하여 MultiPoint 객체 생성
            multipoint = MultiPoint(sorted_coords)

            # 볼록 껍질 계산
            convex_hull = multipoint.convex_hull
            percentage_convex_hull = (convex_hull.area / area_a) * 100

            # convex hull 비율을 리스트에 저장
            convex_hull_percentages.append(percentage_convex_hull)

    # 볼록 껍질의 비율 분포 시각화
    plt.figure(figsize=(10, 6))
    sns.histplot(convex_hull_percentages, kde=False, bins=30, color='green')
    plt.title('Convex Hull Percentage Distribution Across Cities')
    plt.xlabel('Percentage of Convex Hull Area relative to Boundary Area (%)')
    plt.ylabel('Frequency')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
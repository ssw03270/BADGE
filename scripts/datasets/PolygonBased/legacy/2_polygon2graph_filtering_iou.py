import os
import pickle
from shapely.geometry import Polygon
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib.cm as cm

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


def visualize_boundary_with_buildings(boundary_coords, building_polygons):
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

    # Building polygons를 시각화
    for poly in building_polygons:
        if isinstance(poly, Polygon):  # 만약 `poly`가 `Polygon` 객체라면
            x, y = poly.exterior.xy  # 외곽선의 좌표를 추출
            ax.fill(x, y, alpha=0.5, fc='orange', ec='black')  # 폴리곤을 시각화

    # 시각화 설정
    ax.set_title('Boundary and Building Polygons')
    ax.set_aspect('equal')
    plt.show()

def overlap_percentage(polygon_a, polygon_b):
    # 두 폴리곤의 교차 면적 계산
    intersection_area = polygon_a.intersection(polygon_b).area

    # 각 폴리곤의 면적 계산
    area_a = polygon_a.area
    area_b = polygon_b.area

    # 겹치는 면적을 polygon_a의 면적에 대한 퍼센트로 구함
    overlap_percent_a = (intersection_area / area_a) * 100

    # 겹치는 면적을 polygon_b의 면적에 대한 퍼센트로 구함
    overlap_percent_b = (intersection_area / area_b) * 100

    return overlap_percent_b

percentage_list = []
a = 0
b = 0
for city in city_list:
    print(city)
    folder_path = f'./{city}/polygon/'
    all_files = os.listdir(folder_path)

    for idx in tqdm(range(len(all_files))):
        file_path = f'./{city}/polygon/block_{idx + 1}.pkl'
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        normalized_boundary = data['normalized_block_polygon']
        normalized_buildings = data['normalized_buildings_polygons']
        scale_factor = 0.9 / data['scale_factor']
        unit_length = UNIT_LENGTH / scale_factor

        boundary_coords = list(normalized_boundary.exterior.coords)
        modified_boundary_coords = remove_overlapping_segments(boundary_coords, unit_length)
        boundary_segments = divide_boundary_into_segments(modified_boundary_coords, unit_length)

        flag = False
        # Store the number of segments for this boundary
        if len(boundary_segments) <= 200 and len(normalized_buildings) > 2 and len(normalized_buildings) <= 100:
            for normalized_building in normalized_buildings:
                overlap_percent = overlap_percentage(normalized_building, normalized_building.minimum_rotated_rectangle)
                percentage_list.append(overlap_percent)

                if overlap_percent < 70:
                    flag = True

        if flag:
            b += 1
        else:
            a += 1
    # Visualize the distribution of segment counts
    plt.figure(figsize=(10, 6))
    plt.hist(percentage_list, bins=100, edgecolor='black')
    plt.title('Distribution of Boundary Segment Counts')
    plt.xlabel('Number of Segments')
    plt.ylabel('Frequency')
    plt.show()

    print('70 up:', a, '70 down:', b)
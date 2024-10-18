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
        normalized_buildings_polygons = polygon_data['normalized_buildings_polygons']

        simplified_polygon = normalized_boundary.simplify(0.001, preserve_topology=True)

        # 원본 폴리곤 좌표 개수
        original_points_count = len(normalized_boundary.exterior.coords)

        # 간소화된 폴리곤 좌표 개수
        simplified_points_count = len(simplified_polygon.exterior.coords)

        # 시각화를 위한 원본 폴리곤 좌표
        x_original, y_original = normalized_boundary.exterior.xy

        # 시각화를 위한 간소화된 폴리곤 좌표
        x_simplified, y_simplified = simplified_polygon.exterior.xy

        # 건물 폴리곤도 간소화
        simplified_buildings = [building.simplify(0.002, preserve_topology=True) for building in
                                normalized_buildings_polygons]

        # 원본 빌딩 점 개수 계산
        original_building_points_counts = [len(building.exterior.coords) for building in normalized_buildings_polygons]

        # 간소화된 빌딩 점 개수 계산
        simplified_building_points_counts = [len(building.exterior.coords) for building in simplified_buildings]

        # 서브플롯을 사용한 시각화 (1행 2열), 정사각형 비율 유지
        fig, axes = plt.subplots(1, 2, figsize=(8, 8))

        # 축을 0에서 1로 설정
        xlim = ylim = (0, 1)

        # 왼쪽: 원본 폴리곤 + 원본 건물
        axes[0].plot(x_original, y_original, color="blue", linewidth=2)
        axes[0].fill(x_original, y_original, alpha=0.5, fc='blue', ec='black')

        # 각 원본 건물 폴리곤 시각화
        for idx, building in enumerate(normalized_buildings_polygons):
            x_building, y_building = building.exterior.xy
            axes[0].plot(x_building, y_building, color="green", linewidth=1)
            axes[0].fill(x_building, y_building, alpha=0.3, fc='green', ec='black')

        axes[0].set_title(
            f"Original Polygon ({original_points_count} points)\nwith Buildings ({round(np.mean(original_building_points_counts), 1)} points)")
        axes[0].set_xlabel("X-axis")
        axes[0].set_ylabel("Y-axis")
        axes[0].grid(True)
        axes[0].set_xlim(xlim)
        axes[0].set_ylim(ylim)
        axes[0].set_aspect('equal')  # 정사각형 비율 유지

        # 오른쪽: 간소화된 폴리곤 + 간소화된 건물
        axes[1].plot(x_simplified, y_simplified, color="red", linewidth=2)
        axes[1].fill(x_simplified, y_simplified, alpha=0.5, fc='red', ec='black')

        # 각 간소화된 건물 폴리곤 시각화
        for idx, simplified_building in enumerate(simplified_buildings):
            x_building, y_building = simplified_building.exterior.xy
            axes[1].plot(x_building, y_building, color="green", linewidth=1)
            axes[1].fill(x_building, y_building, alpha=0.3, fc='green', ec='black')

        axes[1].set_title(
            f"Simplified Polygon ({simplified_points_count} points)\nwith Simplified Buildings ({round(np.mean(simplified_building_points_counts), 1)} points)")
        axes[1].set_xlabel("X-axis")
        axes[1].set_ylabel("Y-axis")
        axes[1].grid(True)
        axes[1].set_xlim(xlim)
        axes[1].set_ylim(ylim)
        axes[1].set_aspect('equal')  # 정사각형 비율 유지

        # 서브플롯 간격 조정
        plt.tight_layout()

        # 시각화
        plt.show()
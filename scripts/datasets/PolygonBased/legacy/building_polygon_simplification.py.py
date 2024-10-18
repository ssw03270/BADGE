import os
import pickle
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from shapely.geometry import Polygon

city_list = [
    "Atlanta", "Boston", "Dallas", "Denver", "Houston", "Lasvegas",
    "Littlerock", "Miami", "NewOrleans", "Philadelphia", "Phoenix",
    "Pittsburgh", "Portland", "Providence", "Richmond", "Saintpaul",
    "Sanfrancisco", "Seattle", "Washington"
]

# 분석을 위한 데이터 저장 리스트
original_building_points_counts = []
simplified_building_points_counts = []

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

        # 원본 및 간소화된 폴리곤 데이터
        normalized_buildings_polygons = polygon_data['normalized_buildings_polygons']

        # 각 빌딩 폴리곤에 대해 점 개수 계산
        for building in normalized_buildings_polygons:
            # 원본 빌딩 폴리곤의 점 개수
            original_points_count = len(building.exterior.coords)
            original_building_points_counts.append(original_points_count)

            # 간소화된 빌딩 폴리곤의 점 개수
            simplified_building = building.simplify(0.001, preserve_topology=True)
            simplified_points_count = len(simplified_building.exterior.coords)
            simplified_building_points_counts.append(simplified_points_count)

    # 히스토그램 그리기
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # 원본 빌딩 폴리곤에 대한 점 개수 분포 (최대 50, 5단위로 나누기)
    bins = np.arange(0, 21, 1)

    # 원본 빌딩 폴리곤 분포
    sns.histplot(original_building_points_counts, kde=False, ax=axes[0], color='blue', bins=bins)
    axes[0].set_title('Original Building Polygon Point Count Distribution')
    axes[0].set_xlabel('Number of Points')
    axes[0].set_ylabel('Frequency')
    axes[0].set_xlim(0, 20)  # x축 범위를 0에서 50으로 설정
    axes[0].set_xticks(bins)  # x축 구간을 5단위로 설정

    # 간소화된 빌딩 폴리곤 분포
    sns.histplot(simplified_building_points_counts, kde=False, ax=axes[1], color='red', bins=bins)
    axes[1].set_title('Simplified Building Polygon Point Count Distribution')
    axes[1].set_xlabel('Number of Points')
    axes[1].set_ylabel('Frequency')
    axes[1].set_xlim(0, 20)  # x축 범위를 0에서 50으로 설정
    axes[1].set_xticks(bins)  # x축 구간을 5단위로 설정

    # 두 그래프에서 나타난 최대 빈도값 구하기
    max_y_value = max(axes[0].get_ylim()[1], axes[1].get_ylim()[1])

    # Y축을 두 그래프에 동일하게 설정
    axes[0].set_ylim(0, max_y_value)
    axes[1].set_ylim(0, max_y_value)

    # 그래프 간격 및 레이아웃 조정
    plt.tight_layout()
    plt.show()

import matplotlib.pyplot as plt

import os
import pickle
import numpy as np
from tqdm import tqdm
from shapely.ops import unary_union

UNIT_LENGTH = 0.00010

def divide_boundary_into_segments(boundary_coords, unit_length):
    new_boundary_coords = []
    lower_left_point = None

    for i in range(len(boundary_coords) - 1):
        x1, y1 = boundary_coords[i]
        x2, y2 = boundary_coords[(i + 1) % len(boundary_coords)]

        # 두 점 사이의 거리 계산
        distance = np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2]))

        # 두 점 사이의 세그먼트 수 계산
        num_segments = int(distance // unit_length)

        # 시작점 추가
        new_boundary_coords.append([x1, y1])

        # 좌하단 점 업데이트
        if lower_left_point is None or (y1 < lower_left_point[1] or (y1 == lower_left_point[1] and x1 < lower_left_point[0])):
            lower_left_point = [x1, y1]

        # 세그먼트를 단위 길이만큼 추가
        if num_segments > 0:
            vector = np.array([x2 - x1, y2 - y1]) / distance
            for j in range(1, num_segments):
                new_point = np.array([x1, y1]) + vector * unit_length * j
                new_boundary_coords.append(new_point.tolist())

                # 좌하단 점 업데이트
                if lower_left_point is None or (new_point[1] < lower_left_point[1] or (new_point[1] == lower_left_point[1] and new_point[0] < lower_left_point[0])):
                    lower_left_point = new_point.tolist()
                    
    start_index = new_boundary_coords.index(lower_left_point)  # lower_left_point의 인덱스 찾기
    new_boundary_coords = new_boundary_coords[start_index:] + new_boundary_coords[:start_index]  # 시작점 기준으로 재정렬
    new_boundary_coords = new_boundary_coords + [new_boundary_coords[0]]
    return new_boundary_coords

dataset_path = "E:/Resources/COHO/Our_Dataset"
output_dir_path = "E:/Resources/COHO"
subfolders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]

for folder in subfolders:
    output_path = os.path.join(output_dir_path, 'outputs', folder)  # output 경로 설정
    os.makedirs(output_path, exist_ok=True)  # output 디렉토리 생성

    output_file_path = os.path.join(output_path, f'{folder}_segments.pkl')
    if os.path.exists(output_file_path):
        print(f"{output_file_path} 파일이 이미 존재합니다. 건너뜁니다.")
        continue  # 다음 폴더로 넘어감

    data_path = os.path.join(dataset_path, folder, f'shapefile/{folder}_clipped_blk.pkl')
    if not os.path.exists(data_path):
        print(f"{data_path} 파일이 존재하지 않습니다. 건너뜁니다.")
        continue  # 다음 폴더로 넘어감

    with open(data_path, 'rb') as f:
        data = pickle.load(f)
    geometry_list= data[0]

    blk_segments_list = []
    is_multi_polygon_list = []
    for blk_geometry in tqdm(geometry_list):
        is_multi_polygon = False
        if blk_geometry.geom_type == 'MultiPolygon':
            union = unary_union(blk_geometry)
            blk_coords = [list(coord) for polygon in union.geoms for coord in polygon.exterior.coords]
            is_multi_polygon = True
        else:
            blk_coords = [list(coord) for coord in blk_geometry.exterior.coords]
            is_multi_polygon = False

        blk_segments = divide_boundary_into_segments(blk_coords, UNIT_LENGTH)
        blk_segments_list.append(blk_segments)
        is_multi_polygon_list.append(is_multi_polygon)
    new_data = data + [blk_segments_list] + [is_multi_polygon_list]

    # pkl 파일로 저장
    with open(os.path.join(output_file_path), 'wb') as f:
        pickle.dump(new_data, f)

import matplotlib.pyplot as plt

import os
import pickle
import numpy as np
from tqdm import tqdm
from pyproj import Proj, transform
from shapely.affinity import rotate
from shapely.affinity import translate
from shapely.ops import unary_union
from shapely.geometry import Polygon

UNIT_LENGTH = 1

dataset_path = "E:/Resources/COHO/Our_Dataset"
output_dir_path = "E:/Resources/COHO"
subfolders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]

# UTM 좌표계 정의 (여기서는 예시로 UTM Zone 14N을 사용)
utm_proj = Proj(proj='utm', zone=14, ellps='WGS84')
# WGS84 좌표계 정의
wgs84_proj = Proj(proj='latlong', datum='WGS84')

def convert_to_utm(coords):
    utm_coords = []
    for lon, lat in coords:
        x, y = utm_proj(lon, lat)  # 위도 경도를 UTM으로 변환
        utm_coords.append([x, y])
    return utm_coords

def normalize_coords(coords):
    coords = np.array(coords)
    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0)
    return (coords - min_coords) / (max_coords - min_coords)  # 0~1 사이로 정규화

def normalize_coords_uniform(coords):
    coords = np.array(coords)
    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0)
    range_max = (max_coords - min_coords).max()
    return (coords - min_coords) / range_max

def get_minimum_bounding_rectangle(coords):
    """폴리곤의 최소 경계 사각형을 계산합니다."""
    polygon = Polygon(coords)
    min_rect = polygon.minimum_rotated_rectangle
    return min_rect

def get_rotation_angle(coords):
    """
    MBR의 긴 변이 x축에 평행하도록 회전하기 위한 각도를 계산합니다.
    
    Args:
        coords (list of tuple): 폴리곤의 좌표 리스트.
    
    Returns:
        float: 회전 각도(도 단위).
    """
    polygon = Polygon(coords)
    min_rect = get_minimum_bounding_rectangle(polygon)
    x, y = min_rect.exterior.coords.xy

    # MBR은 첫 번째 점이 마지막 점과 동일하므로 첫 두 변을 검사
    edge1 = np.array([x[1] - x[0], y[1] - y[0]])
    edge2 = np.array([x[2] - x[1], y[2] - y[1]])

    # 각 변의 길이 계산
    length1 = np.linalg.norm(edge1)
    length2 = np.linalg.norm(edge2)

    # 더 긴 변을 선택
    if length1 >= length2:
        longer_edge = edge1
    else:
        longer_edge = edge2

    # 회전 각도 계산 (x축과의 각도)
    angle = np.degrees(np.arctan2(longer_edge[1], longer_edge[0]))

    return angle % 360

def center_coords(coords, target_point=(0, 0)):
    """
    폴리곤을 특정 점을 중심으로 이동시킵니다.
    
    Parameters:
    - polygon (shapely.geometry.Polygon): 이동할 폴리곤
    - target_point (tuple): 폴리곤의 BBox 중심을 이동시킬 목표 점 (기본값은 원점)
    
    Returns:
    - shifted_polygon (shapely.geometry.Polygon): 이동된 폴리곤
    """
    # BBox의 최소 및 최대 좌표 구하기
    polygon = Polygon(coords)
    minx, miny, maxx, maxy = polygon.bounds
    # BBox의 중심 계산
    center_x = (minx + maxx) / 2
    center_y = (miny + maxy) / 2
    # 이동 벡터 계산
    shift_x = target_point[0] - center_x
    shift_y = target_point[1] - center_y
    # 폴리곤 이동
    shifted_polygon = translate(polygon, xoff=shift_x, yoff=shift_y)
    return [list(coord) for coord in shifted_polygon.exterior.coords]

def create_grid_dict(grid_size=64):
    grid_dict = {}
    for i in range(grid_size):
        for j in range(grid_size):
            index = i * grid_size + j  # 0부터 시작하는 인덱스
            grid_dict[index] = (i / (grid_size- 1), j / (grid_size - 1))  # (x, y) 좌표
    return grid_dict

def find_grid_corners(points, grid_size=64):
    grid_corner_coords = []
    grid_corner_indices = []
    for point in points:
        x, y = point        
        corner_coords = []
        corner_indices = []

        for index, (grid_x, grid_y) in grid_dict.items():
            if grid_x <= x <= grid_x + (1 / (grid_size - 1)) and grid_y <= y <= grid_y + (1 / (grid_size - 1)):
                # 좌하단, 우하단, 좌상단, 우상단 점을 찾기
                corner_coords.append([grid_x, grid_y])  # 현재 그리드 점 추가
                corner_coords.append([grid_x + (1 / (grid_size - 1)), grid_y])  # 우하단
                corner_coords.append([grid_x, grid_y + (1 / (grid_size - 1))])  # 좌상단
                corner_coords.append([grid_x + (1 / (grid_size - 1)), grid_y + (1 / (grid_size - 1))])  # 우상단
                
                corner_indices.append(index)  # 현재 그리드 점 추가
                corner_indices.append(index + 1)  # 우하단
                corner_indices.append(index + grid_size)  # 좌상단
                corner_indices.append(index + grid_size + 1)  # 우상단
                break
        grid_corner_coords.append(corner_coords)
        grid_corner_indices.append(corner_indices)
    return grid_corner_coords, grid_corner_indices

def divide_boundary_into_segments(building_coords, unit_length):
    new_building_coords = []
    lower_left_point = None

    for i in range(len(building_coords) - 1):
        x1, y1 = building_coords[i]
        x2, y2 = building_coords[(i + 1) % len(building_coords)]

        # 두 점 사이의 거리 계산
        distance = np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2]))

        # 두 점 사이의 세그먼트 수 계산
        num_segments = int(distance // unit_length)

        # 시작점 추가
        new_building_coords.append([x1, y1])

        # 좌하단 점 업데이트
        if lower_left_point is None or (y1 < lower_left_point[1] or (y1 == lower_left_point[1] and x1 < lower_left_point[0])):
            lower_left_point = [x1, y1]

        # 세그먼트를 단위 길이만큼 추가
        if num_segments > 0:
            vector = np.array([x2 - x1, y2 - y1]) / distance
            for j in range(1, num_segments):
                new_point = np.array([x1, y1]) + vector * unit_length * j
                new_building_coords.append(new_point.tolist())

                # 좌하단 점 업데이트
                if lower_left_point is None or (new_point[1] < lower_left_point[1] or (new_point[1] == lower_left_point[1] and new_point[0] < lower_left_point[0])):
                    lower_left_point = new_point.tolist()
                    
    start_index = new_building_coords.index(lower_left_point)  # lower_left_point의 인덱스 찾기
    new_building_coords = new_building_coords[start_index:] + new_building_coords[:start_index]  # 시작점 기준으로 재정렬
    new_building_coords = new_building_coords + [new_building_coords[0]]
    return new_building_coords

def divide_boundary_into_n_segments(building_coords, n):
    """
    Divides the boundary of a polygon into n segments, ensuring that original vertices are included,
    and adds (n - len(building_coords)) new points as uniformly as possible.
    The reordered polygon starts from the point closest to the origin (0,0).
    
    Parameters:
        building_coords (list of [x, y]): Original polygon vertices.
        n (int): Total number of desired segments.
        
    Returns:
        list of [x, y]: New polygon coordinates with n segments.
    """
    m = len(building_coords)
    v = 0.1
    while n < m:
        polygon = Polygon(building_coords)
        polygon = polygon.simplify(v)
        v += 0.1
        building_coords = list(polygon.exterior.coords)
        m = len(building_coords)
    
    # Step 1: Calculate the length of each edge and total perimeter
    edge_lengths = []
    total_length = 0
    for i in range(m):
        x1, y1 = building_coords[i]
        x2, y2 = building_coords[(i + 1) % m]
        length = np.linalg.norm(np.array([x2 - x1, y2 - y1]))
        edge_lengths.append(length)
        total_length += length
    
    # Step 2: Determine the number of extra points to add
    extra_points = n - m
    
    # Distribute extra points proportionally based on edge lengths
    extra_points_per_edge = [ (length / total_length) * extra_points for length in edge_lengths ]
    
    # Convert to integer counts and handle rounding
    extra_points_int = [int(np.floor(x)) for x in extra_points_per_edge]
    remainder = extra_points - sum(extra_points_int)
    
    # Distribute the remaining points based on the largest fractional parts
    fractional_parts = [x - np.floor(x) for x in extra_points_per_edge]
    sorted_indices = np.argsort(fractional_parts)[::-1]  # Descending order
    for i in range(int(remainder)):
        extra_points_int[sorted_indices[i]] += 1
    
    # Step 3: Add new points to each edge
    new_building_coords = []
    for i in range(m):
        x1, y1 = building_coords[i]
        x2, y2 = building_coords[(i + 1) % m]
        new_building_coords.append([x1, y1])
        
        num_extra = extra_points_int[i]
        if num_extra > 0:
            vector = np.array([x2 - x1, y2 - y1])
            interval = edge_lengths[i] / (num_extra + 1)
            for j in range(1, num_extra + 1):
                new_point = np.array([x1, y1]) + (vector / edge_lengths[i]) * interval * j
                new_building_coords.append(new_point.tolist())
    
    # Step 4: Reorder to start from the point closest to the origin
    origin_closest_point = min(new_building_coords, key=lambda p: p[0]**2 + p[1]**2)
    start_index = new_building_coords.index(origin_closest_point)
    new_building_coords = new_building_coords[start_index:] + new_building_coords[:start_index]
    new_building_coords.append(new_building_coords[0])  # Close the polygon
    
    return new_building_coords

def plot_polygons(rotated, min_rect_rotated):
    """원본 및 회전된 폴리곤과 그 최소 경계 사각형을 시각화합니다."""
    fig, axs = plt.subplots(1, 1, figsize=(14, 7))

    # 회전된 폴리곤과 MBR
    axs.set_title('Rotated Polygon and MBR')
    x_rot, y_rot = rotated.exterior.xy
    axs.plot(x_rot, y_rot, label='Rotated Polygon')
    axs.plot(x_rot, y_rot, 'o', label='Rotated Polygon')

    rotated_points = list(zip(x_rot, y_rot))
    if rotated_points[0] == rotated_points[-1]:
        rotated_points = rotated_points[:-1]

    for idx, (x, y) in enumerate(rotated_points):
        axs.text(x + 0.01, y + 0.01, str(idx), fontsize=9, color='blue')

    x_mbr_rot, y_mbr_rot = min_rect_rotated.exterior.xy
    axs.plot(x_mbr_rot, y_mbr_rot, label='MBR', color='red')
    axs.legend()
    axs.set_aspect('equal')

    plt.show()
    
grid_dict = create_grid_dict(grid_size=64)
for folder in subfolders:
    output_path = os.path.join(output_dir_path, 'NormalizedBuildings', folder)  # output 경로 설정
    os.makedirs(output_path, exist_ok=True)  # output 디렉토리 생성

    data_path = os.path.join(dataset_path, folder, f'graph/{folder}_graph_prep_list.pkl')
    if not os.path.exists(data_path):
        print(f"{data_path} 파일이 존재하지 않습니다. 건너뜁니다.")
        continue  # 다음 폴더로 넘어감

    with open(data_path, 'rb') as f:
        data_list = pickle.load(f)
    
    for data in tqdm(data_list):
        if not 'bldg_poly_list' in data:
            continue
        
        bldg_poly_list = data['bldg_poly_list']
        for bldg_poly_data in bldg_poly_list:
            bldg_id = bldg_poly_data[0]
            bldg_poly = bldg_poly_data[1]
            bldg_coords = [list(coord) for coord in bldg_poly.exterior.coords]
            
            bldg_utm_coords = convert_to_utm(bldg_coords)
            angle = get_rotation_angle(bldg_utm_coords)
            rotated_bldg_poly = rotate(Polygon(bldg_utm_coords), -angle, origin='centroid', use_radians=False)
            rotated_bldg_coords = [list(coord) for coord in rotated_bldg_poly.exterior.coords]

            bldg_segments_coords = divide_boundary_into_n_segments(rotated_bldg_coords, 16)

            bldg_normalized_coords = normalize_coords_uniform(bldg_segments_coords)
            bldg_center_coords = center_coords(bldg_normalized_coords, target_point=(0.5, 0.5))

            bldg_center_coords = bldg_center_coords[:-1]
            bldg_grid_corner_coords, bldg_grid_corner_indices = find_grid_corners(bldg_center_coords)

            # 시각화
            # min_rect_rotated = get_minimum_bounding_rectangle(Polygon(bldg_center_coords))
            # plot_polygons(Polygon(bldg_center_coords), min_rect_rotated)
            if len(bldg_center_coords) == 16:
                data = {
                    "bldg_coords": bldg_center_coords,
                    "grid_corner_coords": bldg_grid_corner_coords,
                    "grid_corner_indices": bldg_grid_corner_indices
                }

                output_file_path = os.path.join(output_path, f'{folder}_{bldg_id}_normalized_buildings.pkl')
                # pkl 파일로 저장
                with open(os.path.join(output_file_path), 'wb') as f:
                    pickle.dump(data, f)

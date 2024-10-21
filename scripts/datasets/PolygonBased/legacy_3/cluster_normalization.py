import matplotlib.pyplot as plt
import multiprocessing

import os
import pickle
import numpy as np
from tqdm import tqdm
from pyproj import Proj, transform
from shapely.affinity import rotate
from shapely.affinity import translate
from shapely.ops import unary_union
from shapely.geometry import Polygon, box

UNIT_LENGTH = 1

dataset_path = "Z:/iiixr-drive/Projects/2023_City_Team/000_2024CVPR/COHO_dataset"
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

def normalize_coords_uniform(coords, min_coords=None, range_max=None):
    if not min_coords is None and not range_max is None:
        return (coords - min_coords) / range_max, min_coords, range_max
    
    coords = np.array(coords)
    min_coords = coords.min(axis=0)
    max_coords = coords.max(axis=0)
    range_max = (max_coords - min_coords).max()
    return (coords - min_coords) / range_max, min_coords, range_max

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

def create_bounding_box(x, y, w, h, r):
    """
    주어진 중심점, 너비, 높이, 회전 각도를 사용하여 바운딩 박스를 생성합니다.
    
    Parameters:
        x (float): 중심점 x 좌표
        y (float): 중심점 y 좌표
        w (float): 바운딩 박스의 너비
        h (float): 바운딩 박스의 높이
        r (float): 회전 각도 (도 단위)
    
    Returns:
        Polygon: 회전된 바운딩 박스의 Polygon 객체
    """
    # 바운딩 박스의 네 모서리 좌표 계산
    corners = np.array([
        [-w / 2, -h / 2],
        [w / 2, -h / 2],
        [w / 2, h / 2],
        [-w / 2, h / 2]
    ])
    
    # 회전 행렬 생성
    theta = np.radians(r)
    rotation_matrix = np.array([
        [np.cos(theta), -np.sin(theta)],
        [np.sin(theta), np.cos(theta)]
    ])
    
    # 모서리 회전 및 이동
    rotated_corners = corners @ rotation_matrix.T + np.array([x, y])
    
    return Polygon(rotated_corners)
    
debug = False
error_count = 0
def process_folder(folder):
    output_path = os.path.join(dataset_path, folder, f'graph/{folder}_graph_prep_list_with_clusters_detail.pkl')
    if os.path.exists(output_path):
        print(f"{output_path} 파일이 이미 존재합니다. 건너뜁니다.")
        return  # 이미 처리된 경우 건너뜁니다.

    data_path = os.path.join(dataset_path, folder, f'graph/{folder}_graph_prep_list_with_clusters.pkl')
    if not os.path.exists(data_path):
        print(f"{data_path} 파일이 존재하지 않습니다. 건너뜁니다.")
        return
    print(f"{folder} 파일을 처리하는 중입니다. ")

    with open(data_path, 'rb') as f:
        data_list = pickle.load(f)

    new_data_list = []
    for data in tqdm(data_list):
        if 'bldg_poly_list' not in data:
            continue

        bldg_poly_list = data['bldg_poly_list']
        hierarchical_clustering_list = data['hierarchical_clustering_k_10_debug']
        blk_utm_geometry = data['blk_utm_geometry']

        try:
            angle = get_rotation_angle(list(blk_utm_geometry.exterior.coords))
            blk_utm_geometry = rotate(blk_utm_geometry, -angle, origin='centroid', use_radians=False)
            centroid = blk_utm_geometry.centroid
            blk_utm_geometry, min_coords, range_max = normalize_coords_uniform(blk_utm_geometry.exterior.coords)
            blk_utm_geometry = Polygon(blk_utm_geometry)

            bldg_id2cluster_id = {}
            for idx, hierarchical_clustering in enumerate(hierarchical_clustering_list):
                for bldg_idx in hierarchical_clustering:
                    bldg_id2cluster_id[bldg_idx] = idx

            bldg_id2geometry = {}
            bldg_id2layout = {}
            for bldg_poly_data in bldg_poly_list:
                bldg_id = bldg_poly_data[0]
                bldg_poly = bldg_poly_data[1]

                bldg_coords = [list(coord) for coord in bldg_poly.exterior.coords]
                bldg_utm_coords = convert_to_utm(bldg_coords)

                bldg_poly = rotate(Polygon(bldg_utm_coords), -angle, origin=centroid, use_radians=False)
                bldg_poly, _, _ = normalize_coords_uniform(bldg_poly.exterior.coords, min_coords, range_max)
                bldg_poly = Polygon(bldg_poly)
                bldg_id2geometry[bldg_id] = bldg_poly

                layout = bldg_poly.minimum_rotated_rectangle
                r = get_rotation_angle(list(layout.exterior.coords))
                layout = rotate(layout, -r, origin='centroid', use_radians=False)
                minx, miny, maxx, maxy = layout.bounds
                x = (minx + maxx) / 2
                y = (miny + maxy) / 2
                w = maxx - minx
                h = maxy - miny
                bldg_id2layout[bldg_id] = [x, y, w, h, r]

            cluster_id2geometry_list = {}
            for bldg_id, geometry in bldg_id2geometry.items():
                cluster_id = bldg_id2cluster_id[bldg_id]
                if cluster_id in cluster_id2geometry_list:
                    cluster_id2geometry_list[cluster_id].append(geometry)
                else:
                    cluster_id2geometry_list[cluster_id] = [geometry]

            cluster_id2cluster_geometry = {}
            for cluster_id, geometry_list in cluster_id2geometry_list.items():
                combined_geometry = unary_union(geometry_list)
                min_bbox = combined_geometry.bounds
                bbox_polygon = box(*min_bbox)
                cluster_id2cluster_geometry[cluster_id] = bbox_polygon

            cluster_id2trans_layout_list = {}
            for bldg_id, layout in bldg_id2layout.items():
                bbox_polygon = cluster_id2cluster_geometry[bldg_id2cluster_id[bldg_id]]
                _, min_coords, range_max = normalize_coords_uniform(bbox_polygon.exterior.coords)

                x, y, w, h, r = layout
                layout_poly = create_bounding_box(x, y, w, h, r)
                normalized_layout, _, _ = normalize_coords_uniform(layout_poly.exterior.coords, min_coords, range_max)
                normalized_layout = Polygon(normalized_layout)

                minx, miny, maxx, maxy = normalized_layout.bounds
                x = (minx + maxx) / 2
                y = (miny + maxy) / 2
                w = maxx - minx
                h = maxy - miny

                trans_layout = [x, y, w, h, r / 360, 1]

                cluster_id = bldg_id2cluster_id[bldg_id]
                if cluster_id in cluster_id2trans_layout_list:
                    cluster_id2trans_layout_list[cluster_id].append(trans_layout)
                else:
                    cluster_id2trans_layout_list[cluster_id] = [trans_layout]

            new_data = data
            new_data['bldg_id2geometry'] = bldg_id2geometry
            new_data['bldg_id2layout'] = bldg_id2layout
            new_data['cluster_id2geometry_list'] = cluster_id2geometry_list
            new_data['cluster_id2trans_layout_list'] = cluster_id2trans_layout_list
            new_data['cluster_id2cluster_geometry'] = cluster_id2cluster_geometry

            new_data_list.append(new_data)

        except:
            print("error")
            continue

    with open(output_path, 'wb') as f:
        pickle.dump(new_data_list, f)

    print(f"{folder} 처리 완료")

def main():
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        pool.map(process_folder, subfolders)

if __name__ == "__main__":
    subfolders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
    main()

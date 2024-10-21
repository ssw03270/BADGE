'''
clustering 된 COHO 데이터셋을 preprocessing 하는 단계
이는 이후, 다양한 모델의 입력으로 사용하기 위해 필요함
이 코드의 결과물을 각 모델 별 전처리 코드에 넣어서 사용할 것임
'''
import matplotlib.pyplot as plt
import multiprocessing

import math
import matplotlib.cm as cm
import os
import pickle
import numpy as np
from tqdm import tqdm
from pyproj import Proj, transform
from shapely.affinity import rotate
from shapely.affinity import translate
from shapely.ops import unary_union, split
from shapely.geometry import Polygon, box, LineString, Point, MultiPolygon, GeometryCollection, MultiLineString
from shapely.ops import polygonize, nearest_points
import networkx as nx
from collections import defaultdict
from rtree import index
import copy

def divide_linestring_into_segments(linestring_coords, unit_length):
    """
    Divides a LineString's coordinates into segments of approximately `unit_length`.
    
    Parameters:
    - linestring_coords: List of [x, y] pairs representing the LineString.
    - unit_length: The desired length of each segment.
    
    Returns:
    - new_linestring_coords: List of [x, y] pairs with subdivided segments.
    """
    new_coords = []
    
    for i in range(len(linestring_coords) - 1):
        x1, y1 = linestring_coords[i]
        x2, y2 = linestring_coords[i + 1]

        # Compute the distance between the two points
        distance = np.linalg.norm(np.array([x2, y2]) - np.array([x1, y1]))

        # Determine the number of segments to divide the line into
        num_segments = int(distance // unit_length)

        # Add the starting point of the segment
        new_coords.append([x1, y1])

        if num_segments > 0:
            # Calculate the unit direction vector from point1 to point2
            direction = (np.array([x2, y2]) - np.array([x1, y1])) / distance
            for j in range(1, num_segments):
                # Compute intermediate points at each unit length
                new_point = np.array([x1, y1]) + direction * unit_length * j
                new_coords.append(new_point.tolist())
    
    # Add the final point of the LineString
    new_coords.append(linestring_coords[-1])

    return new_coords

def divide_boundary_into_segments(boundary_coords, unit_length, midaxis_start, midaxis_end):
    new_boundary_coords = []

    for i in range(len(boundary_coords) - 1):
        if i - 1 == -1:
            x0, y0 = boundary_coords[-2]
        else:
            x0, y0 = boundary_coords[i-1]
        x1, y1 = boundary_coords[i]
        x2, y2 = boundary_coords[(i + 1) % len(boundary_coords)]

        # 두 점 사이의 거리 계산
        distance = np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2]))

        # 두 점 사이의 세그먼트 수 계산
        num_segments = int(distance // unit_length)

        if LineString([[x0, y0], [x2, y2]]).distance(Point([x1, y1])) < 1e-6:
            continue

        # 시작점 추가
        new_boundary_coords.append([x1, y1])
        
        if LineString([[x1, y1], [x2, y2]]).distance(Point(midaxis_start)) < 1e-6 or LineString([[x1, y1], [x2, y2]]).distance(Point(midaxis_end)) < 1e-6:
            continue

        # 세그먼트를 단위 길이만큼 추가
        if num_segments > 0:
            vector = np.array([x2 - x1, y2 - y1]) / distance
            for j in range(1, num_segments):
                new_point = np.array([x1, y1]) + vector * unit_length * j
                new_boundary_coords.append(new_point.tolist())

    new_boundary_coords = new_boundary_coords + [new_boundary_coords[0]]
    return new_boundary_coords

def remove_duplicate_lines(linestrings):
    unique_lines = []
    seen = set()

    for line in linestrings:
        # 좌표를 가져옵니다.
        coords = list(line.coords)
        # 좌표를 정렬하여 방향성을 제거합니다.
        sorted_coords = tuple(sorted(coords))
        if sorted_coords not in seen:
            seen.add(sorted_coords)
            unique_lines.append(line)
    return unique_lines

def make_valid(geom):
    """
    주어진 Shapely geometry가 유효한지 확인하고, 유효하지 않으면 buffer(0)을 사용하여 수정합니다.
    
    Parameters:
    - geom: Shapely geometry object
    
    Returns:
    - 유효한 Shapely geometry object
    """
    if not geom.is_valid:
        geom = geom.buffer(0)
    return geom

def normalize_coords_uniform(coords, min_coords=None, range_max=None):
    if min_coords is not None and range_max is not None:
        normalized_coords = (coords - min_coords) / range_max
    else:
        coords = np.array(coords)
        min_coords = coords.min(axis=0)
        max_coords = coords.max(axis=0)
        range_max = (max_coords - min_coords).max()
        normalized_coords = (coords - min_coords) / range_max
    
    out_of_bounds = []
    # 정규화된 좌표가 0과 1 사이에 있는지 확인
    if not (np.all(normalized_coords >= -1) and np.all(normalized_coords <= 2)):
        print("경고: 정규화된 좌표 중 일부가 0과 1 사이에 있지 않습니다.")
        # 추가로, 어떤 좌표가 범위를 벗어났는지 출력할 수 있습니다.
        out_of_bounds = normalized_coords[(normalized_coords < -1) | (normalized_coords > 2)]
        print("범위를 벗어난 좌표 값:", out_of_bounds)
    
    return normalized_coords, min_coords, range_max, out_of_bounds

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

dataset_path = "Z:/iiixr-drive/Projects/2023_City_Team/000_2024CVPR/COHO_dataset"
output_dataset_path = "Z:/iiixr-drive/Projects/2023_City_Team/000_2024CVPR/Our_dataset"
subfolders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]

debug = False
def process_folder(folder):
    error_count = 0
    output_path = os.path.join(output_dataset_path, folder, f'preprocessed/{folder}_graph_prep_list_with_clusters_detail.pkl')
    # if os.path.exists(output_path):
    #     print(f"{output_path} 파일이 이미 존재합니다. 건너뜁니다.")
    #     return  # 이미 처리된 경우 건너뜁니다.
    
    os.makedirs(os.path.join(output_dataset_path, folder, f'preprocessed/'), exist_ok=True)  # output 디렉토리 생성

    data_path = os.path.join(dataset_path, folder, f'graph/{folder}_graph_prep_list_with_clusters.pkl')
    if not os.path.exists(data_path):
        print(f"{data_path} 파일이 존재하지 않습니다. 건너뜁니다.")
        return
    print(f"{folder} 파일을 처리하는 중입니다. ")

    with open(data_path, 'rb') as f:
        data_list = pickle.load(f)

    new_data_list = []
    for data in tqdm(data_list):
        try:
            is_stop = False

            hierarchical_clustering_list = data['hierarchical_clustering_k_10_debug']
            # hierarchical_clustering_list = data['RoadNetwork_k_10']
            blk_recover_poly = data['blk_recover_poly']
            bldg_poly_list = data['bldg_poly_list']
            midaxis = data['medaxis']

            minx, miny, maxx, maxy = blk_recover_poly.bounds
            midaxis_start = Point(midaxis.coords[0])
            midaxis_end = Point(midaxis.coords[-1])
            blk_polygon = Polygon(divide_boundary_into_segments(blk_recover_poly.exterior.coords, abs(maxx - minx) // 10, midaxis_start=midaxis_start, midaxis_end=midaxis_end))
            midaxis = LineString(divide_linestring_into_segments(midaxis.coords, abs(maxx - minx) // 10))

            blk_centroid = blk_polygon.centroid
            blk_angle = get_rotation_angle(list(blk_polygon.exterior.coords))
            rotated_blk = rotate(blk_polygon, -blk_angle, origin=blk_centroid, use_radians=False)
            normalized_blk, min_coords, range_max, out_of_bounds = normalize_coords_uniform(rotated_blk.exterior.coords)
            normalized_blk = Polygon(normalized_blk)

            if len(out_of_bounds) > 0:
                is_stop = True

            normalized_bldg_list_blk = []
            for bldg_poly in bldg_poly_list:
                bldg_id = bldg_poly[0]
                rotated_bldg = rotate(bldg_poly[5], -blk_angle, origin=blk_centroid, use_radians=False)
                normalized_bldg, _, _, out_of_bounds = normalize_coords_uniform(rotated_bldg.exterior.coords, min_coords, range_max)
                normalized_bldg = Polygon(normalized_bldg)
                normalized_bldg_list_blk.append([bldg_id, normalized_bldg])

                if len(out_of_bounds) > 0:
                    is_stop = True

            rotated_midaxis = rotate(midaxis, -blk_angle, origin=blk_centroid, use_radians=False)
            normalized_midaxis, _, _, out_of_bounds = normalize_coords_uniform(rotated_midaxis.coords, min_coords, range_max)
            normalized_midaxis = LineString(normalized_midaxis)

            if len(out_of_bounds) > 0:
                is_stop = True

            blk_polygon = normalized_blk
            bldg_poly_list = normalized_bldg_list_blk
            midaxis = normalized_midaxis
            midaxis_start = Point(midaxis.coords[0])
            midaxis_end = Point(midaxis.coords[-1])

            bldg_id2normalized_bldg_poly_blk = {}
            bldg_id2normalized_bldg_layout_blk = {}
            for bldg in bldg_poly_list:
                bldg_id = bldg[0]
                bldg_poly = bldg[1]
                bldg_id2normalized_bldg_poly_blk[bldg_id] = bldg_poly

                layout = bldg_poly.minimum_rotated_rectangle
                r = get_rotation_angle(list(layout.exterior.coords))
                layout = rotate(layout, -r, origin='centroid', use_radians=False)
                minx, miny, maxx, maxy = layout.bounds
                x = (minx + maxx) / 2
                y = (miny + maxy) / 2
                w = maxx - minx
                h = maxy - miny
                bldg_id2normalized_bldg_layout_blk[bldg_id] = [x, y, w, h, r]
            
            bldg_id2cluster_id = {}
            cluster_id2bldg_id_list = {}
            for cluster_id, hierarchical_clustering in enumerate(hierarchical_clustering_list):
                for bldg_id in hierarchical_clustering:
                    bldg_id2cluster_id[bldg_id] = cluster_id + 1
                    if cluster_id + 1 in cluster_id2bldg_id_list:
                        cluster_id2bldg_id_list[cluster_id + 1].append(bldg_id)
                    else:
                        cluster_id2bldg_id_list[cluster_id + 1] = [bldg_id]

            connecting_lines_blk_to_med = []
            for blk_point in blk_polygon.exterior.coords:
                blk_pt = Point(blk_point)
                nearest_pt = min(midaxis.coords, key=lambda med_pt: blk_pt.distance(Point(med_pt)))
                line = LineString([blk_pt, Point(nearest_pt)])
                connecting_lines_blk_to_med.append(line)
            
            blk_lines = []
            blk_coords_list = list(blk_polygon.exterior.coords)
            for i in range(len(blk_coords_list) - 1):
                line = LineString([blk_coords_list[i], blk_coords_list[i+1]])
                blk_lines.append(line)
            
            removed_blk_lines = []
            min_blk_line = min(blk_lines, key=lambda blk_line: blk_line.distance(midaxis_start))
            blk_lines.remove(min_blk_line)
            removed_blk_lines.append(LineString([min_blk_line.coords[0], midaxis_start]))
            removed_blk_lines.append(LineString([min_blk_line.coords[1], midaxis_start]))
            connecting_lines_blk_to_med.append(LineString([min_blk_line.coords[0], midaxis_start]))
            connecting_lines_blk_to_med.append(LineString([min_blk_line.coords[1], midaxis_start]))
            min_blk_line = min(blk_lines, key=lambda blk_line: blk_line.distance(midaxis_end))
            blk_lines.remove(min_blk_line)
            removed_blk_lines.append(LineString([min_blk_line.coords[0], midaxis_end]))
            removed_blk_lines.append(LineString([min_blk_line.coords[1], midaxis_end]))
            connecting_lines_blk_to_med.append(LineString([min_blk_line.coords[0], midaxis_end]))
            connecting_lines_blk_to_med.append(LineString([min_blk_line.coords[1], midaxis_end]))

            medaxis_lines = []
            medaxis_coords_list = list(midaxis.coords)
            for i in range(len(medaxis_coords_list) - 1):
                line = LineString([medaxis_coords_list[i+1],medaxis_coords_list[i]])
                medaxis_lines.append(line)

            blk_lines = remove_duplicate_lines(blk_lines)
            medaxis_lines = remove_duplicate_lines(medaxis_lines)
            connecting_lines_blk_to_med = remove_duplicate_lines(connecting_lines_blk_to_med)
            region_polygons = list(polygonize(medaxis_lines + blk_lines + connecting_lines_blk_to_med))

            region_id2region_polygon = {}
            for region_id, region_poly in enumerate(region_polygons):
                region_id2region_polygon[region_id] = region_poly
                            

            union_polygons = unary_union(region_polygons)
            intersection = blk_polygon.intersection(union_polygons)
            intersection_area = intersection.area
            area_a = blk_polygon.area

            # 비율 계산
            ratio = intersection_area / area_a

            if ratio < 0.99 or ratio > 1.01:
                continue
            
            region_id2cluster_id_list = {}
            cluster_id2region_id_list = {}
            for region_id, region_poly in region_id2region_polygon.items():
                for cluster_id, bldg_id_list in cluster_id2bldg_id_list.items():
                    for bldg_id in bldg_id_list:
                        bldg_poly = bldg_id2normalized_bldg_poly_blk[bldg_id]
                        if region_poly.intersection(bldg_poly):
                            if cluster_id in cluster_id2region_id_list:
                                if not region_id in cluster_id2region_id_list[cluster_id]:
                                    cluster_id2region_id_list[cluster_id].append(region_id)
                            else:
                                cluster_id2region_id_list[cluster_id] = [region_id]
                            if region_id in region_id2cluster_id_list:
                                if not cluster_id in region_id2cluster_id_list[region_id]:
                                    region_id2cluster_id_list[region_id].append(cluster_id)
                            else:
                                region_id2cluster_id_list[region_id] = [cluster_id]
                if region_id not in region_id2cluster_id_list:
                    region_id2cluster_id_list[region_id] = [0]
                    if 0 in cluster_id2region_id_list:
                        cluster_id2region_id_list[0].append(region_id)
                    else:
                        cluster_id2region_id_list[0] = [region_id]
                        
            cluster_id2cluster_bbox = {}
            for cluster_id, region_id_list in cluster_id2region_id_list.items():
                region_poly_list = []
                for region_id in region_id_list:
                    region_poly = region_id2region_polygon[region_id]
                    region_poly_list.append(region_poly)
                
                combined_region = unary_union(region_poly_list)
                min_bbox = combined_region.bounds
                bbox_poly = box(*min_bbox)
                cluster_id2cluster_bbox[cluster_id] = bbox_poly

            bldg_id2normalized_bldg_poly_cluster = {}
            bldg_id2normalized_bldg_layout_cluster = {}
            for cluster_id, _bldg_id_list in cluster_id2bldg_id_list.items():
                cluster_bbox = cluster_id2cluster_bbox[cluster_id]
                for bldg_id in _bldg_id_list:
                    normalized_bldg_poly_blk = bldg_id2normalized_bldg_poly_blk[bldg_id]
                    normalized_bldg_poly_layout = bldg_id2normalized_bldg_layout_blk[bldg_id]

                    _, min_coords, range_max, out_of_bounds = normalize_coords_uniform(cluster_bbox.exterior.coords)

                    if len(out_of_bounds) > 0:
                        is_stop = True

                    normalized_bldg_poly_cluster, _, _, out_of_bounds = normalize_coords_uniform(normalized_bldg_poly_blk.exterior.coords, min_coords, range_max)
                    normalized_bldg_poly_cluster = Polygon(normalized_bldg_poly_cluster)

                    if len(out_of_bounds) > 0:
                        is_stop = True

                    x, y, w, h, r = normalized_bldg_poly_layout
                    layout_poly = create_bounding_box(x, y, w, h, r)
                    normalized_layout, _, _, out_of_bounds = normalize_coords_uniform(layout_poly.exterior.coords, min_coords, range_max)
                    normalized_layout = Polygon(normalized_layout)

                    if len(out_of_bounds) > 0:
                        is_stop = True

                    r = get_rotation_angle(list(normalized_layout.exterior.coords))
                    normalized_layout = rotate(normalized_layout, -r, origin='centroid', use_radians=False)

                    minx, miny, maxx, maxy = normalized_layout.bounds
                    x = (minx + maxx) / 2
                    y = (miny + maxy) / 2
                    w = maxx - minx
                    h = maxy - miny

                    normalized_bldg_layout_cluster = [x, y, w, h, r]

                    bldg_id2normalized_bldg_poly_cluster[bldg_id] = normalized_bldg_poly_cluster
                    bldg_id2normalized_bldg_layout_cluster[bldg_id] = normalized_bldg_layout_cluster

            cluster_id2face_blk_linestring_list = {}
            for cluster_id, region_id_list in cluster_id2region_id_list.items():
                for region_id in region_id_list:
                    region_poly = region_id2region_polygon[region_id]
                    exterior_coords = list(region_poly.exterior.coords)

                    edges = set()
                    for i in range(len(exterior_coords) - 1):
                        point1 = exterior_coords[i]
                        point2 = exterior_coords[i + 1]
                        edge = frozenset([point1, point2])
                        edges.add(edge)
                        
                    for blk_line in blk_lines:
                        blk_points = list(blk_line.coords)
                        blk_edge = frozenset([blk_points[0], blk_points[1]])
                        if blk_edge in edges:
                            if cluster_id in cluster_id2face_blk_linestring_list:
                                cluster_id2face_blk_linestring_list[cluster_id].append(blk_line)
                            else:
                                cluster_id2face_blk_linestring_list[cluster_id] = [blk_line]
                                
                    for blk_line in removed_blk_lines:
                        blk_points = list(blk_line.coords)
                        blk_edge = frozenset([blk_points[0], blk_points[1]])
                        if blk_edge in edges:
                            if cluster_id in cluster_id2face_blk_linestring_list:
                                cluster_id2face_blk_linestring_list[cluster_id].append(blk_line)
                            else:
                                cluster_id2face_blk_linestring_list[cluster_id] = [blk_line]

            if is_stop:
                continue

            new_data = data
            new_data['bldg_id2cluster_id'] = bldg_id2cluster_id
            new_data['bldg_id2normalized_bldg_poly_blk'] = bldg_id2normalized_bldg_poly_blk
            new_data['bldg_id2normalized_bldg_poly_cluster'] = bldg_id2normalized_bldg_poly_cluster
            new_data['bldg_id2normalized_bldg_layout_blk'] = bldg_id2normalized_bldg_layout_blk
            new_data['bldg_id2normalized_bldg_layout_cluster'] = bldg_id2normalized_bldg_layout_cluster
            new_data['cluster_id2bldg_id_list'] = cluster_id2bldg_id_list
            new_data['cluster_id2cluster_bbox'] = cluster_id2cluster_bbox
            new_data['cluster_id2region_id_list'] = cluster_id2region_id_list
            new_data['cluster_id2face_blk_linestring_list'] = cluster_id2face_blk_linestring_list
            new_data['region_id2cluster_id_list'] = region_id2cluster_id_list
            new_data['region_id2region_polygon'] = region_id2region_polygon
            new_data_list.append(new_data)

            if debug:
                fig, axs = plt.subplots(2, 2, figsize=(14, 7))
                for line in medaxis_lines:
                    x, y = line.coords.xy
                    axs[0][0].plot(x, y)
                for line in blk_lines:
                    x, y = line.coords.xy
                    axs[0][0].plot(x, y)
                for line in connecting_lines_blk_to_med:
                    x, y = line.coords.xy
                    axs[0][0].plot(x, y)

                colors = cm.get_cmap('tab20')(np.linspace(0, 1, len(hierarchical_clustering_list) + 1))
                for region_id, cluster_id_list in region_id2cluster_id_list.items():
                    region_poly = region_id2region_polygon[region_id]
                    x, y = region_poly.exterior.coords.xy
                    for cluster_id in cluster_id_list:
                        axs[0][1].plot(x, y, color=colors[cluster_id])
                        axs[0][1].fill(x, y, alpha=0.5, color=colors[cluster_id])

                for cluster_id, _bldg_id_list in cluster_id2bldg_id_list.items():
                    if len(_bldg_id_list) > 0:
                        for bldg_id in _bldg_id_list:
                            normalized_bldg_poly_blk = bldg_id2normalized_bldg_poly_blk[bldg_id]
                            x, y = normalized_bldg_poly_blk.exterior.coords.xy
                            axs[0][1].plot(x, y, color=colors[cluster_id])

                for cluster_id, cluster_bbox in cluster_id2cluster_bbox.items():
                    x, y = cluster_bbox.exterior.coords.xy
                    axs[0][1].plot(x, y, color=colors[cluster_id])

                for cluster_id, face_blk_linestring_list in cluster_id2face_blk_linestring_list.items():
                    for face_blk_linestring in face_blk_linestring_list:
                        x, y = face_blk_linestring.coords.xy
                        axs[1][0].plot(x, y, color=colors[cluster_id])

                for cluster_id, _bldg_id_list in cluster_id2bldg_id_list.items():
                    if len(_bldg_id_list) > 0:
                        for bldg_id in _bldg_id_list:
                            normalized_bldg_poly_cluster = bldg_id2normalized_bldg_poly_cluster[bldg_id]
                            x, y = normalized_bldg_poly_cluster.exterior.coords.xy
                            axs[1][1].plot(x, y, color=colors[cluster_id])

                            normalized_bldg_layout_cluster = bldg_id2normalized_bldg_layout_cluster[bldg_id]
                            x, y, w, h, r = normalized_bldg_layout_cluster
                            layout_poly = create_bounding_box(x, y, w, h, r)
                            x, y = layout_poly.exterior.coords.xy
                            axs[1][1].plot(x, y, color=colors[cluster_id])

                axs[0][0].legend()
                axs[0][0].set_aspect('equal')
                axs[0][1].legend()
                axs[0][1].set_aspect('equal')
                axs[1][0].legend()
                axs[1][0].set_aspect('equal')
                axs[1][1].legend()
                axs[1][1].set_aspect('equal')

                plt.show()
        except:
            error_count += 1
            continue
    with open(output_path, 'wb') as f:
        pickle.dump(new_data_list, f)

        print(f"{folder}, error: {error_count}")
        print(f"{folder} 처리 완료")
           
def main():
    with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
        pool.map(process_folder, subfolders)

if __name__ == "__main__":
    subfolders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
    main()
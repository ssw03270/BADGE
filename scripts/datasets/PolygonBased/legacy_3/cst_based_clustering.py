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

def remove_larger_overlapping_polygons(polygons):
    """
    주어진 폴리곤 리스트에서 겹치는 폴리곤 중 더 큰 폴리곤을 제거하고, 크기가 작은 폴리곤만 남깁니다.
    
    Parameters:
    - polygons: Polygon 객체들의 리스트
    
    Returns:
    - 크기가 작은 폴리곤만 남은 새로운 리스트
    """
    # 남길 폴리곤을 저장할 리스트
    filtered_polygons = polygons.copy()
    
    # 모든 폴리곤 쌍을 비교
    for i, poly_a in enumerate(polygons):
        for j, poly_b in enumerate(polygons):
            # 동일한 폴리곤은 건너뜀
            if i >= j:
                continue
            
            # 폴리곤이 겹치는지 확인 (면을 공유하는 것은 제외)
            if poly_a.intersects(poly_b) and not poly_a.touches(poly_b):
                # 두 폴리곤의 면적 비교
                if poly_a.area < poly_b.area:
                    # poly_b 제거
                    if poly_b in filtered_polygons:
                        filtered_polygons.remove(poly_b)
                else:
                    # poly_a 제거
                    if poly_a in filtered_polygons:
                        filtered_polygons.remove(poly_a)
    
    return filtered_polygons

def merge_polygons_by_cluster(polygons, cluster_indices):
    """
    같은 클러스터 인덱스를 가진 겹치는 폴리곤을 병합합니다.

    Parameters:
    - polygons: List of Shapely Polygon objects.
    - cluster_indices: List of integers representing cluster indices.

    Returns:
    - merged_polygons: List of merged Shapely Polygon objects.
    - merged_clusters: List of cluster indices corresponding to merged_polygons.
    """
    cluster_to_polygons = defaultdict(list)
    for poly, cluster_idx in zip(polygons, cluster_indices):
        cluster_to_polygons[cluster_idx].append(poly)
    
    merged_polygons = []
    merged_clusters = []
    
    for cluster_idx, poly_list in cluster_to_polygons.items():
        if len(poly_list) == 1:
            merged_polygons.append(poly_list[0])
            merged_clusters.append(cluster_idx)
        else:
            merged = unary_union(poly_list)
            # merged가 Polygon, MultiPolygon 또는 GeometryCollection인지 확인
            if isinstance(merged, Polygon):
                merged_polygons.append(merged)
                merged_clusters.append(cluster_idx)
            elif isinstance(merged, MultiPolygon):
                for p in merged.geoms:
                    merged_polygons.append(p)
                    merged_clusters.append(cluster_idx)
            elif isinstance(merged, GeometryCollection):
                for geom in merged.geoms:
                    if isinstance(geom, Polygon):
                        merged_polygons.append(geom)
                        merged_clusters.append(cluster_idx)
            else:
                print(f"Unexpected geometry type after union: {merged.geom_type}")
    
    return merged_polygons, merged_clusters

def adjust_polygons_to_remove_overlap(merged_polygons, merged_clusters):
    """
    다른 클러스터 인덱스를 가진 겹치는 폴리곤 간의 겹침을 제거합니다.
    규칙: 낮은 클러스터 인덱스를 가진 폴리곤이 우선권을 가지며, 높은 클러스터 인덱스를 가진 폴리곤에서 겹치는 부분을 제거합니다.

    Parameters:
    - merged_polygons: List of Shapely Polygon objects after cluster-wise merging.
    - merged_clusters: List of cluster indices corresponding to merged_polygons.

    Returns:
    - final_polygons: List of Shapely Polygon objects with overlaps removed.
    - final_clusters: List of cluster indices corresponding to final_polygons.
    """
    # R-tree 인덱스 생성
    rtree_idx = index.Index()
    for i, poly in enumerate(merged_polygons):
        rtree_idx.insert(i, poly.bounds)
    
    # 폴리곤과 클러스터 인덱스를 복사하여 수정
    final_polygons = copy.deepcopy(merged_polygons)
    final_clusters = copy.deepcopy(merged_clusters)
    
    # 클러스터 인덱스가 낮은 순서대로 정렬
    sorted_indices = sorted(range(len(final_polygons)), key=lambda x: final_clusters[x])
    
    for i in sorted_indices:
        poly_i = final_polygons[i]
        cluster_i = final_clusters[i]
        if poly_i.is_empty:
            continue
        # 현재 폴리곤과 겹칠 수 있는 후보 찾기
        possible_matches = list(rtree_idx.intersection(poly_i.bounds))
        for j in possible_matches:
            if i == j:
                continue  # 자기 자신은 제외
            poly_j = final_polygons[j]
            cluster_j = final_clusters[j]
            if cluster_i == cluster_j:
                continue  # 같은 클러스터는 이미 병합됨
            if poly_i.intersects(poly_j):
                intersection = poly_i.intersection(poly_j)
                if not intersection.is_empty:
                    # 낮은 클러스터 인덱스가 우선권을 가짐
                    if cluster_i < cluster_j:
                        # 클러스터 j에서 겹치는 부분을 제거
                        final_polygons[j] = poly_j.difference(intersection)
                    else:
                        # 클러스터 i에서 겹치는 부분을 제거
                        final_polygons[i] = poly_i.difference(intersection)
    
    # 유효하지 않은 폴리곤 수정 및 MultiPolygon 분해
    cleaned_polygons = []
    cleaned_clusters = []
    
    for poly, cluster_idx in zip(final_polygons, final_clusters):
        if poly.is_empty:
            continue
        if isinstance(poly, MultiPolygon):
            for p in poly.geoms:
                if p.is_valid:
                    cleaned_polygons.append(p)
                    cleaned_clusters.append(cluster_idx)
                else:
                    # 간단한 정정을 시도
                    p_fixed = p.buffer(0)
                    if p_fixed.is_valid and isinstance(p_fixed, Polygon):
                        cleaned_polygons.append(p_fixed)
                        cleaned_clusters.append(cluster_idx)
        elif isinstance(poly, Polygon):
            if poly.is_valid:
                cleaned_polygons.append(poly)
                cleaned_clusters.append(cluster_idx)
            else:
                poly_fixed = poly.buffer(0)
                if poly_fixed.is_valid and isinstance(poly_fixed, Polygon):
                    cleaned_polygons.append(poly_fixed)
                    cleaned_clusters.append(cluster_idx)
        elif isinstance(poly, GeometryCollection):
            for geom in poly.geoms:
                if isinstance(geom, Polygon):
                    if geom.is_valid:
                        cleaned_polygons.append(geom)
                        cleaned_clusters.append(cluster_idx)
                    else:
                        # 간단한 정정을 시도
                        geom_fixed = geom.buffer(0)
                        if geom_fixed.is_valid and isinstance(geom_fixed, Polygon):
                            cleaned_polygons.append(geom_fixed)
                            cleaned_clusters.append(cluster_idx)
        else:
            print(f"Unexpected geometry type: {poly.geom_type}")
    
    return cleaned_polygons, cleaned_clusters
from canonical_transform import warp_bldg_by_midaxis

# UTM 좌표계 정의 (여기서는 예시로 UTM Zone 14N을 사용)
utm_proj = Proj(proj='utm', zone=14, ellps='WGS84', units='m')
# WGS84 좌표계 정의
wgs84_proj = Proj(proj='latlong', datum='WGS84')

def convert_to_utm(coords):
    utm_coords = []
    for lon, lat in coords:
        x, y = utm_proj(lon, lat)  # 위도 경도를 UTM으로 변환
        utm_coords.append([x, y])
    return utm_coords

def convert_to_wgs84(coords):
    wgs84_coords = []  # 변수 이름 변경
    for x, y in coords:  # UTM 좌표를 사용하도록 변경
        lon, lat = transform(utm_proj, wgs84_proj, x, y)
        wgs84_coords.append([lon, lat])  # 변수 이름 변경
    return wgs84_coords  # 변수 이름 변경

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

dataset_path = "Z:/iiixr-drive/Projects/2023_City_Team/000_2024CVPR/COHO_dataset"
subfolders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]

debug = True
def process_folder(folder):
    output_path = os.path.join(dataset_path, folder, f'graph/{folder}_graph_prep_list_with_clusters_detail_region.pkl')
    if os.path.exists(output_path):
        print(f"{output_path} 파일이 이미 존재합니다. 건너뜁니다.")
        return  # 이미 처리된 경우 건너뜁니다.
    
    data_path = os.path.join(dataset_path, folder, f'graph/{folder}_graph_prep_list_with_clusters_detail.pkl')
    if not os.path.exists(data_path):
        print(f"{data_path} 파일이 존재하지 않습니다. 건너뜁니다.")
        return
    print(f"{folder} 파일을 처리하는 중입니다. ")

    with open(data_path, 'rb') as f:
        data_list = pickle.load(f)

    new_data_list = []
    for data in tqdm(data_list):
        hierarchical_clustering_list = data['hierarchical_clustering_k_10_debug']
        # hierarchical_clustering_list = data['RoadNetwork_k_10']
        blk_recover_poly = data['blk_recover_poly']
        bldg_poly_list = data['bldg_poly_list']
        minx, miny, maxx, maxy = blk_recover_poly.bounds
        midaxist_start = Point(data['medaxis'].coords[0])
        midaxist_end = Point(data['medaxis'].coords[-1])
        blk_polygon = Polygon(divide_boundary_into_segments(blk_recover_poly.exterior.coords, abs(maxx - minx) // 10, midaxis_start=midaxist_start, midaxis_end=midaxist_end))
        medaxis_line = LineString(divide_linestring_into_segments(data['medaxis'].coords, abs(maxx - minx) // 10))
        try:
            bldg_dict = {}
            for bldg in bldg_poly_list:
                bldg_id = bldg[0]
                bldg_poly = bldg[5]
                bldg_dict[bldg_id] = bldg_poly

            bldg_id2cluster_id = {}
            for idx, hierarchical_clustering in enumerate(hierarchical_clustering_list):
                for bldg_idx in hierarchical_clustering:
                    bldg_id2cluster_id[bldg_idx] = idx

            connecting_lines_blk_to_med = []
            for blk_point in blk_polygon.exterior.coords:
                blk_pt = Point(blk_point)
                nearest_pt = min(medaxis_line.coords, key=lambda med_pt: blk_pt.distance(Point(med_pt)))
                line = LineString([blk_pt, Point(nearest_pt)])
                connecting_lines_blk_to_med.append(line)
            
            blk_lines = []
            blk_coords_list = list(blk_polygon.exterior.coords)
            for i in range(len(blk_coords_list) - 1):
                line = LineString([blk_coords_list[i], blk_coords_list[i+1]])
                blk_lines.append(line)
            
            min_blk_line = min(blk_lines, key=lambda blk_line: blk_line.distance(midaxist_start))
            blk_lines.remove(min_blk_line)
            connecting_lines_blk_to_med.append(LineString([min_blk_line.coords[0], midaxist_start]))
            connecting_lines_blk_to_med.append(LineString([min_blk_line.coords[1], midaxist_start]))
            min_blk_line = min(blk_lines, key=lambda blk_line: blk_line.distance(midaxist_end))
            blk_lines.remove(min_blk_line)
            connecting_lines_blk_to_med.append(LineString([min_blk_line.coords[0], midaxist_end]))
            connecting_lines_blk_to_med.append(LineString([min_blk_line.coords[1], midaxist_end]))

            medaxis_lines = []
            medaxis_coords_list = list(medaxis_line.coords)
            for i in range(len(medaxis_coords_list) - 1):
                line = LineString([medaxis_coords_list[i+1],medaxis_coords_list[i]])
                medaxis_lines.append(line)

            blk_lines = remove_duplicate_lines(blk_lines)
            medaxis_lines = remove_duplicate_lines(medaxis_lines)
            connecting_lines_blk_to_med = remove_duplicate_lines(connecting_lines_blk_to_med)
            region_polygons = list(polygonize(medaxis_lines + blk_lines + connecting_lines_blk_to_med))

            bldg_multi_polygon_list = []
            for cluster in hierarchical_clustering_list:
                bldg_multi_polygon = []
                for bldg_id in cluster:
                    bldg_multi_polygon.append(bldg_dict[bldg_id])
                bldg_multi_polygon_list.append(MultiPolygon(bldg_multi_polygon))

            max_cluster_indices = []
            for region in region_polygons:
                overlaps = []
                fixed_region = make_valid(region)
                for bldg_multi in bldg_multi_polygon_list:
                    # MultiPolygon의 각 폴리곤과 교차 여부 확인
                    overlap_area = 0
                    for bldg in bldg_multi.geoms:
                        fixed_bldg = make_valid(bldg)
                        try:
                            intersection = fixed_region.intersection(fixed_bldg)
                            overlap_area += intersection.area
                        except Exception as e:
                            print(f"Intersection 오류 발생: {e}")
                            continue
                    overlaps.append(overlap_area)
                if overlaps:
                    max_overlap = max(overlaps)
                    if max_overlap > 0:
                        max_cluster_index = overlaps.index(max_overlap) + 1
                    else:
                        max_cluster_index = 0
                else:
                    max_cluster_index = 0
                max_cluster_indices.append(max_cluster_index)

            union_polygons = unary_union(region_polygons)
            intersection = blk_polygon.intersection(union_polygons)
            intersection_area = intersection.area
            area_a = blk_polygon.area

            # 비율 계산
            ratio = intersection_area / area_a

            if ratio < 0.99 or ratio > 1.01 or len(region_polygons) != len(max_cluster_indices):
                continue

            new_data = data
            new_data['cluster_indices'] = max_cluster_indices
            new_data['region_polygons'] = region_polygons
            new_data['bldg_id2cluster_id'] = bldg_id2cluster_id
            new_data_list.append(new_data)

            if debug:
                fig, axs = plt.subplots(1, 2, figsize=(14, 7))
                for line in medaxis_lines:
                    x, y = line.coords.xy
                    axs[0].plot(x, y)
                for line in blk_lines:
                    x, y = line.coords.xy
                    axs[0].plot(x, y)
                for line in connecting_lines_blk_to_med:
                    x, y = line.coords.xy
                    axs[0].plot(x, y)

                colors = cm.get_cmap('tab20')(np.linspace(0, 1, len(hierarchical_clustering_list) + 1))
                # 생성된 폴리곤 그리기
                for poly, max_cluster_index in zip(region_polygons, max_cluster_indices):
                    x, y = poly.exterior.coords.xy
                    # 선 그리기`
                    axs[1].plot(x, y, color=colors[max_cluster_index])
                    # 면 채우기
                    axs[1].fill(x, y, alpha=0.5, color=colors[max_cluster_index])
                    # axs[0].text(poly.centroid.x, poly.centroid.y, str(max_cluster_index), fontsize=12, color='black')

                colors = cm.get_cmap('tab10')(np.linspace(0, 1, len(hierarchical_clustering_list)))
                for idx, poly in enumerate(bldg_poly_list):
                    cluster_id = bldg_id2cluster_id[poly[0]]
                    x, y = poly[5].exterior.coords.xy
                    # 면 채우기
                    axs[1].fill(x, y, alpha=0.5, color=colors[cluster_id])

                axs[0].legend()
                axs[0].set_aspect('equal')
                axs[1].legend()
                axs[1].set_aspect('equal')

                plt.show()
        except:
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
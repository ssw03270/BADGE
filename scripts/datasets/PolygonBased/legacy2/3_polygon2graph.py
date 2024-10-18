import os
import pickle
from shapely.geometry import Polygon, Point, LineString
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from shapely.ops import unary_union, nearest_points
import matplotlib.cm as cm
import math

city_list = [
    "Atlanta", "Boston", "Dallas", "Denver", "Houston", "Lasvegas",
    "Littlerock", "Miami", "NewOrleans", "Philadelphia", "Phoenix",
    "Pittsburgh", "Portland", "Providence", "Richmond", "Saintpaul",
    "Sanfrancisco", "Seattle", "Washington"
]

UNIT_LENGTH = 10

def rotated_90(p1, p2, unit_length):
    p1 = np.array(p1)
    p2 = np.array(p2)

    v = p2 - p1
    R = np.array([[0, 1], [-1, 0]])
    v_rotated = np.dot(R, v)
    magnitude_v = np.linalg.norm(v_rotated)
    v_normalized = v_rotated / magnitude_v

    v_rotated = [(p1 + p2) / 2, (p1 + p2) / 2 + v_normalized * unit_length]
    return np.array(v_rotated)

def remove_consecutive_duplicates(coordinate_list):
    cleaned_list = []

    for i, coord in enumerate(coordinate_list):
        # 첫 번째 좌표이거나 이전 좌표와 다르면 리스트에 추가
        if coord != coordinate_list[(i + 1)% len(coordinate_list)]:
            cleaned_list.append(coord)

    return cleaned_list

def remove_close_points_numpy(coordinate_list, threshold_distance):
    cleaned_list = []
    temp_group = []

    for i, coord in enumerate(coordinate_list):
        if i == 0:
            temp_group.append(coord)
        else:
            previous_coord = np.array(coordinate_list[i - 1])
            current_coord = np.array(coord)
            distance = np.linalg.norm(current_coord - previous_coord)

            # threshold_distance보다 작으면 같은 그룹에 넣음
            if distance <= threshold_distance:
                temp_group.append(coord)
            else:
                # 이전 그룹에서 중간 좌표만 추가
                if len(temp_group) > 0:
                    mid_index = len(temp_group) // 2
                    cleaned_list.append(temp_group[mid_index])
                temp_group = [coord]  # 새 그룹 시작

    # 마지막 그룹 처리
    if len(temp_group) > 0:
        mid_index = len(temp_group) // 2
        cleaned_list.append(temp_group[mid_index])

    return cleaned_list
def visualize_boundary_with_buildings(boundary_segments, building_polygons, edge_index):
    boundary_segments = boundary_segments + [boundary_segments[0]]
    boundary_segments = np.array(boundary_segments)
    plt.plot(boundary_segments[:, 0], boundary_segments[:, 1], marker='o', color='red', alpha=0.5)

    for i, poly in enumerate(building_polygons):
        if isinstance(poly, Polygon):  # 만약 `poly`가 `Polygon` 객체라면
            x, y = poly.exterior.xy  # 외곽선의 좌표를 추출
            plt.fill(x, y, alpha=0.5, fc='blue', ec='black')  # 폴리곤을 시각화
            plt.text(poly.centroid.x, poly.centroid.y, str(i), fontsize=6, ha='center', va='center', color='black')

    boundary_count = len(boundary_segments) - 1
    for edge in edge_index:
        idx1, idx2 = edge

        if idx1 < boundary_count:
            x1, y1 = boundary_segments[idx1]
        else:
            x1 = building_polygons[idx1 - boundary_count].centroid.x
            y1 = building_polygons[idx1 - boundary_count].centroid.y

        if idx2 < boundary_count:
            x2, y2 = boundary_segments[idx2]
        else:
            x2 = building_polygons[idx2 - boundary_count].centroid.x
            y2 = building_polygons[idx2 - boundary_count].centroid.y

        plt.fill([x1, x2], [y1, y2], alpha=0.5, fc='blue', ec='red')

    plt.axis('square')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.show()

def get_bbox_details(rotated_rectangle):
    # 사각형의 꼭짓점들을 얻음
    x, y = rotated_rectangle.exterior.coords.xy

    min_angle = 999
    min_idx = -1
    for i in range(len(x) - 1):
        dx = x[(i+1) % (len(x)-1)] - x[i]
        dy = y[(i+1) % (len(y)-1)] - y[i]
        angle = math.degrees(math.atan2(dy, dx))
        if min_angle > abs(angle):
            min_angle = abs(angle)
            min_idx = i

    idx = min_idx
    w = math.sqrt((x[(idx + 1) % (len(y)-1)] - x[idx]) ** 2 + (y[(idx + 1) % (len(y)-1)] - y[idx]) ** 2)
    h = math.sqrt((x[(idx + 2) % (len(y)-1)] - x[(idx + 1) % (len(y)-1)]) ** 2 + (y[(idx + 2) % (len(y)-1)] - y[(idx + 1) % (len(y)-1)]) ** 2)

    dx = x[idx + 1] - x[idx]
    dy = y[idx + 1] - y[idx]
    theta = math.degrees(math.atan2(dy, dx))
    theta = (theta + 45) / 90

    # 좌하단 꼭짓점을 x, y로 선택
    x, y = rotated_rectangle.centroid.x, rotated_rectangle.centroid.y
    return x, y, w, h, theta

for city in city_list:
    print(city)
    os.makedirs(f'./3_Polygon2Graph/{city}', exist_ok=True)

    folder_path = f'./2_FilteredData/{city}/'
    all_files = os.listdir(folder_path)

    file_idx = 1
    for idx in tqdm(range(len(all_files))):
        filtered_file_path = f'./2_FilteredData/{city}/block_{idx + 1}.pkl'
        with open(filtered_file_path, 'rb') as f:
            data = pickle.load(f)

        boundary_polygon = data['simplified_boundary']
        boundary_segments = data['simplified_boundary_segments']
        building_polygons = data['simplified_building_polygons']
        polygon_file_path = data['polygon_file_path']
        raw_boundary_file_path = data['raw_boundary_file_path']
        raw_building_file_path = data['raw_building_file_path']

        with open(polygon_file_path, 'rb') as f:
            raw_data = pickle.load(f)

        scale_factor = 0.9 / raw_data['scale_factor']
        unit_length = UNIT_LENGTH / scale_factor

        original_boundary_segments = boundary_segments
        boundary_segments = remove_close_points_numpy(boundary_segments, unit_length / 5)
        boundary_segments = remove_consecutive_duplicates(boundary_segments)

        boundary_line = LineString(boundary_segments + [boundary_segments[0]])

        segment_count = len(boundary_segments)
        edge_index = []
        for segment_idx in range(segment_count):
            edge_index.append([segment_idx, segment_idx])
            edge_index.append([segment_idx, (segment_idx + 1) % segment_count])
            edge_index.append([(segment_idx + 1) % segment_count, segment_idx])

        visited_building_indices = set()
        for segment_idx in range(segment_count):
            pre_segment = boundary_segments[segment_idx - 1]
            cur_segment = boundary_segments[segment_idx]
            next_segment = boundary_segments[(segment_idx + 1) % segment_count]

            rotated_line = rotated_90(cur_segment, next_segment, unit_length * 2.5)
            rotated_line = rotated_line - np.mean([cur_segment, next_segment], axis=0) + cur_segment
            rotated_line = LineString(rotated_line)

            is_valid_edge = True
            for segment_jdx in range(segment_count):
                if segment_idx == segment_jdx or segment_idx == (segment_jdx + 1) % segment_count or segment_idx == segment_jdx - 1:
                    continue
                new_line = LineString([boundary_segments[segment_jdx - 1],
                                       boundary_segments[segment_jdx],
                                       boundary_segments[(segment_jdx + 1) % segment_count]])

                if new_line.intersects(rotated_line):
                    is_valid_edge = False

            if not is_valid_edge:
                continue

            min_distance = float('inf')
            closest_building_idx = None

            for building_idx in range(len(building_polygons)):
                if rotated_line.intersects(building_polygons[building_idx]):
                    distance = Point(cur_segment).distance(building_polygons[building_idx])
                    if distance < min_distance:
                        min_distance = distance
                        closest_building_idx = building_idx

            if closest_building_idx is not None:
                edge_index.append([segment_idx, closest_building_idx + segment_count])
                edge_index.append([closest_building_idx + segment_count, segment_idx])
                visited_building_indices.add(closest_building_idx)

        for building_idx in range(len(building_polygons)):
            edge_index.append([building_idx + segment_count, building_idx + segment_count])

            for building_jdx in range(building_idx + 1, len(building_polygons)):
                if building_polygons[building_idx].distance(building_polygons[building_jdx]) > unit_length * 2.5:
                    continue

                nearest_check_line = LineString(nearest_points(building_polygons[building_idx], building_polygons[building_jdx]))
                centroid_check_line = LineString([building_polygons[building_idx].centroid, building_polygons[building_jdx].centroid])

                is_valid_edge = True
                for building_kdx in range(len(building_polygons)):
                    if building_kdx == building_idx or building_kdx == building_jdx:
                        continue

                    if building_polygons[building_kdx].intersects(nearest_check_line) or \
                            building_polygons[building_kdx].intersects(centroid_check_line):
                        is_valid_edge = False
                        break

                if boundary_line.intersects(nearest_check_line) or \
                        boundary_line.intersects(centroid_check_line):
                    is_valid_edge = False
                    break

                if is_valid_edge:
                    edge_index.append([building_idx + segment_count, building_jdx + segment_count])
                    edge_index.append([building_jdx + segment_count, building_idx + segment_count])
                    visited_building_indices.add(building_idx)
                    visited_building_indices.add(building_jdx)

        all_building_indices = set(range(len(building_polygons)))
        not_visited_building_indices = all_building_indices - visited_building_indices

        for not_visited_building_idx in not_visited_building_indices:
            if not_visited_building_idx in visited_building_indices:
                continue

            min_distance = float('inf')
            closest_idx = None

            for building_idx in range(len(building_polygons)):
                if not_visited_building_idx == building_idx:
                    continue

                distance = building_polygons[not_visited_building_idx].distance(building_polygons[building_idx])
                if distance < min_distance:
                    min_distance = distance
                    closest_idx = building_idx + segment_count

            closet_segment_idx = None
            for segment_idx in range(segment_count):
                distance = building_polygons[not_visited_building_idx].distance(Point(boundary_segments[segment_idx]))
                if distance < min_distance:
                    min_distance = distance
                    closest_idx = segment_idx

            if closest_idx is not None:
                edge_index.append([not_visited_building_idx + segment_count, closest_idx])
                edge_index.append([closest_idx, not_visited_building_idx + segment_count])
                visited_building_indices.add(not_visited_building_idx)
                if closest_idx > segment_count:
                    visited_building_indices.add(closest_idx - segment_count)

        building_bbox_features = []
        for building_polygon in building_polygons:
            building_bbox = building_polygon.minimum_rotated_rectangle
            building_bbox_feature = get_bbox_details(building_bbox)
            building_bbox_features.append(building_bbox_feature)

        save_path = f'./3_Polygon2Graph/{city}/block_{file_idx}.pkl'
        file_idx += 1

        # 저장할 데이터
        transformed_data = {
            'boundary_polygon': boundary_polygon,
            'boundary_segments': boundary_segments,
            'building_polygons': building_polygons,
            'building_bbox_features': building_bbox_features,
            'edge_index': edge_index,
            'boundary_segment_count': segment_count,
            'building_count': len(building_polygons),
            'polygon_file_path': polygon_file_path,
            'filtered_file_path': filtered_file_path,
            'raw_boundary_file_path': raw_boundary_file_path,
            'raw_building_file_path': raw_building_file_path,
            'city_name': city
        }

        # pkl 파일로 저장
        with open(save_path, 'wb') as f:
            pickle.dump(transformed_data, f)

        # visualize_boundary_with_buildings(boundary_segments, building_polygons, edge_index, original_boundary_segments)
        seen = set()
        duplicate_found = False

        for edge in edge_index:
            edge_tuple = tuple(edge)
            if edge_tuple in seen:
                duplicate_found = True
                print(f"Duplicate found: {edge_tuple} {save_path}")
                print(edge_index)
                visualize_boundary_with_buildings(boundary_segments, building_polygons, edge_index)
            else:
                seen.add(edge_tuple)
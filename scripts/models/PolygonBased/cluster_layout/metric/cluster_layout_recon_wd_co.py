import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import Polygon, box
from shapely.affinity import rotate
from tqdm import tqdm
from scipy.stats import wasserstein_distance
import ot

def load_data(output_dir):
    coords_path = os.path.join(output_dir, 'predicted_coords.npz')

    if not os.path.exists(coords_path):
        raise FileNotFoundError(f"Coordinates file not found at {coords_path}")

    data = np.load(coords_path)
    predicted_coords = data['all_coords_outputs']
    gt_coords = data['gt_coords_outputs']
    min_coords_outputs = data['min_coords_outputs']
    range_max_outputs = data['range_max_outputs']
    return predicted_coords, gt_coords, min_coords_outputs, range_max_outputs

def denormalize_coords_uniform(norm_coords, min_coords, range_max):
    """
    정규화된 좌표를 원래의 좌표로 되돌립니다.

    Parameters:
    - norm_coords (array-like): 정규화된 좌표.
    - min_coords (array-like): 정규화 시 사용된 최소 좌표값.
    - range_max (float): 정규화 시 사용된 최대 범위값.

    Returns:
    - ndarray: 원래의 좌표.
    """
    norm_coords = np.array(norm_coords)
    min_coords = np.array(min_coords)
    return norm_coords * range_max + min_coords

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

def overlap(predicted_coords, gt_coords, min_coords_outputs, range_max_outputs, save_dir):
    """
    Visualize the processed coordinates.

    Parameters:
    - processed_coords: list of numpy arrays, each containing the valid coordinates for a sample
    - save_dir: directory to save the plots
    - sample_size: number of samples to visualize
    """

    wds = []
    for idx in tqdm(range(len(predicted_coords))):
        min_coords = min_coords_outputs[idx]
        range_max = range_max_outputs[idx][0]

        predicted_layouts = []
        gt_layouts = []
        for layout_idx in range(len(predicted_coords[idx])):
            x, y, w, h, r, c = predicted_coords[idx][layout_idx]
            predicted_layouts.append([c])

            x, y, w, h, r, c = gt_coords[idx][layout_idx]
            gt_layouts.append([c])

        predicted_layouts = np.array(predicted_layouts)
        gt_layouts = np.array(gt_layouts)
        # 각 분포의 가중치 설정 (동일 가중치 또는 사용자 지정 가중치)
        a = np.ones((predicted_layouts.shape[0],)) / predicted_layouts.shape[0]  # data1의 가중치
        b = np.ones((gt_layouts.shape[0],)) / gt_layouts.shape[0]  # data2의 가중치

        # 비용 행렬 계산 (유클리드 거리)
        M = ot.dist(predicted_layouts, gt_layouts, metric='euclidean')

        # Wasserstein 거리 계산 (정확한 거리)
        wasserstein_dist = ot.emd2(a, b, M)
        wds.append(wasserstein_dist)

    # Calculate statistics
    wds_mean = np.mean(wds)
    wds_min = np.min(wds)
    wds_max = np.max(wds)
    # Print results
    print("f'Wasserstein Distance - Mean:", wds_mean, "Min:", wds_min, "Max:", wds_max)
    
    # Save to txt file
    os.makedirs(save_dir, exist_ok=True)
    with open(save_dir + '/result.txt', "w") as file:
        file.write("Wasserstein Distance:\n")
        file.write(f"Mean: {wds_mean}\nMin: {wds_min}\nMax: {wds_max}\n\n")

def main():
    parser = argparse.ArgumentParser(description='Visualize Inference Results.')
    parser.add_argument('--output_dir', type=str, default='inference_outputs/d_256_cb_512_st_9', help='Directory where inference results are saved.')
    parser.add_argument('--save_dir', type=str, default='metric_result/wd_co', help='Directory to save the visualization plots.')
    args = parser.parse_args()

    model_name = args.output_dir.split('/')[1]
    args.save_dir = args.save_dir + '/' + model_name

    predicted_coords, gt_coords, min_coords_outputs, range_max_outputs = load_data(args.output_dir)

    # Visualize individual samples
    overlap(predicted_coords, gt_coords, min_coords_outputs, range_max_outputs, args.save_dir)

if __name__ == "__main__":
    main()

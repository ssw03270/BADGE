import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import Polygon, box
from tqdm import tqdm

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

def overlap(predicted_coords, gt_coords, min_coords_outputs, range_max_outputs, save_dir):
    """
    Visualize the processed coordinates.

    Parameters:
    - processed_coords: list of numpy arrays, each containing the valid coordinates for a sample
    - save_dir: directory to save the plots
    - sample_size: number of samples to visualize
    """

    predicted_overlaps = []
    gt_overlaps = []
    for idx in tqdm(range(len(predicted_coords))):
        min_coords = min_coords_outputs[idx]
        range_max = range_max_outputs[idx][0]

        predicted_layouts = []
        gt_layouts = []
        for layout_idx in range(len(predicted_coords[idx])):
            x, y, w, h, r, c = predicted_coords[idx][layout_idx]
            if c > 0.5:
                bbox = create_bounding_box(x, y, w, h, r * 360)
                bbox_coords = denormalize_coords_uniform(bbox.exterior.coords, min_coords, range_max)
                bbox = Polygon(bbox_coords)

                predicted_layouts.append(bbox)

            x, y, w, h, r, c = gt_coords[idx][layout_idx]
            if c > 0.5:
                bbox = create_bounding_box(x, y, w, h, r * 360)
                bbox_coords = denormalize_coords_uniform(bbox.exterior.coords, min_coords, range_max)
                bbox = Polygon(bbox_coords)

                gt_layouts.append(bbox)
        
        predict_overlap_count = 0
        for layout_idx in range(len(predicted_layouts)):
            for layout_jdx in range(layout_idx + 1, len(predicted_layouts)):
                if layout_idx == layout_jdx:
                    continue
                    
                if predicted_layouts[layout_idx].intersection(predicted_layouts[layout_jdx]):
                    predict_overlap_count += 1
                 
        gt_overlap_count = 0   
        for layout_idx in range(len(gt_layouts)):
            for layout_jdx in range(layout_idx + 1, len(gt_layouts)):
                if layout_idx == layout_jdx:
                    continue
                    
                if gt_layouts[layout_idx].intersection(gt_layouts[layout_jdx]):
                    gt_overlap_count += 1
                    
        predicted_overlaps.append(predict_overlap_count)
        gt_overlaps.append(gt_overlap_count)

    # Calculate statistics
    predicted_mean = np.mean(predicted_overlaps)
    predicted_min = np.min(predicted_overlaps)
    predicted_max = np.max(predicted_overlaps)
    
    gt_mean = np.mean(gt_overlaps)
    gt_min = np.min(gt_overlaps)
    gt_max = np.max(gt_overlaps)
    
    # Print results
    print("Predicted Overlaps - Mean:", predicted_mean, "Min:", predicted_min, "Max:", predicted_max)
    print("GT Overlaps - Mean:", gt_mean, "Min:", gt_min, "Max:", gt_max)
    
    # Save to txt file
    os.makedirs(save_dir, exist_ok=True)
    with open(save_dir + '/result.txt', "w") as file:
        file.write("Predicted Overlaps:\n")
        file.write(f"Mean: {predicted_mean}\nMin: {predicted_min}\nMax: {predicted_max}\n\n")
        
        file.write("GT Overlaps:\n")
        file.write(f"Mean: {gt_mean}\nMin: {gt_min}\nMax: {gt_max}\n")

def main():
    parser = argparse.ArgumentParser(description='Visualize Inference Results.')
    parser.add_argument('--output_dir', type=str, default='inference_outputs/d_256_cb_512_st_9', help='Directory where inference results are saved.')
    parser.add_argument('--save_dir', type=str, default='metric_result/overlap', help='Directory to save the visualization plots.')
    args = parser.parse_args()

    model_name = args.output_dir.split('/')[1]
    args.save_dir = args.save_dir + '/' + model_name

    predicted_coords, gt_coords, min_coords_outputs, range_max_outputs = load_data(args.output_dir)

    # Visualize individual samples
    overlap(predicted_coords, gt_coords, min_coords_outputs, range_max_outputs, args.save_dir)

if __name__ == "__main__":
    main()

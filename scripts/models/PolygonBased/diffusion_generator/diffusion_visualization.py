import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from shapely.geometry import Polygon, box
from tqdm import tqdm
import pickle

def load_data(output_dir):
    coords_path = os.path.join(output_dir, 'predicted_coords.pkl')

    if not os.path.exists(coords_path):
        raise FileNotFoundError(f"Coordinates file not found at {coords_path}")

    with open(coords_path, 'rb') as f:
        data = pickle.load(f)
    predicted_coords = data['all_coords_outputs']
    gt_coords = data['gt_coords_outputs']
    region_polygons_outputs = data['region_polygons_outputs']
    return predicted_coords, gt_coords, region_polygons_outputs

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

def visualize(predicted_coords, gt_coords, region_polygons_outputs, save_dir, sample_size=100):
    """
    Visualize the processed coordinates.

    Parameters:
    - processed_coords: list of numpy arrays, each containing the valid coordinates for a sample
    - save_dir: directory to save the plots
    - sample_size: number of samples to visualize
    """
    os.makedirs(save_dir, exist_ok=True)

    num_samples = min(sample_size, len(predicted_coords))

    for i in tqdm(range(num_samples)):
        plt.figure(figsize=(8, 8))
        ax = plt.gca()
        
        if region_polygons_outputs is not None:
            region_polygons = region_polygons_outputs[i]
            
            for x, y in region_polygons:
                ax.plot(x, y, color='black')

        layout_i = predicted_coords[i]  # Shape: (num_coords, 2)
        for layout_ii in layout_i:
            x, y, w, h, r, c = layout_ii
            if c < 0.5:
                continue
            bbox = create_bounding_box(x, y, w, h, r * 360)

            x, y = bbox.exterior.xy
            ax.fill(x, y, color='black')  # 색상 추가

        ax.set_aspect('equal')
        ax.axis('off')  # 축, 틱, 라벨 제거

        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.5, 1.5)

        # Save the plot
        plot_path = os.path.join(save_dir, f'{i+1}_predict.png')
        plt.savefig(plot_path)
        plt.close()

    for i in tqdm(range(num_samples)):
        plt.figure(figsize=(8, 8))
        ax = plt.gca()
        
        if region_polygons_outputs is not None:
            region_polygons = region_polygons_outputs[i]
            
            for x, y in region_polygons:
                ax.plot(x, y, color='black')
         
        layout_i = gt_coords[i]  # Shape: (num_coords, 2)
        for layout_ii in layout_i:
            x, y, w, h, r, c = layout_ii
            if c < 0.5:
                continue
            bbox = create_bounding_box(x, y, w, h, r * 360)

            x, y = bbox.exterior.xy
            ax.fill(x, y, color='black')  # 색상 추가

        ax.set_aspect('equal')
        ax.axis('off')  # 축, 틱, 라벨 제거

        ax.set_xlim(-0.5, 1.5)
        ax.set_ylim(-0.5, 1.5)

        # Save the plot
        plot_path = os.path.join(save_dir, f'{i+1}_gt.png')
        plt.savefig(plot_path)
        plt.close()

    print(f"Visualization of {num_samples} samples completed. Plots saved to '{save_dir}'")

def main():
    parser = argparse.ArgumentParser(description='Visualize Inference Results.')
    parser.add_argument('--output_dir', type=str, default='inference_outputs/original_conditional_noise', help='Directory where inference results are saved.')
    parser.add_argument('--save_dir', type=str, default='visualizations/', help='Directory to save the visualization plots.')
    parser.add_argument('--sample_size', type=int, default=1000, help='Number of individual samples to visualize.')
    parser.add_argument('--aggregate', action='store_true', help='Whether to create an aggregate visualization.')
    parser.add_argument('--vis_blk', type=bool, default=False, help='Whether to create an aggregate visualization.')
    args = parser.parse_args()

    predicted_coords, gt_coords, region_polygons_outputs = load_data(args.output_dir)

    if not args.vis_blk:
        model_name = args.output_dir.split('/')[1]
        region_polygons_outputs = None
    else:
        model_name = args.output_dir.split('/')[1] + '_w_blk'

    # Visualize individual samples
    visualize(predicted_coords, gt_coords, region_polygons_outputs, args.save_dir + '/' + model_name, sample_size=args.sample_size)

if __name__ == "__main__":
    main()

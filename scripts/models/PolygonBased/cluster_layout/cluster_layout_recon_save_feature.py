import multiprocessing
from PIL import Image, ImageDraw

import os
import pickle
import numpy as np
from tqdm import tqdm

import torch
from accelerate.utils import set_seed
from transformer import Transformer

def create_mask(polygon, image_size=(64, 64)):
    """
    노멀라이즈된 shapely 폴리곤을 이미지 마스크로 변환합니다.

    Parameters:
    - polygon: shapely.geometry.Polygon, 0과 1 사이로 노멀라이즈된 좌표를 가진 폴리곤
    - image_size: tuple, 생성할 마스크 이미지의 크기 (width, height)

    Returns:
    - mask: numpy.ndarray, 마스크 이미지 (0과 1로 구성된 배열)
    """
    width, height = image_size

    # 폴리곤의 좌표를 이미지 크기에 맞게 스케일링
    scaled_coords = [(x * width, (1 - y) * height) for x, y in polygon.exterior.coords]

    # PIL 이미지 생성 (흰색 배경)
    img = Image.new('L', image_size, 0)
    ImageDraw.Draw(img).polygon(scaled_coords, outline=1, fill=1)

    # numpy 배열로 변환
    mask = np.array(img)

    return mask

dataset_path = "Z:/iiixr-drive/Projects/2023_City_Team/000_2024CVPR/Our_dataset"
subfolders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]

# 패딩할 최대 빌딩 개수
MAX_BUILDINGS = 10
PADDING_BUILDING = [0, 0, 0, 0, 0, 0]

debug = False
def process_folder(folder):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    set_seed(42)

    # Initialize the model architecture (ensure it matches the training setup)
    d_model = 128
    d_inner = d_model * 4
    n_layer = 4
    n_head = 8
    dropout = 0.1
    codebook_size, commitment_cost = 16, 0.25
    n_tokens = 10
    model = Transformer(
        d_model=d_model,
        d_inner=d_inner,
        n_layer=n_layer,
        n_head=n_head,
        dropout=dropout,
        codebook_size=codebook_size,
        commitment_cost=commitment_cost,
        n_tokens=n_tokens
    )

    # Load the checkpoint
    checkpoint_path = "./vq_model_checkpoints/d_model_128_codebook_16/best_model.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    error_count = 0
    output_path = os.path.join(dataset_path, folder, f'codebook_indices/{folder}_graph_prep_list_with_clusters_detail.pkl')
    # if os.path.exists(output_path):
    #     print(f"{output_path} 파일이 이미 존재합니다. 건너뜁니다.")
    #     return  # 이미 처리된 경우 건너뜁니다.
    
    os.makedirs(os.path.join(dataset_path, folder, f'codebook_indices/'), exist_ok=True)  # output 디렉토리 생성

    data_path = os.path.join(dataset_path, folder, f'train_codebook/{folder}_graph_prep_list_with_clusters_detail.pkl')
    if not os.path.exists(data_path):
        print(f"{data_path} 파일이 존재하지 않습니다. 건너뜁니다.")
        return

    with open(data_path, 'rb') as f:
        data_list = pickle.load(f)
    print(f"{folder} 파일을 처리하는 중입니다. ")
    new_data_list = []
    for data in tqdm(data_list):
        cluster_id2normalized_bldg_layout_cluster_list = data['cluster_id2normalized_bldg_layout_cluster_list']
        cluster_id2normalized_bldg_layout_cluster_list = data['cluster_id2normalized_bldg_layout_cluster_list']

        cluster_id2encoding_indices = {}
        cluster_id2encoding_indices[0] = [codebook_size] * d_model
        for cluster_id, normalized_bldg_layout_cluster_list in cluster_id2normalized_bldg_layout_cluster_list.items():
            # 패딩할 최대 빌딩 개수
            MAX_BUILDINGS = 10
            PADDING_BUILDING = [0, 0, 0, 0, 0, 0]

            bldg_layout_list = np.array(normalized_bldg_layout_cluster_list)

            current_building_count = bldg_layout_list.shape[0]
            if current_building_count < MAX_BUILDINGS:
                padding_needed = MAX_BUILDINGS - current_building_count
                # 패딩할 배열 생성
                padding_array = np.array([PADDING_BUILDING] * padding_needed)
                # 패딩된 배열 결합
                cluster_boundary_padded = np.vstack((bldg_layout_list, padding_array))
            else:
                # 빌딩 개수가 이미 최대인 경우 필요시 자름
                cluster_boundary_padded = bldg_layout_list[:MAX_BUILDINGS]
            
            cluster_boundary_padded = cluster_boundary_padded.reshape(1, 10, 6)
            input_layout = torch.tensor(cluster_boundary_padded, dtype=torch.float32, requires_grad=True).to(device)

            encoding_indices = model.get_encoding_indices(input_layout).cpu().numpy().reshape(-1).tolist()
            cluster_id2encoding_indices[cluster_id] = encoding_indices
            
        new_data = data
        new_data['cluster_id2encoding_indices'] = cluster_id2encoding_indices
        new_data_list.append(new_data)

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
'''
이 코드는 generator model 의 dataset 을 preprocessing 하기 위해 필요함
preprocessing_all.py 코드에 의해 전처리된 데이터를 한 번 더 전처리 할 것임

이떄, 필요한 정보는 다음과 같음
Condition: block mask (채워진 거, 테두리), region mask (채널 별 클러스터 세그멘테이션 마스크), 
Target: cluster bbox
'''

import multiprocessing
from PIL import Image, ImageDraw

import os
import pickle
import numpy as np
from tqdm import tqdm

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
    error_count = 0
    output_path = os.path.join(dataset_path, folder, f'train_codebook/{folder}_graph_prep_list_hierarchical_10_fixed.pkl')
    if os.path.exists(output_path):
        print(f"{output_path} 파일이 이미 존재합니다. 건너뜁니다.")
        return  # 이미 처리된 경우 건너뜁니다.
    
    os.makedirs(os.path.join(dataset_path, folder, f'train_codebook/'), exist_ok=True)  # output 디렉토리 생성

    data_path = os.path.join(dataset_path, folder, f'preprocessed/{folder}_graph_prep_list_hierarchical_10_fixed.pkl')
    if not os.path.exists(data_path):
        print(f"{data_path} 파일이 존재하지 않습니다. 건너뜁니다.")
        return

    with open(data_path, 'rb') as f:
        data_list = pickle.load(f)
    print(f"{folder} 파일을 처리하는 중입니다. ")
    new_data_list = []
    for data in tqdm(data_list):
        cluster_id2bldg_id_list = data['cluster_id2bldg_id_list']
        bldg_id2normalized_bldg_layout_cluster = data['bldg_id2normalized_bldg_layout_cluster']
        bldg_id2normalized_bldg_layout_bldg_bbox = data['bldg_id2normalized_bldg_layout_bldg_bbox']
        hierarchical_clustering_list = data['hierarchical_k_10']

        for clustering in hierarchical_clustering_list:
            if len(clustering) > 10:
                print(len(clustering))
       
        cluster_id2normalized_bldg_layout_cluster_list = {}
        cluster_id2normalized_bldg_layout_bldg_bbox_list = {}
        for cluster_id, bldg_id_list in cluster_id2bldg_id_list.items():
                
            for bldg_id in bldg_id_list:
                bldg_layout = bldg_id2normalized_bldg_layout_cluster[bldg_id]
                x, y, w, h, r = bldg_layout
                gt_bldg_layout = [x, y, w, h, r / 360, 1]

                if cluster_id in cluster_id2normalized_bldg_layout_cluster_list:
                    cluster_id2normalized_bldg_layout_cluster_list[cluster_id].append(gt_bldg_layout)
                else:
                    cluster_id2normalized_bldg_layout_cluster_list[cluster_id] = [gt_bldg_layout]
                    
                bldg_layout = bldg_id2normalized_bldg_layout_bldg_bbox[bldg_id]
                x, y, w, h, r = bldg_layout
                gt_bldg_layout = [x, y, w, h, r / 360, 1]

                if cluster_id in cluster_id2normalized_bldg_layout_bldg_bbox_list:
                    cluster_id2normalized_bldg_layout_bldg_bbox_list[cluster_id].append(gt_bldg_layout)
                else:
                    cluster_id2normalized_bldg_layout_bldg_bbox_list[cluster_id] = [gt_bldg_layout]

        new_data = data
        new_data['cluster_id2normalized_bldg_layout_cluster_list'] = cluster_id2normalized_bldg_layout_cluster_list
        new_data['cluster_id2normalized_bldg_layout_bldg_bbox_list'] = cluster_id2normalized_bldg_layout_bldg_bbox_list
        
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
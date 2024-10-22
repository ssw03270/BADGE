'''
이 코드는 generator model 의 dataset 을 preprocessing 하기 위해 필요함
preprocessing_all.py 코드에 의해 전처리된 데이터를 한 번 더 전처리 할 것임

이떄, 필요한 정보는 다음과 같음
Condition: block mask (채워진 거, 테두리), region mask (채널 별 클러스터 세그멘테이션 마스크), 
Target: cluster bbox
'''

import matplotlib.pyplot as plt
import multiprocessing
from PIL import Image, ImageDraw

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

debug = False
def process_folder(folder):
    error_count = 0
    output_path = os.path.join(dataset_path, folder, f'train_generator/{folder}_graph_prep_list_with_clusters_detail.pkl')
    # if os.path.exists(output_path):
    #     print(f"{output_path} 파일이 이미 존재합니다. 건너뜁니다.")
    #     return  # 이미 처리된 경우 건너뜁니다.
    
    os.makedirs(os.path.join(dataset_path, folder, f'train_generator/'), exist_ok=True)  # output 디렉토리 생성

    data_path = os.path.join(dataset_path, folder, f'preprocessed/{folder}_graph_prep_list_with_clusters_detail.pkl')
    if not os.path.exists(data_path):
        print(f"{data_path} 파일이 존재하지 않습니다. 건너뜁니다.")
        return

    with open(data_path, 'rb') as f:
        data_list = pickle.load(f)
    print(f"{folder} 파일을 처리하는 중입니다. ")
    new_data_list = []
    for data in tqdm(data_list):
        cluster_id2region_id_list = data['cluster_id2region_id_list']
        region_id2region_polygon = data['region_id2region_polygon']

        cluster_id2cluster_mask = {}
        for cluster_id, region_id_list in cluster_id2region_id_list.items():
            cluster_image_mask = np.zeros((64, 64))
            for region_id in region_id_list:
                region_polygon = region_id2region_polygon[region_id]
                                
                # 마스크 생성
                mask = create_mask(region_polygon, image_size=(64, 64))
                cluster_image_mask += mask

            cluster_id2cluster_mask[cluster_id] = cluster_image_mask
        
        new_data = data
        new_data['cluster_id2cluster_mask'] = cluster_id2cluster_mask
        
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
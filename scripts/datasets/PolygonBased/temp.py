import matplotlib.pyplot as plt

import os
import pickle
import numpy as np
from tqdm import tqdm
from pyproj import Proj, transform
from shapely.affinity import rotate
from shapely.affinity import translate
from shapely.ops import unary_union
from shapely.geometry import Polygon

UNIT_LENGTH = 1

dataset_path = "E:/Resources/COHO/Our_Dataset"
output_dir_path = "E:/Resources/COHO"
subfolders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]

n_coords = []
for folder in subfolders:
    output_path = os.path.join(output_dir_path, 'NormalizedBuildings', folder)  # output 경로 설정
    os.makedirs(output_path, exist_ok=True)  # output 디렉토리 생성

    data_path = os.path.join(dataset_path, folder, f'graph/{folder}_graph_prep_list.pkl')
    if not os.path.exists(data_path):
        print(f"{data_path} 파일이 존재하지 않습니다. 건너뜁니다.")
        continue  # 다음 폴더로 넘어감

    with open(data_path, 'rb') as f:
        data_list = pickle.load(f)
    
    for data in tqdm(data_list):
        if not 'bldg_poly_list' in data:
            continue
        
        bldg_poly_list = data['bldg_poly_list']
        for bldg_poly_data in bldg_poly_list:
            bldg_id = bldg_poly_data[0]
            bldg_poly = bldg_poly_data[1]
            bldg_coords = [list(coord) for coord in bldg_poly.exterior.coords]
            n_coords.append(len(bldg_coords))

avg_coords = np.mean(n_coords)
min_coords = np.min(n_coords)
max_coords = np.max(n_coords)
print(f"평균: {avg_coords}, 최소: {min_coords}, 최대: {max_coords}")

# n_coords에 대한 히스토그램 그리기
plt.hist(n_coords, bins=100, color='blue', alpha=0.7)
plt.title('Building Coordinates Histogram')
plt.xlabel('Number of Coordinates')
plt.ylabel('Frequency')
plt.grid(axis='y', alpha=0.75)
plt.show()  # 히스토그램 표시

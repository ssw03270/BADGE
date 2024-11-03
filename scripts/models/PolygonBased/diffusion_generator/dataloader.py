import torch
from torch.utils.data import Dataset
import numpy as np
import os
import pickle
import glob
from tqdm import tqdm

from PIL import Image  # PIL 라이브러리 사용
from torchvision import transforms
from shapely.geometry import Polygon

from torchvision import models

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

    return normalized_coords, min_coords, [range_max], out_of_bounds

class BlkLayoutDataset(Dataset):
    def __init__(self, data_type='train', device='cpu', processed_dir='./processed', is_main_process=True,
                 retrieval_type=None):
        """
        BlkLayoutDataset 클래스의 인스턴스를 초기화합니다.

        매개변수:
        - data_type (str): 데이터의 종류를 지정합니다 ('train', 'test', 'val').
        """

        super(BlkLayoutDataset, self).__init__()
        self.device = device

        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),  # ResNet도 224x224 이미지를 입력으로 받음
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ResNet의 표준 정규화 값
        ])

        self.data_type = data_type
        self.processed_dir = processed_dir
        self.retrieval_type = retrieval_type
        os.makedirs(self.processed_dir, exist_ok=True)

        if data_type == 'test':
            self.folder_path = f'E:/Resources/Our_dataset_divided_without_segmentation_mask'
        else:
            # self.folder_path = f'/data/{user_name}/datasets/CITY2024/Our_dataset'
            self.folder_path = f'/data2/local_datasets/CITY2024/Our_dataset_divided_without_segmentation_mask'
        self.pkl_files = glob.glob(os.path.join(self.folder_path, '**', '*.pkl'), recursive=True)
        self.pkl_files = np.array(self.pkl_files)

        shuffled_indices = np.random.permutation(self.pkl_files.shape[0])
        self.pkl_files = self.pkl_files[shuffled_indices]

        # 데이터 분할
        total_files = len(self.pkl_files)
        train_split = int(0.7 * total_files)
        val_split = int(0.2 * total_files)

        if data_type == 'train':
            self.pkl_files = self.pkl_files[:train_split]
        elif data_type == 'val':
            self.pkl_files = self.pkl_files[train_split:train_split + val_split]
        elif data_type == 'test':
            self.pkl_files = self.pkl_files[train_split + val_split:]

        if self.retrieval_type == 'retrieval':
            generated_output_dict_path = "inference_outputs/d_256_cb_512_coords_continuous_norm_blk_generate_retrieval/generated_output_dict.pkl"
            with open(generated_output_dict_path, 'rb') as f:
                self.dict_data = pickle.load(f)

        self.data_length = len(self.pkl_files)
        print(f"총 {self.data_length}개의 데이터를 로드했습니다 (after preprocessing).")

    def __getitem__(self, idx):
        with open(self.pkl_files[idx], 'rb') as f:
            data = pickle.load(f)
            region_polygons = data['region_id2region_polygon']
            cluster_id2normalized_bldg_layout_blk_list = data['cluster_id2normalized_bldg_layout_blk_list']
            image_mask = data['blk_image_mask']
            region_id2cluster_id_list = data['region_id2cluster_id_list']

            if self.retrieval_type == 'retrieval':
                file_path = self.pkl_files[idx]
                file_path = file_path.replace(self.folder_path, '')
                file_path = file_path.replace('\\', '/')
                layouts = self.dict_data[file_path]

        # 어텐션 맵 초기화 (고정 크기 300x300)
        MAX_BUILDINGS = 300
        cross_attn_map = np.zeros((MAX_BUILDINGS, MAX_BUILDINGS), dtype=np.float32)
        self_attn_map = np.zeros((MAX_BUILDINGS, MAX_BUILDINGS), dtype=np.float32)

        # 빌딩 인덱스 매핑 (클러스터 내 빌딩 위치를 추적)
        bldg_indices = {}
        current_index = 0
        for cluster_id, bldg_layout_list in cluster_id2normalized_bldg_layout_blk_list.items():
            indices = []
            for _ in bldg_layout_list:
                if current_index >= MAX_BUILDINGS:
                    break  # 최대 빌딩 개수를 초과하면 중단
                indices.append(current_index)
                current_index += 1
            bldg_indices[cluster_id] = indices
            if current_index >= MAX_BUILDINGS:
                break  # 최대 빌딩 개수를 초과하면 전체 루프 중단

        # 클러스터 폴리곤 생성
        from shapely.ops import unary_union

        cluster_polygons = {}
        for cluster_id in bldg_indices.keys():
            # 해당 클러스터에 속한 모든 region_id 가져오기
            region_ids_in_cluster = [region_id for region_id, c_id in region_id2cluster_id_list.items() if cluster_id in c_id]
            # 해당 region_id들의 폴리곤 가져오기
            polygons_in_cluster = [region_polygons[region_id] for region_id in region_ids_in_cluster]
            # 폴리곤 합치기 (union)
            cluster_polygon = unary_union(polygons_in_cluster)
            cluster_polygons[cluster_id] = cluster_polygon
            
        # Self-Attention 맵 설정
        for cluster_id, indices in bldg_indices.items():
            for i in indices:
                for j in indices:
                    if i < MAX_BUILDINGS and j < MAX_BUILDINGS:
                        self_attn_map[i, j] = 1  # 같은 클러스터 내 빌딩끼리 연결

        # Cross-Attention 맵 설정
        clusters = list(bldg_indices.keys())
        for i in clusters:
            for j in clusters:
                if i >= j:
                    continue  # 자기 자신 및 이미 처리한 쌍은 건너뜁니다.

                # 두 클러스터의 폴리곤 가져오기
                polygon1 = cluster_polygons[i]
                polygon2 = cluster_polygons[j]

                # 인접성 확인
                if polygon1.intersection(polygon2):
                    # 빌딩 인덱스 가져오기
                    indices_i = bldg_indices[i]
                    indices_j = bldg_indices[j]

                    # cross_attn_map 설정
                    for bldg1 in indices_i:
                        for bldg2 in indices_j:
                            if bldg1 < MAX_BUILDINGS and bldg2 < MAX_BUILDINGS:
                                cross_attn_map[bldg1, bldg2] = 1
                                cross_attn_map[bldg2, bldg1] = 1  # 양방향 설정

        # 이미지 마스크 전처리
        image_mask = Image.fromarray(image_mask).convert("RGB")
        image_mask = self.preprocess(image_mask)

        # 빌딩 레이아웃 리스트 생성
        bldg_layout_list = []
        if self.retrieval_type == 'retrieval':
            # 'retrieval' 타입일 경우 layouts에서 빌딩 레이아웃을 가져옴
            for cluster_id, bldg_layouts in layouts.items():
                for bldg_layout in bldg_layouts:
                    bldg_layout_list.append(bldg_layout)
        else:
            # 다른 타입일 경우 cluster_id2normalized_bldg_layout_blk_list에서 빌딩 레이아웃을 가져옴
            for cluster_id, bldg_layouts in cluster_id2normalized_bldg_layout_blk_list.items():
                for bldg_layout in bldg_layouts:
                    bldg_layout_list.append(bldg_layout)

        bldg_layout_list = np.array(bldg_layout_list, dtype=np.float32)
        current_building_count = bldg_layout_list.shape[0]
        padding_mask = np.ones(current_building_count, dtype=np.float32)

        # 패딩 처리
        PADDING_BUILDING = [0, 0, 0, 0, 0, 0]  # 필요에 따라 패딩 값을 조정
        if current_building_count < MAX_BUILDINGS:
            padding_needed = MAX_BUILDINGS - current_building_count
            padding_array = np.array([PADDING_BUILDING] * padding_needed, dtype=np.float32)
            bldg_layout_list = np.vstack((bldg_layout_list, padding_array))
            padding_mask = np.concatenate([padding_mask, np.zeros(padding_needed, dtype=np.float32)])
        else:
            bldg_layout_list = bldg_layout_list[:MAX_BUILDINGS]
            padding_mask = padding_mask[:MAX_BUILDINGS]

        # region_polygons 처리 (좌표 리스트로 변환)
        region_polygons_processed = []
        for region_id, region_poly in region_polygons.items():
            if isinstance(region_poly, Polygon):
                # (N, 2) 형태로 변환
                coords = np.array(region_poly.exterior.coords.xy).T
                region_polygons_processed.append(coords)
            else:
                # Polygon이 아닌 경우 (예: MultiPolygon), 적절히 처리
                if hasattr(region_poly, 'geoms') and len(region_poly.geoms) > 0:
                    coords = np.array(region_poly.geoms[0].exterior.coords.xy).T
                    region_polygons_processed.append(coords)
                else:
                    # 처리할 수 없는 폴리곤 형태인 경우 빈 배열로 추가
                    region_polygons_processed.append(np.array([]))
                    print(f"Warning: region_polygon is not a Polygon or MultiPolygon for region_id {region_id}.")

        # 텐서 변환
        bldg_layout_tensor = torch.tensor(bldg_layout_list, dtype=torch.float32)
        padding_mask_tensor = torch.tensor(padding_mask, dtype=torch.float32)
        cross_attn_map_tensor = torch.tensor(cross_attn_map, dtype=torch.float32)
        self_attn_map_tensor = torch.tensor(self_attn_map, dtype=torch.float32)

        # region_polygons는 길이가 가변적이므로, 텐서로 변환이 어려움
        # 필요에 따라 패딩하거나, 리스트로 반환할 수 있음
        # 여기서는 리스트로 반환
        region_polygons_tensor = region_polygons_processed  # 리스트로 유지

        # except EOFError:
        #     print(f"EOFError: Failed to load {self.pkl_files[idx]}. The file may be corrupted or incomplete.")
        #     return None  # Skip this file
        # except Exception as e:
        #     print(f"Error loading {self.pkl_files[idx]}: {e}")
        #     return None  # Skip this file

        # 데이터 반환 (딕셔너리 형태로 반환하여 접근 편리)
        return (bldg_layout_tensor, image_mask, padding_mask_tensor, cross_attn_map_tensor,
                self_attn_map_tensor, region_polygons_tensor)
        return {
            'bldg_layout': bldg_layout_tensor,          # (300, 6)
            'image_mask': image_mask,                   # (3, 224, 224)
            'padding_mask': padding_mask_tensor,        # (300,)
            'cross_attn_map': cross_attn_map_tensor,    # (300, 300)
            'self_attn_map': self_attn_map_tensor,      # (300, 300)
            'region_polygons': region_polygons_tensor    # List of (N, 2) arrays
        }

    def __len__(self):
        """
        데이터셋의 총 아이템 수를 반환합니다.

        반환값:
        - int: 데이터셋의 총 아이템 수.
        """

        return self.data_length
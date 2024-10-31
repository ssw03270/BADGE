import torch
from torch.utils.data import Dataset
import numpy as np
import os
import pickle
import glob
from tqdm import tqdm

from PIL import Image  # PIL 라이브러리 사용
from torchvision import transforms

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
    def __init__(self, data_type='train', device='cpu'):
        """
        BlkLayoutDataset 클래스의 인스턴스를 초기화합니다.

        매개변수:
        - data_type (str): 데이터의 종류를 지정합니다 ('train', 'test', 'val').
        """

        super(BlkLayoutDataset, self).__init__()

        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),  # ResNet도 224x224 이미지를 입력으로 받음
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ResNet의 표준 정규화 값
        ])

        self.data_type = data_type

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
        self.pkl_files = self.pkl_files

        self.resnet18 = models.resnet18(pretrained=True)
        modules = list(self.resnet18.children())[:-1]  # Remove the last FC layer
        self.resnet18 = torch.nn.Sequential(*modules)
        self.resnet18 = self.resnet18.to(device)
        self.resnet18.eval()
        
        self.data_list = []
        for file_path in tqdm(self.pkl_files, desc="데이터를 메모리에 적재 중"):
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                    region_polygons = data['region_id2region_polygon']
                    layouts = data['cluster_id2normalized_bldg_layout_blk_list']
                    image_mask = data['blk_image_mask']
                    
                    image_mask = Image.fromarray(image_mask)  # NumPy 배열을 PIL 이미지로 변환
                    image_mask = image_mask.convert("RGB")
                    image_mask = self.preprocess(image_mask).unsqueeze(0).to(device)
                    image_mask = self.resnet18(image_mask).squeeze().detach().cpu().numpy()
                    
                # 필요한 데이터만 저장
                self.data_list.append({
                    'image_mask': image_mask,
                    'layouts': layouts,
                    'region_polygons': [region_poly.exterior.coords.xy for region_poly in list(region_polygons.values())]
                })
            except EOFError:
                print(f"EOFError: {file_path} 로드에 실패했습니다. 파일이 손상되었거나 불완전할 수 있습니다.")
                continue  # 해당 파일 건너뜀
            except Exception as e:
                print(f"{file_path} 로드 중 오류 발생: {e}")
                continue  # 해당 파일 건너뜀
            
        self.data_length = len(self.data_list)
        print(f"총 {self.data_length}개의 데이터를 로드합니다.")

    def __getitem__(self, idx):
        """
        지정된 인덱스의 데이터를 반환합니다.

        매개변수:
        - idx (int): 가져올 데이터의 인덱스.

        반환값:
        - 데이터 텐서 및 관련 정보.
        """
        data = self.data_list[idx]
        
        image_mask = data['blk_image_mask']
        region_polygons = data['region_id2region_polygon']
        layouts = data['cluster_id2normalized_bldg_layout_blk_list']

        MAX_BUILDINGS = 300
        PADDING_BUILDING = [0, 0, 0, 0, 0, 0]

        bldg_layout_list = []
        for cluster_id, bldg_layouts in layouts.items():
            for bldg_layout in bldg_layouts:
                bldg_layout_list.append(bldg_layout)

        bldg_layout_list = np.array(bldg_layout_list)
        current_building_count = bldg_layout_list.shape[0]
        padding_mask = np.ones(current_building_count, dtype=int)

        if current_building_count < MAX_BUILDINGS:
            padding_needed = MAX_BUILDINGS - current_building_count
            padding_array = np.array([PADDING_BUILDING] * padding_needed)
            bldg_layout_list = np.vstack((bldg_layout_list, padding_array))
            padding_mask = np.concatenate([padding_mask, np.zeros(padding_needed, dtype=int)])
        else:
            bldg_layout_list = bldg_layout_list[:MAX_BUILDINGS]
            padding_mask = padding_mask[:MAX_BUILDINGS]

        return (torch.tensor(bldg_layout_list, dtype=torch.float32),
                torch.tensor(image_mask, dtype=torch.float32),
                torch.tensor(padding_mask, dtype=torch.float32),
                region_polygons)

    def __len__(self):
        """
        데이터셋의 총 아이템 수를 반환합니다.

        반환값:
        - int: 데이터셋의 총 아이템 수.
        """

        return self.data_length
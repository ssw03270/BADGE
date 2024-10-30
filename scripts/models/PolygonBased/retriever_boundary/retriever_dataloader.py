import torch
from torch.utils.data import Dataset
import numpy as np
import os
import pickle
import glob
from tqdm import tqdm
from PIL import Image  # PIL 라이브러리 사용
from torchvision import transforms

class BoundaryBlkDataset(Dataset):
    def __init__(self, data_type='train', user_name='ssw03270', coords_type='continuous', norm_type="bldg_bbox"):
        """
        ClusterLayoutDataset 클래스의 인스턴스를 초기화합니다.

        매개변수:
        - data_type (str): 데이터의 종류를 지정합니다 ('train', 'test', 'val').
        """

        super(BoundaryBlkDataset, self).__init__()

        self.preprocess = transforms.Compose([
            transforms.Resize((224, 224)),  # ResNet도 224x224 이미지를 입력으로 받음
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # ResNet의 표준 정규화 값
        ])
        
        self.data_type = data_type
        self.coords_type = coords_type
        self.norm_type = norm_type

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
        
        self.data_length = len(self.pkl_files)
        print(f"총 {self.data_length}개의 데이터를 로드합니다.")

    def __getitem__(self, idx):
        """
        지정된 인덱스의 데이터를 반환합니다.

        매개변수:
        - idx (int): 가져올 데이터의 인덱스.

        반환값:
        - 데이터 텐서 및 관련 정보.
        """
        pkl_file = self.pkl_files[idx]
        try:
            with open(pkl_file, 'rb') as f:
                    data = pickle.load(f)
        except EOFError:
            print(f"EOFError: Failed to load {pkl_file}. The file may be corrupted or incomplete.")
            return None  # 무효한 항목을 나타내는 None 반환
        except Exception as e:
            print(f"Error loading {pkl_file}: {e}")
            return None  # 무효한 항목을 나타내는 None 반환

        image_mask = data['blk_image_mask']
        image_mask = Image.fromarray(image_mask)  # NumPy 배열을 PIL 이미지로 변환
        image_mask = image_mask.convert("RGB")
        image_mask = self.preprocess(image_mask)

        file_path = pkl_file.replace(self.folder_path, '')
        file_path = file_path.replace('\\', '/')
        return (file_path, image_mask)

    def __len__(self):
        """
        데이터셋의 총 아이템 수를 반환합니다.

        반환값:
        - int: 데이터셋의 총 아이템 수.
        """

        return self.data_length


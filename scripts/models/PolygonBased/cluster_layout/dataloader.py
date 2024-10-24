import torch
from torch.utils.data import Dataset
import numpy as np
import os
import pickle
import random
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed
from functools import partial


def load_pickle_file_with_cache(subfolder, folder_path):
    """
    Loads and processes a single pickle file corresponding to a subfolder.

    Parameters:
    - subfolder (str): The name of the subfolder.
    - folder_path (str): The base path where subfolders are located.

    Returns:
    - list: A list of clusters extracted from the pickle file.
    """
    file_path = os.path.join(folder_path, subfolder,
                             f'train_codebook/{subfolder}_graph_prep_list_hierarchical_10_fixed.pkl')
    if not os.path.exists(file_path):
        return []

    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        dataset = [list(d['cluster_id2normalized_bldg_layout_bldg_bbox_list'].values()) for d in data if
                   'cluster_id2normalized_bldg_layout_bldg_bbox_list' in d]
        dataset = [cluster for boundary in dataset for cluster in boundary]
        return dataset
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
        return []


class ClusterLayoutDataset(Dataset):
    def __init__(self, data_type='train', user_name='ssw03270'):
        """
        Initializes an instance of the ClusterLayoutDataset class.

        Parameters:
        - data_type (str): Specifies the type of the data ('train', 'test', 'val'), which determines the folder from which data is loaded.
        """

        super(ClusterLayoutDataset, self).__init__()

        self.data_type = data_type

        if data_type == 'test':
            self.folder_path = f'Z:/iiixr-drive/Projects/2023_City_Team/000_2024CVPR/Our_dataset'
        else:
            self.folder_path = f'/data/{user_name}/datasets/CITY2024/Our_dataset'
        subfolders = [f for f in os.listdir(self.folder_path) if os.path.isdir(os.path.join(self.folder_path, f))][:]

        datasets = []

        # Create a partial function with fixed folder_path
        load_func = partial(load_pickle_file_with_cache, folder_path=self.folder_path)

        # Use ProcessPoolExecutor.map for ordered loading
        with ProcessPoolExecutor(max_workers=8) as executor:
            # Map returns results in the order of subfolders
            results = list(tqdm(executor.map(load_func, subfolders), total=len(subfolders), desc="Loading pickle files with caching"))
            for result in results:
                datasets += result

        
        # 패딩할 최대 빌딩 개수
        MAX_BUILDINGS = 10
        PADDING_BUILDING = [0, 0, 0, 0, 0, 0]
        # NumPy 배열로 변환
        # 우선 빌딩 개수가 다른 리스트를 객체 배열로 만듭니다
        data_np = np.empty(len(datasets), dtype=object)
        for i, cluster_boundary in enumerate(datasets):
            data_np[i] = np.array(cluster_boundary)
        # 패딩 수행
        padded_data_np = []
        for cluster_boundary in tqdm(data_np):
            current_building_count = cluster_boundary.shape[0]
            if current_building_count < MAX_BUILDINGS:
                padding_needed = MAX_BUILDINGS - current_building_count
                # 패딩할 배열 생성
                padding_array = np.array([PADDING_BUILDING] * padding_needed)
                # 패딩된 배열 결합
                cluster_boundary_padded = np.vstack((cluster_boundary, padding_array))
            else:
                # 빌딩 개수가 이미 최대인 경우 필요시 자름
                cluster_boundary_padded = cluster_boundary[:MAX_BUILDINGS]
            padded_data_np.append(cluster_boundary_padded)
        # 최종 배열로 변환 (모든 클러스터-바운더리가 동일한 크기를 가짐)
        final_padded_data = np.stack(padded_data_np)
        # Shuffle the pkl files to ensure random split

        shuffled_indices = np.random.permutation(final_padded_data.shape[0])
        final_padded_data_shuffled = final_padded_data[shuffled_indices]

        self.dataset = final_padded_data_shuffled

        # Compute the split sizes
        total_files = len(self.dataset)
        train_split = int(0.7 * total_files)
        val_split = int(0.2 * total_files)

        # Split the dataset
        if data_type == 'train':
            self.dataset = self.dataset[:train_split]
        elif data_type == 'val':
            self.dataset = self.dataset[train_split:train_split + val_split]
        elif data_type == 'test':
            self.dataset = self.dataset[train_split + val_split:]

        self.data_length = len(self.dataset)
        print(self.data_length)

    def __getitem__(self, idx):
        """
        Retrieves a single item from the dataset at the specified index.

        Parameters:
        - idx (int): The index of the item to retrieve.

        Returns:
        - dict: A dictionary containing the padded matrices and padding masks for boundary adjacency matrix, building adjacency matrix, and boundary-building adjacency matrix, as well as the boundary positions and the number of boundaries and buildings. For test data, it also returns the filename of the loaded pickle file.
        """
        data = self.dataset[idx] # seq, 6

        return torch.tensor(data, dtype=torch.float32, requires_grad=True)
    
        discrete_data = data.copy()
        discrete_data[:, :5] = np.floor(data[:, :5] * 63).astype(int)
        discrete_data[:, :5] = np.clip(discrete_data[:, :5], 0, 63)

        bbox_labels = discrete_data[:, :5]  # x, y, w, h, r
        category_labels = discrete_data[:, 5][:, np.newaxis]  # c

        # Create mask: 1 for actual tokens, 0 for pad tokens
        mask = (discrete_data[:, 5] == 1).astype(np.float32)[:, np.newaxis]  # shape: (MAX_BUILDINGS, 1)

        return {
            "bbox_labels": torch.tensor(bbox_labels, dtype=torch.long),
            "category_labels": torch.tensor(category_labels, dtype=torch.float32),
            "mask": torch.tensor(mask, dtype=torch.float32)
        }


    def __len__(self):
        """
        Returns the total number of items in the dataset.

        Returns:
        - int: The total number of items in the dataset.
        """

        return self.data_length
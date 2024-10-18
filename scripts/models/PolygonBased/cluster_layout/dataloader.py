import torch
from torch.utils.data import Dataset
import numpy as np
import os
import pickle
import random
from tqdm import tqdm

class ClusterLayoutDataset(Dataset):
    def __init__(self, data_type='train'):
        """
        Initializes an instance of the ClusterLayoutDataset class.

        Parameters:
        - data_type (str): Specifies the type of the data ('train', 'test', 'val'), which determines the folder from which data is loaded.
        """

        super(ClusterLayoutDataset, self).__init__()

        self.data_type = data_type

        self.folder_path = f'F:/City_Team/COHO/data_with_cluster'
        self.folder_path = f'/data/ssw03270/datasets/CITY2024/COHO_dataset'
        subfolders = [f for f in os.listdir(self.folder_path) if os.path.isdir(os.path.join(self.folder_path, f))]
        datasets = []
        for subfolder in tqdm(subfolders):
            file_path = os.path.join(self.folder_path, subfolder, f'graph/{subfolder}_graph_prep_list_with_clusters_detail.pkl')
            if not os.path.exists(file_path):
                continue

            with open(file_path, 'rb') as f:
                data = pickle.load(f)
            dataset = [list(d['cluster_id2trans_layout_list'].values()) for d in data if 'cluster_id2trans_layout_list' in d]
            dataset = [cluster for boundary in dataset for cluster in boundary]
            datasets += dataset
            break
        
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
        data = self.dataset[idx]
            
        return torch.tensor(data, dtype=torch.float32, requires_grad=True)


    def __len__(self):
        """
        Returns the total number of items in the dataset.

        Returns:
        - int: The total number of items in the dataset.
        """

        return self.data_length
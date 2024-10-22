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
    image_mask_file_path = os.path.join(folder_path, subfolder, f'train_generator/{subfolder}_graph_prep_list_with_clusters_detail.pkl')
    if not os.path.exists(image_mask_file_path):
        return {
            "cluster_img_mask": [],
            "cluster_encoding_indices": []
        }
    encoding_indices_file_path = os.path.join(folder_path, subfolder, f'codebook_indices/{subfolder}_graph_prep_list_with_clusters_detail.pkl')
    if not os.path.exists(encoding_indices_file_path):
        return {
            "cluster_img_mask": [],
            "cluster_encoding_indices": []
        }
    
    try:
        with open(image_mask_file_path, 'rb') as f:
            data = pickle.load(f)

        cluster_img_mask = [list(d['cluster_id2cluster_mask'].values()) for d in data if 'cluster_id2cluster_mask' in d]
        cluster_img_mask = [boundary for boundary in cluster_img_mask]

        with open(encoding_indices_file_path, 'rb') as f:
            data = pickle.load(f)

        cluster_encoding_indices = [list(d['cluster_id2encoding_indices'].values()) for d in data if 'cluster_id2encoding_indices' in d]
        cluster_encoding_indices = [boundary for boundary in cluster_encoding_indices]

        return {
            "cluster_img_mask": cluster_img_mask,
            "cluster_encoding_indices": cluster_encoding_indices
        }
    except Exception as e:
        print(f"Error loading {image_mask_file_path} and {encoding_indices_file_path}: {e}")
        return []


class DiffusionGeneratorDataset(Dataset):
    def __init__(self, data_type='train', user_name='ssw03270'):
        """
        Initializes an instance of the DiffusionGeneratorDataset class.

        Parameters:
        - data_type (str): Specifies the type of the data ('train', 'test', 'val'), which determines the folder from which data is loaded.
        """

        super(DiffusionGeneratorDataset, self).__init__()

        self.data_type = data_type

        if data_type == 'test':
            self.folder_path = f'Z:/iiixr-drive/Projects/2023_City_Team/000_2024CVPR/Our_dataset'
        else:
            self.folder_path = f'/data/{user_name}/datasets/CITY2024/Our_dataset'
            self.folder_path = f'Z:/iiixr-drive/Projects/2023_City_Team/000_2024CVPR/Our_dataset'
        subfolders = [f for f in os.listdir(self.folder_path) if os.path.isdir(os.path.join(self.folder_path, f))][:1]

        cluster_img_mask_datasets = []
        cluster_encoding_indices_datasets = []

        # Create a partial function with fixed folder_path
        load_func = partial(load_pickle_file_with_cache, folder_path=self.folder_path)

        # Use ProcessPoolExecutor.map for ordered loading
        with ProcessPoolExecutor(max_workers=8) as executor:
            # Map returns results in the order of subfolders
            results = list(tqdm(executor.map(load_func, subfolders), total=len(subfolders), desc="Loading pickle files with caching"))
            for result in results:
                cluster_img_mask = result['cluster_img_mask']
                cluster_encoding_indices = result['cluster_encoding_indices']

                cluster_img_mask_datasets += cluster_img_mask                   # batch, cluster_count, feature_count
                cluster_encoding_indices_datasets += cluster_encoding_indices   # batch, cluster_count, image_size, image_size

        MAX_CLUSTERS = 22 + 2
        PADDING_MASK = np.zeros((len(cluster_img_mask_datasets[0][0]), len(cluster_img_mask_datasets[0][0][0]))).tolist()
        PADDING_INDICES = [17] * len(cluster_encoding_indices_datasets[0][0])

        indices_data_np = np.empty(len(cluster_encoding_indices_datasets), dtype=object)
        for i, cluster_encoding_indices in enumerate(cluster_encoding_indices_datasets):
            indices_data_np[i] = np.array(cluster_encoding_indices)
        # 패딩 수행
        padded_indices_data_np = []
        for cluster_encoding_indices in tqdm(indices_data_np):
            current_count = cluster_encoding_indices.shape[0]
            if current_count < MAX_CLUSTERS:
                padding_needed = MAX_CLUSTERS - current_count
                # 패딩할 배열 생성
                padding_array = np.array([PADDING_INDICES] * padding_needed)
                # 패딩된 배열 결합
                encoding_indices_padded = np.vstack((cluster_encoding_indices, padding_array))
            else:
                # 빌딩 개수가 이미 최대인 경우 필요시 자름
                encoding_indices_padded = cluster_encoding_indices[:MAX_CLUSTERS]
            padded_indices_data_np.append(encoding_indices_padded)

        # 최종 배열로 변환 (모든 클러스터-바운더리가 동일한 크기를 가짐)
        final_indices_padded_data = np.stack(padded_indices_data_np)
        final_indices_padded_data = np.reshape(final_indices_padded_data, (final_indices_padded_data.shape[0], -1))

        mask_data_np = np.empty(len(cluster_img_mask_datasets), dtype=object)
        for i, cluster_img_mask in enumerate(cluster_img_mask_datasets):
            mask_data_np[i] = np.array(cluster_img_mask)
        # 패딩 수행
        padded_mask_data_np = []
        for cluster_mask in tqdm(mask_data_np):
            current_count = cluster_mask.shape[0]
            if current_count < MAX_CLUSTERS:
                padding_needed = MAX_CLUSTERS - current_count
                # 패딩할 배열 생성
                padding_array = np.array([PADDING_MASK] * padding_needed)
                # 패딩된 배열 결합
                mask_padded = np.vstack((cluster_mask, padding_array))
            else:
                # 빌딩 개수가 이미 최대인 경우 필요시 자름
                mask_padded = cluster_mask[:MAX_CLUSTERS]
            
            blk_mask = np.where(np.any(mask_padded == 1, axis=0), 1, 0)
            blk_mask = blk_mask[np.newaxis, :, :]
            mask_padded = np.concatenate((blk_mask, mask_padded), axis=0)[:MAX_CLUSTERS]

            padded_mask_data_np.append(mask_padded)


        # 최종 배열로 변환 (모든 클러스터-바운더리가 동일한 크기를 가짐)
        final_mask_padded_data = np.stack(padded_mask_data_np)

        # Shuffle the pkl files to ensure random split
        shuffled_indices = np.random.permutation(final_indices_padded_data.shape[0])
        final_indices_padded_data_shuffled = final_indices_padded_data[shuffled_indices]
        final_mask_padded_data_shuffled = final_mask_padded_data[shuffled_indices]
        self.indices_dataset = final_indices_padded_data_shuffled
        self.masks_dataset = final_mask_padded_data_shuffled

        # Compute the split sizes
        total_files = len(self.indices_dataset)
        train_split = int(0.7 * total_files)
        val_split = int(0.2 * total_files)

        # Split the dataset
        if data_type == 'train':
            self.indices_dataset = self.indices_dataset[:train_split]
            self.masks_dataset = self.masks_dataset[:train_split]
        elif data_type == 'val':
            self.indices_dataset = self.indices_dataset[train_split:train_split + val_split]
            self.masks_dataset = self.masks_dataset[train_split:train_split + val_split]
        elif data_type == 'test':
            self.indices_dataset = self.indices_dataset[train_split + val_split:]
            self.masks_dataset = self.masks_dataset[train_split + val_split:]

        self.data_length = len(self.indices_dataset)
        print(self.data_length)

    def __getitem__(self, idx):
        """
        Retrieves a single item from the dataset at the specified index.

        Parameters:
        - idx (int): The index of the item to retrieve.

        Returns:
        - dict: A dictionary containing the padded matrices and padding masks for boundary adjacency matrix, building adjacency matrix, and boundary-building adjacency matrix, as well as the boundary positions and the number of boundaries and buildings. For test data, it also returns the filename of the loaded pickle file.
        """
        indices_data = self.indices_dataset[idx]
        masks_data = self.masks_dataset[idx]
            
        return {
            "indices_data": torch.tensor(indices_data, dtype=torch.long),
            "masks_data": torch.tensor(masks_data, dtype=torch.float32, requires_grad=True),
            }


    def __len__(self):
        """
        Returns the total number of items in the dataset.

        Returns:
        - int: The total number of items in the dataset.
        """

        return self.data_length
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
    file_path = os.path.join(folder_path, subfolder, f'train_codebook/{subfolder}_graph_prep_list_with_clusters_detail.pkl')
    if not os.path.exists(file_path):
        return []
    
    try:
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
        dataset = [list(d['cluster_id2padded_normalized_bldg_layout_cluster_list'].values()) for d in data if 'cluster_id2padded_normalized_bldg_layout_cluster_list' in d]
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
        subfolders = [f for f in os.listdir(self.folder_path) if os.path.isdir(os.path.join(self.folder_path, f))]

        datasets = []

        # Create a partial function with fixed folder_path
        load_func = partial(load_pickle_file_with_cache, folder_path=self.folder_path)

        # Use ProcessPoolExecutor.map for ordered loading
        with ProcessPoolExecutor(max_workers=8) as executor:
            # Map returns results in the order of subfolders
            results = list(tqdm(executor.map(load_func, subfolders), total=len(subfolders), desc="Loading pickle files with caching"))
            for result in results:
                datasets += result

        for data in datasets:
            if len(data) != 10:
                print(len(data))
        datasets = np.array(datasets)
        # Shuffle the pkl files to ensure random split
        shuffled_indices = np.random.permutation(datasets.shape[0])
        datasets_shuffled = datasets[shuffled_indices]
        self.dataset = datasets_shuffled

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
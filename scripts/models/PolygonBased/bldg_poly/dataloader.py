import torch
from torch.utils.data import Dataset
import numpy as np
import os
import pickle
import random

class BuildingDataset(Dataset):
    def __init__(self, data_type='train'):
        """
        Initializes an instance of the BuildingDataset class.

        Parameters:
        - data_type (str): Specifies the type of the data ('train', 'test', 'val'), which determines the folder from which data is loaded.
        """

        super(BuildingDataset, self).__init__()

        self.data_type = data_type

        self.folder_path = f'E:/Resources/COHO/NormalizedBuildings/Abilene'

        # Shuffle the pkl files to ensure random split
        pkl_files = [f for f in os.listdir(self.folder_path) if f.endswith('.pkl')]
        random.shuffle(pkl_files)

        # Compute the split sizes
        total_files = len(pkl_files)
        train_split = int(0.7 * total_files)
        val_split = int(0.2 * total_files)

        # Split the dataset
        if data_type == 'train':
            self.pkl_files = pkl_files[:train_split]
        elif data_type == 'val':
            self.pkl_files = pkl_files[train_split:train_split + val_split]
        elif data_type == 'test':
            self.pkl_files = pkl_files[train_split + val_split:]

        self.data_length = len(self.pkl_files)
        print(self.data_length)

    def pad_matrix(self, matrix, pad_shape):
        """
        Pads a given matrix to the specified shape and returns the padded matrix along with a padding mask.

        Parameters:
        - matrix (np.ndarray): The matrix to be padded.
        - pad_shape (tuple): The desired shape of the padded matrix.

        Returns:
        - padded_matrix (np.ndarray): The matrix padded to the desired shape.
        - pad_mask (np.ndarray): A mask indicating the original data (0s for padded regions and 1s for original data).
        """
        matrix = np.array(matrix)
        original_shape = matrix.shape
        pad_width = ((0, max(0, pad_shape[0] - original_shape[0])),
                     (0, max(0, pad_shape[1] - original_shape[1])))
        padded_matrix = np.pad(matrix, pad_width=pad_width, mode='constant', constant_values=0)

        pad_mask = np.zeros_like(padded_matrix)
        pad_mask[:original_shape[0], :original_shape[1]] = 1

        return padded_matrix, pad_mask

    def __getitem__(self, idx):
        """
        Retrieves a single item from the dataset at the specified index.

        Parameters:
        - idx (int): The index of the item to retrieve.

        Returns:
        - dict: A dictionary containing the padded matrices and padding masks for boundary adjacency matrix, building adjacency matrix, and boundary-building adjacency matrix, as well as the boundary positions and the number of boundaries and buildings. For test data, it also returns the filename of the loaded pickle file.
        """
        n_node = 16
        load_path = self.folder_path + '/' + self.pkl_files[idx]
        with open(load_path, 'rb') as f:
            self.data = pickle.load(f)
            
        bldg_coords = self.data['bldg_coords']
        grid_corner_coords = self.data['grid_corner_coords']
        grid_corner_indices = self.data['grid_corner_indices']

        return {
            'bldg_coords': torch.tensor(bldg_coords, dtype=torch.float32, requires_grad=True),
            'corner_coords': torch.tensor(grid_corner_coords, dtype=torch.float32, requires_grad=True).reshape(n_node, -1),
            'corner_indices': torch.tensor(grid_corner_indices, dtype=torch.int32)
        }


    def __len__(self):
        """
        Returns the total number of items in the dataset.

        Returns:
        - int: The total number of items in the dataset.
        """

        return self.data_length
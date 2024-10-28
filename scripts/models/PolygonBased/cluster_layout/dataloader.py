import torch
from torch.utils.data import Dataset
import numpy as np
import os
import pickle
import glob
from tqdm import tqdm

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

class ClusterLayoutDataset(Dataset):
    def __init__(self, data_type='train', user_name='ssw03270', coords_type='continuous', norm_type="bldg_bbox"):
        """
        ClusterLayoutDataset 클래스의 인스턴스를 초기화합니다.

        매개변수:
        - data_type (str): 데이터의 종류를 지정합니다 ('train', 'test', 'val').
        """

        super(ClusterLayoutDataset, self).__init__()

        self.data_type = data_type
        self.coords_type = coords_type
        self.norm_type = norm_type

        if data_type == 'test':
            self.folder_path = f'Z:/iiixr-drive/Projects/2023_City_Team/000_2024CVPR/Our_dataset'
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

        # 필요한 키만 메모리에 적재
        self.data_list = []
        for file_path in tqdm(self.pkl_files, desc="데이터를 메모리에 적재 중"):
            try:
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                    # 필요한 키만 추출
                    if self.norm_type == "bldg_bbox":
                        regions = data['cluster_id2cluster_bldg_bbox']
                        layouts = data['cluster_id2normalized_bldg_layout_bldg_bbox_list']
                    elif self.norm_type == "cluster":
                        regions = data['cluster_id2cluster_bbox']
                        layouts = data['cluster_id2normalized_bldg_layout_cluster_list']
                    elif self.norm_type == "blk":
                        regions = None
                        layouts = data['cluster_id2normalized_bldg_layout_blk_list']
                    else:
                        regions = None
                        layouts = None

                    # 필요한 데이터만 저장
                    self.data_list.append({
                        'regions': regions,
                        'layouts': layouts
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
        data_item = self.data_list[idx]
        regions = data_item['regions']
        layouts = data_item['layouts']

        MAX_BUILDINGS = 10
        PADDING_BUILDING = [0, 0, 0, 0, 0, 0]

        bldg_layout_list = []
        min_coords_list = []
        range_max_list = []

        for cluster_id, bldg_layouts in layouts.items():
            if regions is not None:
                cluster_region = regions[cluster_id]
                _, min_coords, range_max, _ = normalize_coords_uniform(cluster_region.exterior.coords)
                min_coords_list.append(min_coords)
                range_max_list.append([range_max])

            bldg_layouts = np.array(bldg_layouts)
            current_building_count = bldg_layouts.shape[0]
            if current_building_count < MAX_BUILDINGS:
                padding_needed = MAX_BUILDINGS - current_building_count
                padding_array = np.array([PADDING_BUILDING] * padding_needed)
                bldg_layouts = np.vstack((bldg_layouts, padding_array))
            else:
                bldg_layouts = bldg_layouts[:MAX_BUILDINGS]

            bldg_layout_list.append(bldg_layouts)

        bldg_layout_list = np.array(bldg_layout_list)
        min_coords_list = np.array(min_coords_list)
        range_max_list = np.array(range_max_list)

        if self.coords_type == 'continuous':
            return (torch.tensor(bldg_layout_list, dtype=torch.float32),
                    torch.tensor(min_coords_list, dtype=torch.float32),
                    torch.tensor(range_max_list, dtype=torch.float32))

        elif self.coords_type == 'discrete':
            bldg_layout_list[:, :5] = np.floor(bldg_layout_list[:, :5] * 63).astype(int)
            bldg_layout_list[:, :5] = np.clip(bldg_layout_list[:, :5], 0, 63)

            return (torch.tensor(bldg_layout_list, dtype=torch.long),
                    torch.tensor(min_coords_list, dtype=torch.float32),
                    torch.tensor(range_max_list, dtype=torch.float32))

    def __len__(self):
        """
        데이터셋의 총 아이템 수를 반환합니다.

        반환값:
        - int: 데이터셋의 총 아이템 수.
        """

        return self.data_length


# import torch
# from torch.utils.data import Dataset
# import numpy as np
# import os
# import pickle
# import glob
# from tqdm import tqdm
# from concurrent.futures import ProcessPoolExecutor, as_completed
# from functools import partial
# from shapely.geometry import Polygon, box
# from shapely.affinity import rotate

# def normalize_coords_uniform(coords, min_coords=None, range_max=None):
#     if min_coords is not None and range_max is not None:
#         normalized_coords = (coords - min_coords) / range_max
#     else:
#         coords = np.array(coords)
#         min_coords = coords.min(axis=0)
#         max_coords = coords.max(axis=0)
#         range_max = (max_coords - min_coords).max()
#         normalized_coords = (coords - min_coords) / range_max

#     out_of_bounds = []
#     # 정규화된 좌표가 0과 1 사이에 있는지 확인
#     if not (np.all(normalized_coords >= -1) and np.all(normalized_coords <= 2)):
#         print("경고: 정규화된 좌표 중 일부가 0과 1 사이에 있지 않습니다.")
#         # 추가로, 어떤 좌표가 범위를 벗어났는지 출력할 수 있습니다.
#         out_of_bounds = normalized_coords[(normalized_coords < -1) | (normalized_coords > 2)]
#         print("범위를 벗어난 좌표 값:", out_of_bounds)

#     return normalized_coords, min_coords, [range_max], out_of_bounds

# class ClusterLayoutDataset(Dataset):
#     def __init__(self, data_type='train', user_name='ssw03270', coords_type='continuous', norm_type="bldg_bbox"):
#         """
#         Initializes an instance of the ClusterLayoutDataset class.

#         Parameters:
#         - data_type (str): Specifies the type of the data ('train', 'test', 'val'), which determines the folder from which data is loaded.
#         """

#         super(ClusterLayoutDataset, self).__init__()

#         self.data_type = data_type
#         self.coords_type = coords_type
#         self.norm_type = norm_type

#         if data_type == 'test':
#             self.folder_path = f'Z:/iiixr-drive/Projects/2023_City_Team/000_2024CVPR/Our_dataset'
#         else:
#             # self.folder_path = f'/data/{user_name}/datasets/CITY2024/Our_dataset'
#             self.folder_path = f'/data2/local_datasets/CITY2024/Our_dataset_divided_without_segmentation_mask'
#         self.pkl_files = glob.glob(os.path.join(self.folder_path, '**', '*.pkl'), recursive=True)
#         self.pkl_files = np.array(self.pkl_files)

#         shuffled_indices = np.random.permutation(self.pkl_files.shape[0])
#         self.pkl_files = self.pkl_files[shuffled_indices]

#         # Compute the split sizes
#         total_files = len(self.pkl_files)
#         train_split = int(0.7 * total_files)
#         val_split = int(0.2 * total_files)

#         # Split the dataset
#         if data_type == 'train':
#             self.pkl_files = self.pkl_files[:train_split]
#         elif data_type == 'val':
#             self.pkl_files = self.pkl_files[train_split:train_split + val_split]
#         elif data_type == 'test':
#             self.pkl_files = self.pkl_files[train_split + val_split:]

#         self.data_length = len(self.pkl_files)
#         print(self.data_length)

#     def __getitem__(self, idx):
#         """
#         Retrieves a single item from the dataset at the specified index.

#         Parameters:
#         - idx (int): The index of the item to retrieve.

#         Returns:
#         - dict: A dictionary containing the padded matrices and padding masks for boundary adjacency matrix, building adjacency matrix, and boundary-building adjacency matrix, as well as the boundary positions and the number of boundaries and buildings. For test data, it also returns the filename of the loaded pickle file.
#         """
#         file_path = self.pkl_files[idx]
#         try:
#             with open(file_path, 'rb') as f:
#                 data = pickle.load(f)
#         except EOFError:
#             print(f"EOFError: Failed to load {file_path}. The file may be corrupted or incomplete.")
#             return None  # 무효한 항목을 나타내는 None 반환
#         except Exception as e:
#             print(f"Error loading {file_path}: {e}")
#             return None  # 무효한 항목을 나타내는 None 반환
        
#         if self.norm_type == "bldg_bbox":
#             regions = data['cluster_id2cluster_bldg_bbox']
#             layouts = data['cluster_id2normalized_bldg_layout_bldg_bbox_list']
#         elif self.norm_type == "cluster":
#             regions = data['cluster_id2cluster_bbox']
#             layouts = data['cluster_id2normalized_bldg_layout_cluster_list']
#         elif self.norm_type == "blk":
#             regions = None
#             layouts = data['cluster_id2normalized_bldg_layout_blk_list']

#         MAX_BUILDINGS = 10
#         PADDING_BUILDING = [0, 0, 0, 0, 0, 0]

#         bldg_layout_list = []
#         min_coords_list = []
#         range_max_list = []

#         for cluster_id, bldg_layouts in layouts.items():
#             if regions is not None:
#                 cluster_region = regions[cluster_id]
#                 _, min_coords, range_max, _ = normalize_coords_uniform(cluster_region.exterior.coords)
#                 min_coords_list.append(min_coords)
#                 range_max_list.append([range_max])

#             bldg_layouts = np.array(bldg_layouts)
#             current_building_count = bldg_layouts.shape[0]
#             if current_building_count < MAX_BUILDINGS:
#                 padding_needed = MAX_BUILDINGS - current_building_count
#                 padding_array = np.array([PADDING_BUILDING] * padding_needed)
#                 bldg_layouts = np.vstack((bldg_layouts, padding_array))
#             else:
#                 bldg_layouts = bldg_layouts[:MAX_BUILDINGS]

#             bldg_layout_list.append(bldg_layouts)

#         bldg_layout_list = np.array(bldg_layout_list)
#         min_coords_list = np.array(min_coords_list)
#         range_max_list = np.array(range_max_list)

#         if self.coords_type == 'continuous':
#             return torch.tensor(bldg_layout_list, dtype=torch.float32), torch.tensor(min_coords_list, dtype=torch.float32), torch.tensor(range_max_list, dtype=torch.float32)
        
#         elif self.coords_type == 'discrete':
#             bldg_layout_list[:, :5] = np.floor(bldg_layout_list[:, :5] * 63).astype(int)
#             bldg_layout_list[:, :5] = np.clip(bldg_layout_list[:, :5], 0, 63)

#             return torch.tensor(bldg_layout_list, dtype=torch.long), torch.tensor(min_coords_list, dtype=torch.float32), torch.tensor(range_max_list, dtype=torch.float32)

#     def __len__(self):
#         """
#         Returns the total number of items in the dataset.

#         Returns:
#         - int: The total number of items in the dataset.
#         """

#         return self.data_length
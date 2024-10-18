import os
import pickle
from shapely.geometry import Polygon
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm

city_list = [
    "Atlanta", "Boston", "Dallas", "Denver", "Houston", "Lasvegas",
    "Littlerock", "Miami", "NewOrleans", "Philadelphia", "Phoenix",
    "Pittsburgh", "Portland", "Providence", "Richmond", "Saintpaul",
    "Sanfrancisco", "Seattle", "Washington"
]

UNIT_LENGTH = 10

def remove_overlapping_segments(boundary_coords, unit_length):
    i = 0
    while i < len(boundary_coords):
        x1, y1 = boundary_coords[i]
        x2, y2 = boundary_coords[(i + 1) % len(boundary_coords)]

        distance = np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2]))
        num_segments = int(distance // unit_length)

        if num_segments == 0:
            del boundary_coords[(i + 1) % len(boundary_coords)]
        else:
            i += 1

    return boundary_coords

def divide_boundary_into_segments(boundary_coords, unit_length):
    new_boundary_coords = []

    for i in range(len(boundary_coords)):
        x1, y1 = boundary_coords[i]
        x2, y2 = boundary_coords[(i + 1) % len(boundary_coords)]

        # 두 점 사이의 거리 계산
        distance = np.linalg.norm(np.array([x1, y1]) - np.array([x2, y2]))

        # 두 점 사이의 세그먼트 수 계산
        num_segments = int(distance // unit_length)

        # 시작점 추가
        new_boundary_coords.append([x1, y1])

        # 세그먼트를 단위 길이만큼 추가
        if num_segments > 0:
            vector = np.array([x2 - x1, y2 - y1]) / distance
            for j in range(1, num_segments):
                new_point = np.array([x1, y1]) + vector * unit_length * j
                new_boundary_coords.append(new_point.tolist())

    return new_boundary_coords

def visualize_boundary(before_coords, after_coords):
    # Create a copy to avoid modifying the original list
    before_copy = before_coords.copy()
    after_copy = after_coords.copy()

    before_copy.append(before_copy[0])  # Close the boundary for visualization
    after_copy.append(after_copy[0])  # Close the boundary for visualization

    # Convert lists to numpy arrays for easier handling
    before_coords = np.array(before_copy)
    after_coords = np.array(after_copy)

    # Create a figure with two subplots
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Plot before transformation
    axs[0].plot(before_coords[:, 0], before_coords[:, 1], marker='o')
    axs[0].set_title('Before Transformation')
    axs[0].set_aspect('equal')

    # Plot after transformation
    axs[1].plot(after_coords[:, 0], after_coords[:, 1], marker='o')
    axs[1].set_title('After Transformation')
    axs[1].set_aspect('equal')

    plt.show()

def remove_percentiles_paired(segment_data, building_data, file_paths, lower_percentile, upper_percentile):
    # Sort both arrays based on the segment data
    sorted_indices = np.argsort(segment_data)
    segment_data_sorted = np.array(segment_data)[sorted_indices]
    building_data_sorted = np.array(building_data)[sorted_indices]
    file_paths_sorted = np.array(file_paths)[sorted_indices]

    # Calculate the indices for the percentiles to remove
    lower_index = int(len(segment_data_sorted) * lower_percentile)
    upper_index = int(len(segment_data_sorted) * (1 - upper_percentile))

    # Apply the percentile filtering
    segment_filtered = segment_data_sorted[lower_index:upper_index]
    building_filtered = building_data_sorted[lower_index:upper_index]
    file_paths_filtered = file_paths_sorted[lower_index:upper_index]

    return segment_filtered, building_filtered, file_paths_filtered

def remove_percentiles_paired2(segment_data, building_data, file_paths, lower_percentile, upper_percentile):
    # Sort both arrays based on the building data
    sorted_indices = np.argsort(building_data)
    segment_data_sorted = np.array(segment_data)[sorted_indices]
    building_data_sorted = np.array(building_data)[sorted_indices]
    file_paths_sorted = np.array(file_paths)[sorted_indices]

    # Calculate the indices for the percentiles to remove
    lower_index = int(len(building_data_sorted) * lower_percentile)
    upper_index = int(len(building_data_sorted) * (1 - upper_percentile))

    # Apply the percentile filtering
    segment_filtered = segment_data_sorted[lower_index:upper_index]
    building_filtered = building_data_sorted[lower_index:upper_index]
    file_paths_filtered = file_paths_sorted[lower_index:upper_index]

    return segment_filtered, building_filtered, file_paths_filtered

segment_counts = []
building_counts = []
file_paths = []

for city in city_list:
    print(city)
    folder_path = f'./{city}/polygon/'
    all_files = os.listdir(folder_path)

    for idx in tqdm(range(len(all_files))):
        file_path = f'./{city}/polygon/block_{idx + 1}.pkl'
        with open(file_path, 'rb') as f:
            data = pickle.load(f)

        normalized_boundary = data['normalized_block_polygon']
        normalized_buildings = data['normalized_buildings_polygons']
        scale_factor = 0.9 / data['scale_factor']
        unit_length = UNIT_LENGTH / scale_factor

        boundary_coords = list(normalized_boundary.exterior.coords)
        modified_boundary_coords = remove_overlapping_segments(boundary_coords, unit_length)
        boundary_segments = divide_boundary_into_segments(modified_boundary_coords, unit_length)

        # visualize_boundary(boundary_coords, boundary_segments)

        # Store the number of segments for this boundary
        # if len(boundary_segments) <= 200 and len(normalized_buildings) > 2 and len(normalized_buildings) <= 100:
        if len(normalized_buildings) > 0:
            segment_counts.append(len(boundary_segments))
            building_counts.append(len(normalized_buildings))
            file_paths.append(file_path)

    # Remove the top and bottom 1% from both segment_counts and building_counts as a pair
segment_counts_filtered, building_counts_filtered, file_paths_filtered = remove_percentiles_paired(
    segment_counts, building_counts, file_paths, 0.01, 0.01)

segment_counts_filtered, building_counts_filtered, file_paths_filtered = remove_percentiles_paired2(
    segment_counts_filtered, building_counts_filtered, file_paths_filtered, 0.00, 0.01)

# Plot the filtered distributions
fig, axs = plt.subplots(1, 2, figsize=(14, 6))

# Plot filtered segment counts distribution
axs[0].hist(segment_counts_filtered, bins=100, edgecolor='black')
axs[0].set_title('Filtered Distribution of Boundary Segment Counts')
axs[0].set_xlabel('Number of Segments')
axs[0].set_ylabel('Frequency')

# Plot filtered building counts distribution
axs[1].hist(building_counts_filtered, bins=100, edgecolor='black')
axs[1].set_title('Filtered Distribution of Building Counts')
axs[1].set_xlabel('Number of Buildings')
axs[1].set_ylabel('Frequency')

plt.tight_layout()
plt.show()

print(file_paths)
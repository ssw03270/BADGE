import os
import pickle
from tqdm import tqdm
import multiprocessing
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Define the dataset path
dataset_path = "Z:/iiixr-drive/Projects/2023_City_Team/000_2024CVPR/COHO_dataset"

# Initialize debug flag
debug = False

def process_folder(folder):
    """
    Process a single folder to extract cluster counts.

    Args:
        folder (str): The name of the folder to process.

    Returns:
        dict or None: A dictionary with folder name as key and list of cluster counts as value,
                      or None if the data file does not exist.
    """
    data_path = os.path.join(dataset_path, folder, f'graph/{folder}_graph_prep_list_with_clusters.pkl')
    if not os.path.exists(data_path):
        print(f"{data_path} 파일이 존재하지 않습니다. 건너뜁니다.")
        return None  # Return None if the file doesn't exist

    with open(data_path, 'rb') as f:
        try:
            data_list = pickle.load(f)
        except Exception as e:
            print(f"Error loading {data_path}: {e}")
            return None

    print(f"{folder} 파일을 처리하는 중입니다.")
    cluster_count = []
    for data in tqdm(data_list, desc=f"Processing {folder}"):
        hierarchical_clustering_k_10_debug = data.get('hierarchical_clustering_k_10_debug', [])
        cluster_count.append(len(hierarchical_clustering_k_10_debug))
    return {folder: cluster_count}  # Return a dictionary for clarity

def main():
    """
    Main function to process all folders in parallel and collect results.

    Returns:
        list: A list of dictionaries containing cluster counts per folder.
    """
    # Get list of subfolders
    subfolders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
    results = []
    
    # Determine the number of processes (you can adjust this as needed)
    num_processes = min(4, os.cpu_count() or 1)
    
    with multiprocessing.Pool(processes=num_processes) as pool:
        # Use imap_unordered for potentially better performance and to handle results as they come in
        for result in pool.imap_unordered(process_folder, subfolders):
            if result is not None:
                results.append(result)
    return results

if __name__ == "__main__":
    # It's good practice to redefine subfolders inside the main block to avoid issues on Windows
    subfolders = [f for f in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, f))]
    all_results = main()
    
    # Example: Print the results
    for folder_result in all_results:
        for folder, counts in folder_result.items():
            print(f"Folder: {folder}, Cluster Counts: {counts}")
    
    # Aggregate all cluster counts into a single list
    aggregated_counts = []
    for folder_result in all_results:
        for counts in folder_result.values():
            aggregated_counts.extend(counts)
    
    # Check if there are any counts to process
    if not aggregated_counts:
        print("No cluster counts available to analyze.")
    else:
        # Convert to numpy array for easier computation
        counts_array = np.array(aggregated_counts)
        
        # Calculate statistics
        max_count = counts_array.max()
        min_count = counts_array.min()
        mean_count = counts_array.mean()
        
        print("\nCluster Counts Statistics:")
        print(f"Maximum Cluster Count: {max_count}")
        print(f"Minimum Cluster Count: {min_count}")
        print(f"Mean Cluster Count: {mean_count:.2f}")
        
        # Plot the distribution using seaborn
        plt.figure(figsize=(10, 6))
        sns.histplot(counts_array, bins=30, kde=True, color='skyblue')
        plt.title('Distribution of Cluster Counts')
        plt.xlabel('Cluster Count')
        plt.ylabel('Frequency')
        plt.grid(True)
        plt.tight_layout()
        plt.show()
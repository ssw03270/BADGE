import torch
from torch.utils.data import DataLoader
from torchvision import models, transforms
import numpy as np
import os
import pickle
import glob
from tqdm import tqdm
import argparse
from tqdm import tqdm
from retriever_dataloader import BoundaryBlkDataset
import faiss
from accelerate.utils import set_seed

def build_faiss_index(latent_vectors, use_gpu=False):
    d = latent_vectors.shape[1]
    index = faiss.IndexFlatIP(d)  # Inner Product for cosine similarity
    faiss.normalize_L2(latent_vectors)
    if use_gpu:
        res = faiss.StandardGpuResources()
        index = faiss.index_cpu_to_gpu(res, 0, index)
    index.add(latent_vectors)
    print(f"FAISS index built with {index.ntotal} vectors.")
    return index

def perform_retrieval(D, I, file_paths, k=5):
    retrieval_dict = {}

    for idx, (distances, indices) in enumerate(zip(D, I)):
        query_path = file_paths[idx]
        retrieved_paths = []

        for distance, neighbor_idx in zip(distances, indices):
            if neighbor_idx == idx:
                continue  # Skip self
            retrieved_paths.append(file_paths[neighbor_idx])
            if len(retrieved_paths) == k:
                break

        retrieval_dict[query_path] = retrieved_paths

    return retrieval_dict

def main():
    parser = argparse.ArgumentParser(description='Inference for the Transformer model.')
    parser.add_argument("--coords_type", type=str, default="continuous", help="coordinate type")
    parser.add_argument("--norm_type", type=str, default="blk", help="coordinate type")
    args = parser.parse_args()

    set_seed(42)
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    # Initialize dataset and DataLoader
    dataset = BoundaryBlkDataset(data_type='test', coords_type=args.coords_type, norm_type=args.norm_type)
    batch_size = 1024
    num_workers = 4
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    # Load and modify ResNet-18
    resnet18 = models.resnet18(pretrained=True)
    modules = list(resnet18.children())[:-1]  # Remove the last FC layer
    model = torch.nn.Sequential(*modules)
    model = model.to(device)
    model.eval()

    # Prepare storage
    latent_vectors = []
    file_paths = []

    # Inference loop
    with torch.no_grad():
        for batch in tqdm(data_loader, desc="Encoding images"):
            pkl_files, image_masks = batch
            image_masks = image_masks.to(device)
            
            features = model(image_masks)
            features = features.view(features.size(0), -1)
            features_np = features.cpu().numpy()
            latent_vectors.append(features_np)
            file_paths.extend(pkl_files)
    
    latent_vectors = np.concatenate(latent_vectors, axis=0)
    print(f'Extracted latent vectors shape: {latent_vectors.shape}')
    print(f"File paths length: {len(file_paths)}")

    # Build FAISS index
    use_gpu_faiss = True  # Set to True if you have a GPU and installed faiss-gpu
    index = build_faiss_index(latent_vectors, use_gpu=use_gpu_faiss)

    # Perform batch similarity search
    k = 5  # Number of nearest neighbors
    D, I = index.search(latent_vectors, k + 1)  # k + 1 to account for self-match

    # Build the retrieval dictionary
    retrieval_dict = perform_retrieval(D, I, file_paths, k=k)

    # Display a sample from the dictionary
    sample_key = list(retrieval_dict.keys())[0]
    sample_retrieved = retrieval_dict[sample_key]
    print(f"\nSample retrieval for {sample_key}:")
    for rank, path in enumerate(sample_retrieved, start=1):
        print(f"  Rank {rank}: {path}")

    # Save the retrieval dictionary
    dict_save_path = 'retrieval_dict.pkl'
    with open(dict_save_path, 'wb') as f:
        pickle.dump(retrieval_dict, f)
    print(f"\nRetrieval dictionary saved to {dict_save_path}")

    # Optionally, save the FAISS index for future use
    index_path = 'faiss_index.bin'
    if use_gpu_faiss:
        # Transfer the GPU index to CPU before saving
        index_cpu = faiss.index_gpu_to_cpu(index)
        faiss.write_index(index_cpu, index_path)
    else:
        # If already on CPU, save directly
        faiss.write_index(index, index_path)
    print(f"FAISS index saved to {index_path}")

if __name__ == "__main__":
    main()
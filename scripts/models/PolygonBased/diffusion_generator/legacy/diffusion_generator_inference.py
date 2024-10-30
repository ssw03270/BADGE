import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm.auto import tqdm

from dataloader import DiffusionGeneratorDataset
from transformer import Transformer

import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import seaborn as sns

def visualization_codebook(model, perplexity_tsne=30, n_iter=1000, random_state=42):
    with torch.no_grad():
        # Extract embedding weights from VectorQuantizer
        embedding_weights = model.vq.embedding.weight.cpu().numpy()  # Shape: (num_embeddings, embedding_dim)
    
    print(f"Embedding shape: {embedding_weights.shape}")

    # Apply t-SNE to reduce dimensionality to 2D
    tsne = TSNE(n_components=2, perplexity=perplexity_tsne, n_iter=n_iter, random_state=random_state)
    embeddings_2d = tsne.fit_transform(embedding_weights)

    print(f"t-SNE completed. Shape of reduced embeddings: {embeddings_2d.shape}")

    # Plot the embeddings
    plt.figure(figsize=(10, 10))
    sns.scatterplot(x=embeddings_2d[:,0], y=embeddings_2d[:,1], s=50, alpha=0.7)

    # Optionally, annotate a subset of points to avoid clutter
    num_annotations = min(100, embeddings_2d.shape[0])  # Adjust number as needed
    indices = np.linspace(0, embeddings_2d.shape[0]-1, num_annotations, dtype=int)
    for i in indices:
        plt.text(embeddings_2d[i,0], embeddings_2d[i,1], str(i), fontsize=9)

    plt.title('t-SNE Visualization of Codebook Embeddings')
    plt.xlabel('t-SNE Dimension 1')
    plt.ylabel('t-SNE Dimension 2')
    plt.grid(True)
    plt.show()

def main():
    parser = argparse.ArgumentParser(description='Inference for the Transformer model.')
    parser.add_argument('--checkpoint_path', type=str, default='./vq_model_chekcpoints/d_model_512_codebook_64/best_model.pt', help='Path to the model checkpoint.')
    parser.add_argument('--output_dir', type=str, default='inference_outputs', help='Directory to save inference results.')
    parser.add_argument('--test_batch_size', type=int, default=5012, required=False, help='Batch size for testing.')
    parser.add_argument('--device', type=str, default=None, help='Device to run inference on (e.g., "cuda" or "cpu"). If not set, uses Accelerator default.')
    parser.add_argument('--num_workers', type=int, default=0, help='Number of workers for DataLoader.')
    args = parser.parse_args()

    model_name = args.checkpoint_path.split('/')[2]
    print(model_name)
    # Initialize Accelerator
    accelerator = Accelerator()
    device = args.device if args.device else accelerator.device
    set_seed(42)

    # Ensure output directory exists
    os.makedirs(args.output_dir + '/' + model_name, exist_ok=True)

    # Load test dataset
    test_dataset = DiffusionGeneratorDataset(data_type="test")
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers)

    # Initialize the model architecture (ensure it matches the training setup)
    d_model = 512
    d_inner = d_model * 4
    n_layer = 4
    n_head = 8
    dropout = 0.1
    codebook_size, commitment_cost = 64, 0.25
    n_tokens = 10
    model = Transformer(
        d_model=d_model,
        d_inner=d_inner,
        n_layer=n_layer,
        n_head=n_head,
        dropout=dropout,
        codebook_size=codebook_size,
        commitment_cost=commitment_cost,
        n_tokens=n_tokens
    )

    # Load the checkpoint
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    # visualization_codebook(model)

    # Prepare model and dataloader with Accelerator
    model, test_dataloader = accelerator.prepare(model, test_dataloader)

    # Containers for storing results
    all_coords_outputs = []
    gt_coords_outputs = []

    # Inference loop
    with torch.no_grad():
        progress_bar = tqdm(test_dataloader, desc="Inference", disable=not accelerator.is_local_main_process)
        for batch in progress_bar:

            # Forward pass
            coords_output, _, _ = model(batch)

            # Post-process outputs if necessary
            # For example, convert logits to predicted tokens
            all_coords_outputs.append(coords_output.cpu().numpy())
            gt_coords_outputs.append(batch.cpu().numpy())

    all_coords_outputs = np.concatenate(all_coords_outputs, axis=0)
    gt_coords_outputs = np.concatenate(gt_coords_outputs, axis=0)

    # Save the outputs
    coords_output_path = os.path.join(args.output_dir + '/' + model_name, 'predicted_coords.npz')
    np.savez(f'{coords_output_path}', all_coords_outputs=all_coords_outputs, gt_coords_outputs=gt_coords_outputs)

    if accelerator.is_main_process:
        print(f"Inference completed. Results saved to '{args.output_dir + '/' + model_name}'")
        print(f"Predicted coordinates saved to: {coords_output_path}")

if __name__ == "__main__":
    main()

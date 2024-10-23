import os
import argparse
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm.auto import tqdm

from dataloader import ClusterLayoutDataset
from transformer import Transformer

def main():
    parser = argparse.ArgumentParser(description='Inference for the Transformer model.')
    parser.add_argument('--checkpoint_path', type=str, default='./vq_model_checkpoints/d_256_cb_64_st_4/best_model.pt', help='Path to the model checkpoint.')
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
    test_dataset = ClusterLayoutDataset(data_type="test")
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, num_workers=args.num_workers)

    # Initialize the model architecture (ensure it matches the training setup)
    d_model = 256
    d_inner = d_model * 4
    n_layer = 4
    n_head = 8
    dropout = 0.1
    codebook_size, commitment_cost = 64, 0.25
    n_tokens = 10
    sample_tokens = 4
    model = Transformer(
        d_model=d_model,
        d_inner=d_inner,
        n_layer=n_layer,
        n_head=n_head,
        dropout=dropout,
        codebook_size=codebook_size,
        commitment_cost=commitment_cost,
        n_tokens=n_tokens,
        sample_tokens=sample_tokens
    )

    # Load the checkpoint
    checkpoint = torch.load(args.checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()

    # Prepare model and dataloader with Accelerator
    model, test_dataloader = accelerator.prepare(model, test_dataloader)

    # Containers for storing results
    all_coords_outputs = []
    gt_coords_outputs = []

    # Inference loop
    with torch.no_grad():
        progress_bar = tqdm(test_dataloader, desc="Inference", disable=not accelerator.is_local_main_process)
        for batch in progress_bar:
            bbox_labels = batch['bbox_labels']
            category_labels = batch['category_labels']

            # 모델 Forward
            bbox_output, category_output, vq_loss, perplexity = model(bbox_labels, category_labels)
            
            bbox_output = torch.argmax(bbox_output, dim=-1) / 63
            output = torch.cat((bbox_output, category_output), dim=-1)

            gt = torch.cat((bbox_labels / 63, category_labels), dim=-1)

            # Post-process outputs if necessary
            # For example, convert logits to predicted tokens
            all_coords_outputs.append(output.cpu().numpy())
            gt_coords_outputs.append(gt.cpu().numpy())

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

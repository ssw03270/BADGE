import os
import argparse
import numpy as np
import pickle

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm.auto import tqdm

import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
sys.path.append(project_root)

from sampling_dataloader import ClusterLayoutDataset
from scripts.models.PolygonBased.cluster_layout.transformer import ContinuousTransformer, DiscreteTransformer

def custom_collate_fn(batch):
    batch = [item for item in batch if item is not None]
    
    if len(batch) == 0:
        # 모든 항목이 무효한 경우, 빈 배치를 반환하거나 예외를 발생시킬 수 있습니다.
        # 여기서는 빈 텐서를 반환합니다.
        return None, None, None

    # 배치에서 각 데이터를 분리합니다.
    bldg_layout_list = [item[0] for item in batch]  # 건물 레이아웃 데이터
    min_coords_list = [item[1] for item in batch]   # 최소 좌표
    range_max_list = [item[2] for item in batch]    # 최대 범위
    region_polygons = [item[3] for item in batch]

    # 각 데이터를 텐서로 변환하여 일관된 배치를 만듭니다.
    # 건물 레이아웃 데이터는 텐서로 변환
    bldg_layout_tensor = torch.cat(bldg_layout_list, dim=0)
    min_coords_tensor = torch.cat(min_coords_list, dim=0)
    range_max_tensor = torch.cat(range_max_list, dim=0)

    bldg_layout_start_indices = []
    current_index = 0
    for layout in bldg_layout_list:
        bldg_layout_start_indices.append(current_index)
        current_index += layout.shape[0]  # 레이아웃의 첫 번째 차원을 더해 인덱스를 증가

    return bldg_layout_tensor, min_coords_tensor, range_max_tensor, bldg_layout_start_indices, region_polygons

def main():
    parser = argparse.ArgumentParser(description='Inference for the Transformer model.')
    parser.add_argument('--user_name', type=str, default="ssw03270", required=False, help='User name.')
    parser.add_argument('--checkpoint_path', type=str, default='./vq_model_checkpoints/d_256_cb_512_coords_continuous_norm_blk/best_model.pt', help='Path to the model checkpoint.')
    parser.add_argument('--output_dir', type=str, default='inference_outputs', help='Directory to save inference results.')
    parser.add_argument('--test_batch_size', type=int, default=512, required=False, help='Batch size for testing.')
    parser.add_argument('--codebook_size', type=int, default=512, required=False, help='Codebook size.')
    parser.add_argument('--d_model', type=int, default=256, required=False, help='Model dimension.')
    parser.add_argument('--device', type=str, default=None, help='Device to run inference on (e.g., "cuda" or "cpu"). If not set, uses Accelerator default.')
    parser.add_argument("--coords_type", type=str, default="continuous", help="coordinate type")
    parser.add_argument("--norm_type", type=str, default="blk", help="coordinate type")
    parser.add_argument("--model_name", type=str, default="none", help="coordinate type")
    parser.add_argument("--inference_type", type=str, default="generate", choices=["generate", "recon"],help="coordinate type")
    parser.add_argument("--retrieval_type", type=str, default="retrieval", choices=["original", "retrieval"],help="coordinate type")
    args = parser.parse_args()

    if args.model_name == "none":
        args.model_name = f"d_{args.d_model}_cb_{args.codebook_size}_coords_{args.coords_type}_norm_{args.norm_type}_{args.inference_type}_{args.retrieval_type}"
        
    # Initialize Accelerator
    accelerator = Accelerator()
    device = args.device if args.device else accelerator.device
    set_seed(42)

    # Ensure output directory exists
    os.makedirs(args.output_dir + '/' + args.model_name, exist_ok=True)

    # Load test dataset
    test_dataset = ClusterLayoutDataset(data_type="test", user_name=args.user_name, coords_type=args.coords_type, norm_type=args.norm_type, retrieval_type=args.retrieval_type)
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, shuffle=False, collate_fn=custom_collate_fn)

    # 모델 초기화
    d_inner = args.d_model * 4
    n_layer = 4
    n_head = 8
    dropout = 0.1
    commitment_cost = 0.25
    n_tokens = 10
    if args.coords_type == "continuous":
        model = ContinuousTransformer(d_model=args.d_model, d_inner=d_inner, n_layer=n_layer, n_head=n_head, dropout=dropout, 
                                      codebook_size=args.codebook_size, commitment_cost=commitment_cost, n_tokens=n_tokens)
    elif args.coords_type == "discrete":
        model = DiscreteTransformer(d_model=args.d_model, d_inner=d_inner, n_layer=n_layer, n_head=n_head, dropout=dropout, 
                                    codebook_size=args.codebook_size, commitment_cost=commitment_cost, n_tokens=n_tokens)

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
    min_coords_outputs = []
    range_max_outputs = []
    region_polygons_outputs = []

    # Inference loop
    with torch.no_grad():
        progress_bar = tqdm(test_dataloader, desc="Inference", disable=not accelerator.is_local_main_process)
        for batch in progress_bar:
            data = batch[0]
            min_coords = batch[1]
            range_max = batch[2]
            start_indices = batch[3]
            region_polygons = batch[4]

            # 모델 Forward
            if args.inference_type == 'recon':
                bbox_output = model(data)
            elif args.inference_type == 'generate':
                bbox_output = model.sampling(data)

            if args.coords_type == 'discrete':
                bbox_output = torch.argmax(bbox_output, dim=-1).view(-1, 10, 6)

                bbox_output = bbox_output.float()
                data = data.float()

                bbox_output[:, :, :5] = bbox_output[:, :, :5] / 63
                data[:, :, :5] = data[:, :, :5] / 63

            for i in range(len(start_indices) - 1):
                all_coords_outputs.append(bbox_output[start_indices[i]:start_indices[i+1]].cpu().numpy())
                gt_coords_outputs.append(data[start_indices[i]:start_indices[i+1]].cpu().numpy())
                min_coords_outputs.append(min_coords[start_indices[i]:start_indices[i+1]].cpu().numpy())
                range_max_outputs.append(range_max[start_indices[i]:start_indices[i+1]].cpu().numpy())
            
            for region_poly in region_polygons:
                region_polygons_outputs.append(region_poly)

    coords_output_path = os.path.join(args.output_dir + '/' + args.model_name, 'predicted_coords.pkl')

    results = {
        'all_coords_outputs': all_coords_outputs,
        'gt_coords_outputs': gt_coords_outputs,
        'min_coords_outputs': min_coords_outputs,
        'range_max_outputs': range_max_outputs,
        'region_polygons_outputs': region_polygons_outputs
    }

    # 데이터를 pkl 파일로 저장
    with open(coords_output_path, 'wb') as f:
        pickle.dump(results, f)

    if accelerator.is_main_process:
        print(f"Inference completed. Results saved to '{args.output_dir + '/' + args.model_name}'")
        print(f"Predicted coordinates saved to: {coords_output_path}")

if __name__ == "__main__":
    main()

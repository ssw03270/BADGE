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

from diffusion_utils import *
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../..'))
sys.path.append(project_root)

from dataloader import BlkLayoutDataset
from diffusion import Diffusion

def custom_collate(batch):
    batch = [item for item in batch if item is not None]
    
    if len(batch) == 0:
        # 모든 항목이 무효한 경우, 빈 배치를 반환하거나 예외를 발생시킬 수 있습니다.
        # 여기서는 빈 텐서를 반환합니다.
        return None, None, None

    # 배치에서 각 데이터를 분리합니다.
    bldg_layout_list = [item[0] for item in batch]  # 건물 레이아웃 데이터
    image_mask_list = [item[1] for item in batch]   # 최소 좌표
    pad_mask_list = [item[2] for item in batch]    # 최대 범위
    cross_attn_mask_list = [item[3] for item in batch]    # 최대 범위
    self_attn_mask_list = [item[4] for item in batch]    # 최대 범위
    region_polygons = [item[5] for item in batch]

    # 각 데이터를 텐서로 변환하여 일관된 배치를 만듭니다.
    # 건물 레이아웃 데이터는 텐서로 변환
    bldg_layout_tensor = torch.stack(bldg_layout_list)
    image_mask_tensor = torch.stack(image_mask_list)
    pad_mask_tensor = torch.stack(pad_mask_list)
    cross_attn_mask_list = torch.stack(cross_attn_mask_list)
    self_attn_mask_list = torch.stack(self_attn_mask_list)

    return bldg_layout_tensor, image_mask_tensor, pad_mask_tensor, cross_attn_mask_list, self_attn_mask_list, region_polygons

def main():
    parser = argparse.ArgumentParser(description='Inference for the Transformer model.')
    parser.add_argument('--user_name', type=str, default="ssw03270", required=False, help='User name.')
    parser.add_argument('--checkpoint_path', type=str, default='./diffusion_checkpoints/retrieval_conditional_refine/best_model.pt', help='Path to the model checkpoint.')
    parser.add_argument('--output_dir', type=str, default='inference_outputs', help='Directory to save inference results.')
    parser.add_argument('--test_batch_size', type=int, default=128, required=False, help='Batch size for testing.')
    parser.add_argument('--d_model', type=int, default=512, required=False, help='Model dimension.')
    parser.add_argument("--sample_t_max", default=999, help="maximum t in training", type=int)
    parser.add_argument('--device', type=str, default=None, help='Device to run inference on (e.g., "cuda" or "cpu"). If not set, uses Accelerator default.')
    parser.add_argument("--coords_type", type=str, default="continuous", help="coordinate type")
    parser.add_argument("--norm_type", type=str, default="blk", help="coordinate type")
    parser.add_argument("--model_name", type=str, default="none", help="coordinate type")
    parser.add_argument("--train_type", type=str, default="conditional", choices=["generation", "conditional"],help="coordinate type")
    parser.add_argument("--retrieval_type", type=str, default="original", choices=["original", "retrieval"],help="coordinate type")
    parser.add_argument("--inference_type", type=str, default="noise", choices=["refine", "noise"],help="coordinate type")
    args = parser.parse_args()

    if args.model_name == "none":
        args.model_name = f"{args.retrieval_type}_{args.train_type}_{args.inference_type}"
        
    # Initialize Accelerator
    accelerator = Accelerator()
    device = args.device if args.device else accelerator.device
    set_seed(42)

    # Ensure output directory exists
    os.makedirs(args.output_dir + '/' + args.model_name, exist_ok=True)

    # Load test dataset
    test_dataset = BlkLayoutDataset(data_type="test", device=device, is_main_process=accelerator.is_main_process, retrieval_type=args.retrieval_type)
    test_dataloader = DataLoader(test_dataset, batch_size=args.test_batch_size, num_workers=4, shuffle=False, collate_fn=custom_collate)

    # 모델 초기화
    d_inner = args.d_model * 4
    n_layer = 4
    n_head = 8
    dropout = 0.1
    model = Diffusion(num_timesteps=1000, n_head=n_head, d_model=args.d_model,
                      d_inner=d_inner, seq_dim=6, n_layer=n_layer,
                      device=device, ddim_num_steps=200, dropout=dropout)

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
    region_polygons_outputs = []

    # Inference loop
    with torch.no_grad():
        progress_bar = tqdm(test_dataloader, desc="Inference", disable=not accelerator.is_local_main_process)
        for batch_idx, batch in enumerate(progress_bar):
            layout = batch[0].to(device)
            image_mask = batch[1].to(device)
            pad_mask = batch[2].to(device)
            cross_attn_mask = batch[3].to(device)
            self_attn_mask = batch[4].to(device)
            region_poly = batch[5]

            # # 모델 Forward
            layout_output = accelerator.unwrap_model(model).reverse_ddim(layout, image_mask, train_type=args.train_type, inference_type=args.inference_type)

            all_coords_outputs += layout_output.cpu().numpy().tolist()
            gt_coords_outputs += layout.cpu().numpy().tolist()
            region_polygons_outputs += region_poly

            if batch_idx == 10:
                break

    coords_output_path = os.path.join(args.output_dir + '/' + args.model_name, 'predicted_coords.pkl')

    results = {
        'all_coords_outputs': all_coords_outputs,
        'gt_coords_outputs': gt_coords_outputs,
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

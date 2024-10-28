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
from transformer import ContinuousTransformer, DiscreteTransformer

def custom_collate_fn(batch):
    # 배치에서 각 데이터를 분리합니다.
    bldg_layout_list = [item[0] for item in batch]  # 건물 레이아웃 데이터
    min_coords_list = [item[1] for item in batch]   # 최소 좌표
    range_max_list = [item[2] for item in batch]    # 최대 범위

    # 각 데이터를 텐서로 변환하여 일관된 배치를 만듭니다.
    # 건물 레이아웃 데이터는 텐서로 변환
    bldg_layout_tensor = torch.stack(bldg_layout_list)

    # min_coords_list와 range_max_list는 각 배치 내의 배열이 같은 길이를 가지지 않을 수 있으므로 패딩을 추가
    min_coords_tensor = torch.tensor(min_coords_list, dtype=torch.float32)
    range_max_tensor = torch.tensor(range_max_list, dtype=torch.float32)

    return bldg_layout_tensor, min_coords_tensor, range_max_tensor

def main():
    parser = argparse.ArgumentParser(description='Inference for the Transformer model.')
    parser.add_argument('--user_name', type=str, default="ssw03270", required=False, help='User name.')
    parser.add_argument('--checkpoint_path', type=str, default='./vq_model_checkpoints/d_256_cb_512_st_9/best_model.pt', help='Path to the model checkpoint.')
    parser.add_argument('--output_dir', type=str, default='inference_outputs', help='Directory to save inference results.')
    parser.add_argument('--test_batch_size', type=int, default=5012, required=False, help='Batch size for testing.')
    parser.add_argument('--codebook_size', type=int, default=64, required=False, help='Codebook size.')
    parser.add_argument('--d_model', type=int, default=512, required=False, help='Model dimension.')
    parser.add_argument('--device', type=str, default=None, help='Device to run inference on (e.g., "cuda" or "cpu"). If not set, uses Accelerator default.')
    parser.add_argument("--coords_type", type=str, default="continuous", help="coordinate type")
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
    test_dataset = ClusterLayoutDataset(data_type="test", user_name=args.user_name, coords_type=args.coords_type)
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

    # Inference loop
    with torch.no_grad():
        progress_bar = tqdm(test_dataloader, desc="Inference", disable=not accelerator.is_local_main_process)
        for batch in progress_bar:
            data = batch[0]
            min_coords = batch[1]
            range_max = batch[2]

            # 모델 Forward
            bbox_output, recon_loss, vq_loss, perplexity = model(data)

            if args.coords_type == 'discrete':
                bbox_output = torch.argmax(bbox_output, dim=-1).view(-1, 10, 6)

                bbox_output = bbox_output.float()
                data = data.float()

                bbox_output[:, :, :5] = bbox_output[:, :, :5] / 63
                data[:, :, :5] = data[:, :, :5] / 63

            all_coords_outputs.append(bbox_output.cpu().numpy())
            gt_coords_outputs.append(data.cpu().numpy())
            min_coords_outputs.append(min_coords.cpu().numpy())
            range_max_outputs.append(range_max.cpu().numpy())

    all_coords_outputs = np.concatenate(all_coords_outputs, axis=0)
    gt_coords_outputs = np.concatenate(gt_coords_outputs, axis=0)
    min_coords_outputs = np.concatenate(min_coords_outputs, axis=0)
    range_max_outputs = np.concatenate(range_max_outputs, axis=0)


    # Save the outputs
    coords_output_path = os.path.join(args.output_dir + '/' + model_name, 'predicted_coords.npz')
    np.savez(f'{coords_output_path}', all_coords_outputs=all_coords_outputs, gt_coords_outputs=gt_coords_outputs,
             min_coords_outputs=min_coords_outputs, range_max_outputs=range_max_outputs)

    if accelerator.is_main_process:
        print(f"Inference completed. Results saved to '{args.output_dir + '/' + model_name}'")
        print(f"Predicted coordinates saved to: {coords_output_path}")

if __name__ == "__main__":
    main()

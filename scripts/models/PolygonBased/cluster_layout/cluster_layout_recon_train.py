import wandb
import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from accelerate import Accelerator
from accelerate.utils import set_seed
from tqdm.auto import tqdm

from dataloader import ClusterLayoutDataset
from transformer import ContinuousTransformer, DiscreteTransformer


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

    # 각 데이터를 텐서로 변환하여 일관된 배치를 만듭니다.
    # 건물 레이아웃 데이터는 텐서로 변환
    bldg_layout_tensor = torch.cat(bldg_layout_list, dim=0)
    min_coords_tensor = torch.cat(min_coords_list, dim=0)
    range_max_tensor = torch.cat(range_max_list, dim=0)

    return bldg_layout_tensor, min_coords_tensor, range_max_tensor

# 학습 코드
def main():
    parser = argparse.ArgumentParser(description='Train the Transformer model.')
    parser.add_argument('--user_name', type=str, default="ssw03270", required=False, help='User name.')
    parser.add_argument('--num_epochs', type=int, default=2000, required=False, help='Number of training epochs.')
    parser.add_argument('--save_epoch', type=int, default=10, required=False, help='Number of save epoch.')
    parser.add_argument('--train_batch_size', type=int, default=512, required=False, help='Batch size for training.')
    parser.add_argument('--val_batch_size', type=int, default=512, required=False, help='Batch size for validation.')
    parser.add_argument('--lr', type=float, default=0.00001, required=False, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.02, required=False, help='Weight decay.')
    parser.add_argument('--codebook_size', type=int, default=64, required=False, help='Codebook size.')
    parser.add_argument('--d_model', type=int, default=512, required=False, help='Model dimension.')
    parser.add_argument("--local-rank", type=int, default=0, help="Local rank for distributed training")
    parser.add_argument("--coords_type", type=str, default="continuous", help="coordinate type")
    parser.add_argument("--norm_type", type=str, default="bldg_bbox", help="coordinate type")
    parser.add_argument("--model_name", type=str, default="none", help="coordinate type")
    args = parser.parse_args()

    accelerator = Accelerator()  # 여기서 설정
    device = accelerator.device
    set_seed(42)

    if args.model_name == "none":
        args.model_name = f"d_{args.d_model}_cb_{args.codebook_size}_coords_{args.coords_type}_norm_{args.norm_type}"

    if accelerator.is_main_process:
        wandb.login(key='0f272b4978c0b450c3765b24b8abd024d7799e80')
        wandb.init(
            project="codebook_train",  # Replace with your WandB project name
            config=vars(args),            # Logs all hyperparameters
            name=args.model_name,  # Optional: Name your run
            save_code=True                # Optional: Save your code with the run
        )

    # 데이터셋 로드
    train_dataset = ClusterLayoutDataset(data_type="train", user_name=args.user_name, coords_type=args.coords_type, norm_type=args.norm_type)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, num_workers=4, shuffle=True, collate_fn=custom_collate_fn)
    
    val_dataset = ClusterLayoutDataset(data_type="val", user_name=args.user_name, coords_type=args.coords_type, norm_type=args.norm_type)
    val_dataloader = DataLoader(val_dataset, batch_size=args.val_batch_size, num_workers=4, shuffle=False, collate_fn=custom_collate_fn)

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

    # 옵티마이저 및 스케줄러 설정
    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # Accelerator 준비
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
    val_dataloader = accelerator.prepare(val_dataloader)

    best_val_loss = float('inf')
    best_epoch = 0

    # 학습 루프
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}", disable=not accelerator.is_local_main_process)

        for batch in progress_bar:
            optimizer.zero_grad()
            data = batch[0]
            min_coords = batch[1]
            range_max = batch[2]

            # # 모델 Forward
            bbox_output, recon_loss, vq_loss, perplexity = model(data)

            loss = recon_loss + vq_loss

            # 역전파 및 최적화
            accelerator.backward(loss)
            optimizer.step()
            total_loss += loss.item()

            progress_bar.set_postfix({'loss': total_loss / (progress_bar.n + 1), 'lr': optimizer.param_groups[0]['lr']})

        # 검증 단계 (옵션)
        model.eval()
        val_loss = 0
        val_recon_loss = 0
        val_loss_vq = 0
        val_loss_coords = 0
        with torch.no_grad():
            for batch in val_dataloader:
                data = batch[0]
                min_coords = batch[1]
                range_max = batch[2]

                bbox_output, recon_loss, vq_loss, perplexity = model(data)

                loss = recon_loss + vq_loss

                val_loss += loss.item()
                val_recon_loss += recon_loss
                val_loss_vq += vq_loss

        # 모델 저장
        if accelerator.is_main_process and (epoch + 1) % args.save_epoch == 0:
            save_dir = f"vq_model_checkpoints/{args.model_name}/checkpoint_epoch_{epoch+1}"
            os.makedirs(save_dir, exist_ok=True)
            unwrapped_model = accelerator.unwrap_model(model)
            torch.save(unwrapped_model.state_dict(), os.path.join(save_dir, "model.pt"))

        # 최저 검증 손실 모델 저장
        if accelerator.is_main_process and val_loss / len(val_dataloader) < best_val_loss:
            best_val_loss = val_loss / len(val_dataloader)
            best_epoch = epoch + 1
            best_model_dir = f"vq_model_checkpoints/{args.model_name}/best_model.pt"
            os.makedirs(os.path.dirname(best_model_dir), exist_ok=True)
            unwrapped_model = accelerator.unwrap_model(model)
            torch.save(unwrapped_model.state_dict(), best_model_dir)
                
        if accelerator.is_main_process:
            avg_train_loss = total_loss / len(train_dataloader)
            avg_val_loss = val_loss / len(val_dataloader)
            avg_val_recon = val_recon_loss / len(val_dataloader)
            avg_val_coords = val_loss_coords / len(val_dataloader)
            avg_val_vq = val_loss_vq / len(val_dataloader) if vq_loss is not None else None

            wandb.log({
                "train_loss": avg_train_loss,
                "validation_loss": avg_val_loss,
                "validation_recon_loss": avg_val_recon,
                "validation_vq_loss": avg_val_vq,
                "best_epoch": best_epoch
            })

            print(f"Validation Loss: {avg_val_loss}")
            print(f"Validation Recon Loss: {avg_val_recon}")
            print(f"Validation Coords Loss: {avg_val_coords}")
            print(f"Validation VQ Loss: {avg_val_vq}")
            print(f"Best Validation Loss: {best_val_loss}, Best Epoch: {best_epoch}")

if __name__ == "__main__":
    main()
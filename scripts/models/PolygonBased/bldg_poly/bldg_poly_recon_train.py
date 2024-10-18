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

from dataloader import BuildingDataset
from transformer import Transformer

# 학습 코드
def main():
    parser = argparse.ArgumentParser(description='Train the Transformer model.')
    parser.add_argument('--num_epochs', type=int, default=2000, required=False, help='Number of training epochs.')
    parser.add_argument('--save_epoch', type=int, default=10, required=False, help='Number of save epoch.')
    parser.add_argument('--train_batch_size', type=int, default=128, required=False, help='Batch size for training.')
    parser.add_argument('--val_batch_size', type=int, default=128, required=False, help='Batch size for validation.')
    parser.add_argument('--lr', type=float, default=0.0001, required=False, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=0.02, required=False, help='Weight decay.')
    args = parser.parse_args()

    accelerator = Accelerator()
    device = accelerator.device
    set_seed(42)

    # 데이터셋 로드
    train_dataset = BuildingDataset(data_type="train")
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    
    val_dataset = BuildingDataset(data_type="val")
    val_dataloader = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False)

    # 모델 초기화
    d_model = 512
    d_inner = 1024
    n_layer = 4
    n_head = 8
    dropout = 0.1
    grid_size = 64
    codebook_size, commitment_cost = 128, 0.25
    n_tokens = 4
    model = Transformer(d_model=d_model, d_inner=d_inner, n_layer=n_layer, n_head=n_head, dropout=dropout, grid_size=grid_size, 
                        codebook_size=codebook_size, commitment_cost=commitment_cost, n_tokens=n_tokens)

    # 옵티마이저 및 스케줄러 설정
    # optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

    # Accelerator 준비
    model, optimizer, train_dataloader = accelerator.prepare(model, optimizer, train_dataloader)
    val_dataloader = accelerator.prepare(val_dataloader)

    # 학습 루프
    for epoch in range(args.num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch+1}", disable=not accelerator.is_local_main_process)
        for batch in progress_bar:
            optimizer.zero_grad()

            # 데이터 로드
            bldg_coords = batch['bldg_coords'].to(device)
            corner_coords = batch['corner_coords'].to(device)
            corner_indices = batch['corner_indices'].to(device)

            # 모델 Forward
            coords_output, vq_loss, perplexity = model(bldg_coords, corner_coords, corner_indices)

            loss_coords = F.mse_loss(coords_output, bldg_coords.clone())
            if vq_loss is not None:
                loss = loss_coords + vq_loss
            else:
                loss = loss_coords

            # 역전파 및 최적화
            accelerator.backward(loss)
            optimizer.step()
            total_loss += loss.item()
            progress_bar.set_postfix({'loss': total_loss / (progress_bar.n + 1)})

        # 스케줄러 업데이트
        lr_scheduler.step()

        # 검증 단계 (옵션)
        model.eval()
        val_loss = 0
        val_loss_coords = 0
        val_loss_vq = 0
        with torch.no_grad():
            for batch in val_dataloader:
                bldg_coords = batch['bldg_coords'].to(device)
                corner_coords = batch['corner_coords'].to(device)
                corner_indices = batch['corner_indices'].to(device)

                coords_output, vq_loss, perplexity = model(bldg_coords, corner_coords, corner_indices)

                loss_coords = F.mse_loss(coords_output, bldg_coords.clone())
                if vq_loss is not None:
                    loss =  loss_coords + vq_loss
                else:
                    loss = loss_coords

                val_loss += loss.item()
                val_loss_coords += loss_coords
                if vq_loss is not None:
                    val_loss_vq += vq_loss

        print(f"Validation Loss: {val_loss / len(val_dataloader)}")
        print(f"Validation Coords Loss: {val_loss_coords / len(val_dataloader)}")
        print(f"Validation VQ Loss: {val_loss_vq / len(val_dataloader)}")

        # 모델 저장
        if accelerator.is_main_process and (epoch + 1) % args.save_epoch == 0:
            save_dir = "vq_model_chekcpoints/checkpoint_epoch_{}".format(epoch+1)
            os.makedirs(save_dir, exist_ok=True)
            accelerator.wait_for_everyone()
            unwrapped_model = accelerator.unwrap_model(model)
            torch.save(unwrapped_model.state_dict(), os.path.join(save_dir, "model.pt"))

if __name__ == "__main__":
    main()
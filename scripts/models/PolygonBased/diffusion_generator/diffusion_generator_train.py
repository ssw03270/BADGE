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

from dataloader import DiffusionGeneratorDataset
from transformer import Transformer

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
    args = parser.parse_args()

    accelerator = Accelerator()  # 여기서 설정
    device = accelerator.device
    set_seed(42)

    # 데이터셋 로드
    train_dataset = DiffusionGeneratorDataset(data_type="train", user_name=args.user_name)
    train_dataloader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    
    val_dataset = DiffusionGeneratorDataset(data_type="val", user_name=args.user_name)
    val_dataloader = DataLoader(val_dataset, batch_size=args.val_batch_size, shuffle=False)

    # 모델 초기화
    d_inner = args.d_model * 4
    n_layer = 4
    n_head = 8
    dropout = 0.1
    commitment_cost = 0.25
    n_tokens = 10
    model = Transformer(d_model=args.d_model, d_inner=d_inner, n_layer=n_layer, n_head=n_head, dropout=dropout, 
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

            # 모델 Forward
            coords_output, vq_loss, perplexity = model(batch)

            loss_coords = F.mse_loss(coords_output, batch.clone())
            if vq_loss is not None:
                loss = loss_coords + vq_loss
            else:
                loss = loss_coords

            # 역전파 및 최적화
            accelerator.backward(loss)
            optimizer.step()
            total_loss += loss.item()

            progress_bar.set_postfix({'loss': total_loss / (progress_bar.n + 1), 'lr': optimizer.param_groups[0]['lr']})

        # 검증 단계 (옵션)
        model.eval()
        val_loss = 0
        val_loss_coords = 0
        val_loss_vq = 0
        with torch.no_grad():
            for batch in val_dataloader:

                coords_output, vq_loss, perplexity = model(batch)

                loss_coords = F.mse_loss(coords_output, batch.clone())
                if vq_loss is not None:
                    loss =  loss_coords + vq_loss
                else:
                    loss = loss_coords

                val_loss += loss.item()
                val_loss_coords += loss_coords
                if vq_loss is not None:
                    val_loss_vq += vq_loss

        if accelerator.is_main_process:
            print(f"Validation Loss: {val_loss / len(val_dataloader)}")
            print(f"Validation Coords Loss: {val_loss_coords / len(val_dataloader)}")
            print(f"Validation VQ Loss: {val_loss_vq / len(val_dataloader)}")

        # 모델 저장
        if accelerator.is_main_process and (epoch + 1) % args.save_epoch == 0:
            save_dir = f"vq_model_checkpoints/d_model_{args.d_model}_codebook_{args.codebook_size}/checkpoint_epoch_{epoch+1}"
            os.makedirs(save_dir, exist_ok=True)
            unwrapped_model = accelerator.unwrap_model(model)
            torch.save(unwrapped_model.state_dict(), os.path.join(save_dir, "model.pt"))

        # 최저 검증 손실 모델 저장
        if accelerator.is_main_process and val_loss < best_val_loss:
            best_val_loss = val_loss
            best_epoch = epoch + 1
            best_model_dir = f"vq_model_checkpoints/d_model_{args.d_model}_codebook_{args.codebook_size}/best_model.pt"
            os.makedirs(os.path.dirname(best_model_dir), exist_ok=True)
            unwrapped_model = accelerator.unwrap_model(model)
            torch.save(unwrapped_model.state_dict(), best_model_dir)

            print(f"Best Validation Loss: {best_val_loss / len(val_dataloader)}, Best Epoch: {best_epoch}")

if __name__ == "__main__":
    main()
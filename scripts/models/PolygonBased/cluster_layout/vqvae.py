from torch import Tensor, LongTensor
from typing import *
import numpy as np

import torch
import torch.nn as nn
from torch.nn import functional as F

class VectorQuantizer(nn.Module):
    def __init__(self, num_embeddings, embedding_dim, commitment_cost, sample_tokens):
        super(VectorQuantizer, self).__init__()
        self.embedding_dim = embedding_dim
        self.num_embeddings = num_embeddings
        self.commitment_cost = commitment_cost
        self.sample_tokens = sample_tokens

        self.embedding = nn.Embedding(self.num_embeddings, self.embedding_dim // sample_tokens)
        # self.embedding = nn.Embedding(self.num_embeddings, 1)
        self.embedding.weight.data.uniform_(-1/self.num_embeddings, 1/self.num_embeddings)
    
    def forward(self, inputs):
        # 입력을 (B, C, H, W)에서 (B*H*W, C)로 변환
        input_shape = inputs.shape
        flat_input = inputs.view(-1, self.embedding_dim // self.sample_tokens)
        # flat_input = inputs.view(-1, 1)
        
        # 거리 계산
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                     + torch.sum(self.embedding.weight**2, dim=1)
                     - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
        
        # 가장 가까운 인덱스 찾기
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        encodings = torch.zeros(encoding_indices.size(0), self.num_embeddings, device=inputs.device)
        encodings.scatter_(1, encoding_indices, 1)
        
        # 양자화된 벡터
        quantized = torch.matmul(encodings, self.embedding.weight).view(input_shape)
        
        # 손실 계산
        e_latent_loss = F.mse_loss(quantized.detach(), inputs)
        q_latent_loss = F.mse_loss(quantized, inputs.detach())

        loss = q_latent_loss + self.commitment_cost * e_latent_loss
        
        quantized = inputs + (quantized - inputs).detach()
        avg_probs = torch.mean(encodings, dim=0)
        perplexity = torch.exp(-torch.sum(avg_probs * torch.log(avg_probs + 1e-10)))
        
        return quantized, loss, perplexity

    def get_encoding_indices(self, inputs):
        # 입력을 (B, C, H, W)에서 (B*H*W, C)로 변환
        input_shape = inputs.shape
        # flat_input = inputs.view(-1, self.embedding_dim)
        flat_input = inputs.view(-1, 1)
        
        # 거리 계산
        distances = (torch.sum(flat_input**2, dim=1, keepdim=True)
                     + torch.sum(self.embedding.weight**2, dim=1)
                     - 2 * torch.matmul(flat_input, self.embedding.weight.t()))
        
        # 가장 가까운 인덱스 찾기
        encoding_indices = torch.argmin(distances, dim=1).unsqueeze(1)
        return encoding_indices
import torch
import torch.nn as nn
from torch import Tensor, LongTensor

from module import ScaledDotProductAttention

class AdaLayerNorm(nn.Module):
    def __init__(self, dim: int, t_dim: int):
        super().__init__()
        self.gelu = nn.GELU()
        self.linear = nn.Linear(t_dim, dim*2)
        self.layernorm = nn.LayerNorm(dim, elementwise_affine=False)

    def forward(self, x: Tensor, t_emb: Tensor):
        emb: Tensor = self.linear(self.gelu(t_emb)).unsqueeze(1)
        while emb.dim() < x.dim():
            emb = emb.unsqueeze(1)

        scale, shift = torch.chunk(emb, 2, dim=-1)
        x = self.layernorm(x) * (1. + scale) + shift

        return x
    
class MultiHeadAttention(nn.Module):
    def __init__(self, n_head=8, d_model=512, dropout=0.1, t_dim=64):
        """
        Initializes the multi-head attention mechanism.

        Parameters:
        - n_head (int): Number of attention heads.
        - d_model (int): Total dimension of the model.
        - dropout (float): Dropout rate.
        """

        super(MultiHeadAttention, self).__init__()

        self.n_head = n_head
        self.d_model = d_model

        self.w_q = nn.Linear(d_model, d_model, bias=False)
        self.w_k = nn.Linear(d_model, d_model, bias=False)
        self.w_v = nn.Linear(d_model, d_model, bias=False)
        self.fc = nn.Linear(d_model, d_model, bias=False)

        self.attention = ScaledDotProductAttention(temperature=d_model // n_head ** 0.5)
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = AdaLayerNorm(d_model, t_dim)

    def forward(self, q, k, v, t, mask=None):
        """
        Forward pass for the multi-head attention mechanism.

        Parameters:
        - q, k, v (Tensor): Queries, keys, and values.
        - mask (Tensor, optional): Mask to prevent attention to certain positions.

        Returns:
        - Tensor: Output of the multi-head attention mechanism.
        """

        q = self.layer_norm(q, t)
        
        d_k, d_v, n_head = self.d_model // self.n_head, self.d_model // self.n_head, self.n_head
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)
        residual = q

        q = self.w_q(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_k(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_v(v).view(sz_b, len_v, n_head, d_k)

        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1).unsqueeze(1)

        q, attn = self.attention(q, k, v, mask=mask)

        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.fc(q)
        q = self.dropout(q)
        q += residual

        return q

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_inner, dropout=0.1, t_dim=64):
        """
        Initializes the position-wise feed-forward layer.

        Parameters:
        - d_model (int): Dimensionality of the input and output.
        - d_inner (int): Dimensionality of the hidden layer.
        - dropout (float): Dropout rate.
        """

        super().__init__()
        self.w1 = nn.Linear(d_model, d_inner)
        self.w2 = nn.Linear(d_inner, d_model)
        self.layer_norm = AdaLayerNorm(d_model, t_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, t):
        """
        Forward pass of the position-wise feed-forward layer.

        Parameters:
        - x (Tensor): Input tensor.

        Returns:
        - Tensor: Output tensor of the feed-forward layer.
        """

        x = self.layer_norm(x, t)
        residual = x
        x = self.w1(x)
        x = torch.relu(x)
        x = self.w2(x)
        x += residual

        return x
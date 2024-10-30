import torch
import torch.nn as nn
from torch import Tensor, LongTensor
from module import ScaledDotProductAttention

import math
from einops.layers.torch import Rearrange

class SinusoidalPosEmb(nn.Module):
    def __init__(self, num_steps: int, dim: int, rescale_steps: int = 4000):
        super().__init__()
        self.dim = dim
        self.num_steps = float(num_steps)
        self.rescale_steps = float(rescale_steps)

    def forward(self, x: Tensor):
        x = x / self.num_steps * self.rescale_steps
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class _AdaNorm(nn.Module):
    def __init__(
        self, n_embd: int, max_timestep: int, emb_type: str = "adalayernorm_abs"
    ):
        super().__init__()
        if "abs" in emb_type:
            self.emb = SinusoidalPosEmb(max_timestep, n_embd)
        elif "mlp" in emb_type:
            self.emb = nn.Sequential(
                Rearrange("b -> b 1"),
                nn.Linear(1, n_embd // 2),
                nn.ReLU(),
                nn.Linear(n_embd // 2, n_embd),
            )
        else:
            self.emb = nn.Embedding(max_timestep, n_embd)
        self.silu = nn.SiLU()
        self.linear = nn.Linear(n_embd, n_embd * 2)


class AdaLayerNorm(_AdaNorm):
    def __init__(
        self, n_embd: int, max_timestep: int, emb_type: str = "adalayernorm_abs"
    ):
        super().__init__(n_embd, max_timestep, emb_type)
        self.layernorm = nn.LayerNorm(n_embd, elementwise_affine=False)

    def forward(self, x: Tensor, timestep: int):

        emb = self.linear(self.silu(self.emb(timestep))).unsqueeze(1)
        scale, shift = torch.chunk(emb, 2, dim=2)
        x = self.layernorm(x) * (1 + scale) + shift
        return x
    
class MultiHeadAttention(nn.Module):
    def __init__(self, n_head=8, d_model=512, dropout=0.1, diffusion_step=1000):
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
        self.layer_norm = AdaLayerNorm(d_model, diffusion_step)

    def forward(self, q, k, v, timestep, mask=None):
        """
        Forward pass for the multi-head attention mechanism.

        Parameters:
        - q, k, v (Tensor): Queries, keys, and values.
        - mask (Tensor, optional): Mask to prevent attention to certain positions.

        Returns:
        - Tensor: Output of the multi-head attention mechanism.
        """

        q = self.layer_norm(q, timestep)
        
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
    def __init__(self, d_model, d_inner, dropout=0.1):
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
        self.layer_norm = nn.LayerNorm(d_model, eps=1e-6)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        Forward pass of the position-wise feed-forward layer.

        Parameters:
        - x (Tensor): Input tensor.

        Returns:
        - Tensor: Output tensor of the feed-forward layer.
        """

        x = self.layer_norm(x)
        residual = x
        x = self.w1(x)
        x = torch.relu(x)
        x = self.w2(x)
        x += residual

        return x
import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import math

import torchvision.models as models  # ### 수정된 부분 시작: ResNet18을 사용하기 위해 torchvision.models 추가

from layer import EncoderLayer, DecoderLayer


def get_1d_sincos_encode(steps: torch.Tensor, emb_dim: int, max_period: int=10000) -> torch.Tensor:
    """Get sinusoidal encodings for a batch of timesteps/positions."""
    assert steps.dim() == 1, f"Parameter `steps` must be a 1D tensor, but got {steps.dim()}D."

    half_dim = emb_dim // 2
    emb = torch.exp(- math.log(max_period) *\
        torch.arange(0, half_dim, device=steps.device).float() / half_dim)
    emb = steps[:, None].float() * emb[None, :]  # (num_steps, half_dim)

    # Concat sine and cosine encodings
    emb = torch.cat([torch.cos(emb), torch.sin(emb)], dim=-1)  # (num_steps, emb_dim)

    # Zero padding
    if emb_dim % 2 == 1: emb = nn.functional.pad(emb, (0, 1))
    assert emb.shape == (steps.shape[0], emb_dim)

    return emb

class Timestep(nn.Module):
    """Encode timesteps with sinusoidal encodings."""
    def __init__(self, time_emb_dim: int):
        super().__init__()
        self.time_emb_dim = time_emb_dim

    def forward(self, timesteps: torch.Tensor) -> torch.Tensor:
        return get_1d_sincos_encode(timesteps, self.time_emb_dim)

class TimestepEmbed(nn.Module):
    """Embed sinusoidal encodings with a 2-layer MLP."""
    def __init__(self, in_dim: int, time_emb_dim: int, act_fn_name: str="GELU"):
        super().__init__()

        self.mlp = nn.Sequential(
            nn.Linear(in_dim, time_emb_dim),
            getattr(nn, act_fn_name)(),
            nn.Linear(time_emb_dim, time_emb_dim)
        )

    def forward(self, sample: torch.Tensor) -> torch.Tensor:
        return self.mlp(sample)

class PositionalEncoding(nn.Module):
    def __init__(self, d_hid, n_node):
        """
        Initializes the positional encoding module.

        Parameters:
        - d_hid (int): The dimension of the hidden layer.
        - n_node (int): The number of nodes (positions) for which encoding will be generated.
        """

        super(PositionalEncoding, self).__init__()
        self.register_buffer('pos_table', self._get_sinusoid_encoding_table(n_node, d_hid))

    def _get_sinusoid_encoding_table(self, n_boundary, d_hid):
        """
        Generates the sinusoidal encoding table.

        Parameters:
        - n_boundary (int): Number of positions.
        - d_hid (int): The dimension of the hidden layer.

        Returns:
        - Tensor: The positional encoding table.
        """

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_boundary)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])

        return torch.FloatTensor(sinusoid_table).unsqueeze(0)

    def forward(self, x):
        """
        Adds the positional encoding to the input tensor.

        Parameters:
        - x (Tensor): The input tensor.

        Returns:
        - Tensor: The input tensor with positional encoding added.
        """

        return self.pos_table[:, :x.size(1)].clone().detach()

class TransformerEncoder(nn.Module):
    def __init__(self, d_model, d_inner, n_layer, n_head, dropout, codebook_size, codebook_dim, cluster_count):
        """
        Initializes the TransformerEncoder.

        Parameters are for configuring the encoder layers and the positional encoding.
        """

        super(TransformerEncoder, self).__init__()

        self.pos_enc = PositionalEncoding(d_model, n_node=codebook_dim * cluster_count)
        self.dropout = nn.Dropout(dropout)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model=d_model, d_inner=d_inner, n_head=n_head, dropout=dropout)
            for _ in range(n_layer)
        ])

    def forward(self, x, t, enc_mask=None):
        """
        Forward pass for the BoundaryEncoder.

        Parameters:
        - x (Tensor): The x features.
        - enc_mask (Tensor): The encoder mask.

        Returns:
        - Tensor: The encoded features.
        """
        enc_input = x + self.pos_enc(x).expand(x.shape[0], -1, -1)
        enc_output = self.dropout(enc_input)

        for enc_layer in self.layer_stack:
            enc_output = enc_layer(enc_output, t, enc_mask)

        return enc_output
    
class TransformerDecoder(nn.Module):
    def __init__(self, d_model, d_inner, n_layer, n_head, dropout, codebook_size, codebook_dim, cluster_count):
        """
        Initializes the TransformerDecoder.

        Parameters are for configuring the encoder layers and the positional encoding.
        """

        super(TransformerDecoder, self).__init__()

        self.pos_enc = PositionalEncoding(d_model, n_node=codebook_dim * cluster_count)
        self.dropout = nn.Dropout(dropout)

        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model=d_model, d_inner=d_inner, n_head=n_head, dropout=dropout)
            for _ in range(n_layer)
        ])

    def forward(self, x, t, condition, enc_mask=None):
        """
        Forward pass for the BoundaryEncoder.

        Parameters:
        - x (Tensor): The x features.
        - enc_mask (Tensor): The encoder mask.

        Returns:
        - Tensor: The encoded features.
        """
        dec_input = x + self.pos_enc(x).expand(x.shape[0], -1, -1)
        dec_output = self.dropout(dec_input)
        
        for dec_layer in self.layer_stack:
            dec_output = dec_layer(dec_output, t, condition)

        return dec_output

class Transformer(nn.Module):
    def __init__(self, d_model, d_inner, n_layer, n_head, dropout, codebook_size, codebook_dim, cluster_count):
        """
        Initializes the Transformer model.
        """

        super(Transformer, self).__init__()

        self.d_model = d_model
        self.t_dim = 64

        self.x_embed = nn.Embedding(codebook_dim, d_model)
        self.t_embed = nn.Sequential(
            Timestep(self.t_dim),
            TimestepEmbed(self.t_dim, self.t_dim)
        )

        self.condition_conv = nn.Conv2d(24, 3, kernel_size=1)
        self.resnet18 = models.resnet18(pretrained=True)
        # Remove the final fully connected layer
        self.resnet18 = nn.Sequential(*list(self.resnet18.children())[:-1])  # Output: (batch, 512, 1, 1)
        self.resnet18.requires_grad_(False)
        
        self.encoder = TransformerEncoder(d_model, d_inner, n_layer, n_head, dropout, codebook_size, codebook_dim, cluster_count)
        self.decoder = TransformerDecoder(d_model, d_inner, n_layer, n_head, dropout, codebook_size, codebook_dim, cluster_count)

        self.node_proj_out = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, codebook_dim)
        )
        self.node_out = nn.Sequential(
            nn.Linear(codebook_dim, codebook_dim),
            nn.GELU(),
            nn.Linear(codebook_dim, codebook_size - 1)  # +1 for empty token
        )

    def forward(self, x, condition, t):
        """
        Forward pass of the Transformer model.

        Processes coords features with attention mechanisms and positional encoding.
        x: torch.Size([512, 24, 128])
        condition: torch.Size([512, 24, 64, 64])
        """
        t = t * torch.ones(x.shape[0], dtype=t.dtype, device=t.device)

        # print(x.shape)  # torch.Size([4, 3072])
        x = self.x_embed(x) 
        # print(x.shape)  # torch.Size([4, 3072, 256])
        t = self.t_embed(t)

        condition = self.condition_conv(condition)
        condition = F.interpolate(condition, size=(224, 224), mode='bilinear', align_corners=False)  # 텐서 리사이즈
        condition = torch.sigmoid(condition)
        condition = self.resnet18(condition)
        # print(condition.shape)  # torch.Size([4, 512, 1, 1])

        condition = condition.view(condition.shape[0], 1, -1)   # torch.Size([4, 1, 512])

        enc_output = self.decoder(x, t, condition)
        # print(enc_output.shape) # torch.Size([4, 3072, 256])

        dec_output = self.node_proj_out(enc_output)
        # print(dec_output.shape) # torch.Size([4, 3072, 128])
        dec_output = self.node_out(dec_output)
        # print(dec_output.shape) # torch.Size([4, 3072, 17])
        dec_output = dec_output.permute(0, 2, 1)
        # print(dec_output.shape) # torch.Size([4, 17, 3072])

        return dec_output
        # dec_output = enc_output

        # print(dec_output.shape)
        # dec_output = self.cluster_out(dec_output)
        # print(dec_output.shape)
        # dec_output = dec_output.reshape(x.shape[0], x.shape[1], self.d_model, -1)
        # dec_output = self.cluster_proj_out(dec_output)
        # dec_output = dec_output.permute(0, 2, 1)

        # return dec_output (32, 24, 128, 16)
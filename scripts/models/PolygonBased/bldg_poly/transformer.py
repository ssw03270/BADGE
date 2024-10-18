import torch
import torch.nn as nn
import numpy as np

from layer import EncoderLayer, DecoderLayer
from vqvae import VectorQuantizer, GumbelQuantize

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
    def __init__(self, n_layer, n_head, d_model, d_inner, dropout, n_tokens):
        """
        Initializes the TransformerEncoder.

        Parameters are for configuring the encoder layers and the positional encoding.
        """

        super(TransformerEncoder, self).__init__()

        self.pos_enc = PositionalEncoding(d_model, n_node=n_tokens)
        self.dropout = nn.Dropout(dropout)

        self.layer_stack = nn.ModuleList([
            EncoderLayer(d_model=d_model, d_inner=d_inner, n_head=n_head, dropout=dropout)
            for _ in range(n_layer)
        ])

    def forward(self, x, condition=None, enc_mask=None):
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
            enc_output = enc_layer(enc_output, enc_mask, condition)

        return enc_output

class Transformer(nn.Module):
    def __init__(self, d_model, d_inner, n_layer, n_head, dropout, grid_size, codebook_size, commitment_cost, n_tokens):
        """
        Initializes the Transformer model.
        """

        super(Transformer, self).__init__()

        self.d_model = d_model
        self.grid_size = grid_size

        self.tokens = nn.Parameter(torch.empty(n_tokens, d_model))
        self.tokens.data.uniform_(-1./d_model, 1./d_model)

        self.embedding = nn.Embedding(grid_size * grid_size + 3, d_model)
        self.encoder = TransformerEncoder(n_layer, n_head, d_model, d_inner, dropout, n_tokens)
        self.decoder = TransformerEncoder(n_layer, n_head, d_model, d_inner, dropout, n_tokens)

        self.vq = VectorQuantizer(codebook_size, d_model, commitment_cost)
        self.quantizer = GumbelQuantize(d_model, d_model, codebook_size)

        self.dropout = nn.Dropout(dropout)
        self.GELU = nn.GELU()

        self.coords_dec = nn.Linear(d_model * n_tokens, d_model * 16)
        self.coords_fc = nn.Linear(d_model, 2)

    def forward(self, bldg_coords, corner_coords, corner_indices):
        """
        Forward pass of the Transformer model.

        Processes coords features with attention mechanisms and positional encoding.
        """
        h = self.tokens.unsqueeze(0).repeat(bldg_coords.shape[0], 1, 1)  # (B, N, D)
        bldg_coords = (torch.mul(64 * torch.multiply(bldg_coords[:, :, 0] - corner_coords[:, :, 0], 
                                                bldg_coords[:, :, 1] - corner_coords[:, :, 1]).unsqueeze(2),  
                                                self.embedding(corner_indices[:, :, 0])) +
                      torch.mul(64 * torch.multiply(bldg_coords[:, :, 0] - corner_coords[:, :, 2], 
                                               bldg_coords[:, :, 1] - corner_coords[:, :, 3]).unsqueeze(2), 
                                               self.embedding(corner_indices[:, :, 1])) +
                      torch.mul(64 * torch.multiply(bldg_coords[:, :, 0] - corner_coords[:, :, 4], 
                                               bldg_coords[:, :, 1] - corner_coords[:, :, 5]).unsqueeze(2), 
                                               self.embedding(corner_indices[:, :, 2])) +
                      torch.mul(64 * torch.multiply(bldg_coords[:, :, 0] - corner_coords[:, :, 6], 
                                               bldg_coords[:, :, 1] - corner_coords[:, :, 7]).unsqueeze(2), 
                                               self.embedding(corner_indices[:, :, 3])))
        enc_output = self.encoder(h, bldg_coords)
        z, vq_loss, perplexity = self.vq(enc_output)
    
        dec_output = self.decoder(z).view(z.shape[0], -1)
        dec_output = self.coords_dec(dec_output).view(z.shape[0], bldg_coords.shape[1], z.shape[-1])
        dec_output = self.GELU(dec_output)

        coords_output = self.coords_fc(dec_output)
        coords_output = torch.sigmoid(coords_output)

        return coords_output, vq_loss, perplexity
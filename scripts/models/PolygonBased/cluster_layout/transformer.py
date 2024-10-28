import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from layer import EncoderLayer
from vqvae import VectorQuantizer

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

class ContinuousTransformer(nn.Module):
    def __init__(self, d_model, d_inner, n_layer, n_head, dropout, codebook_size, commitment_cost, n_tokens, sample_tokens):
        """
        Initializes the ContinuousTransformer model.
        """

        super(ContinuousTransformer, self).__init__()

        self.d_model = d_model
        self.n_tokens = n_tokens
        self.sample_tokens = sample_tokens

        self.encoding = nn.Linear(6, d_model)
        self.encoder = TransformerEncoder(n_layer, n_head, d_model, d_inner, dropout, n_tokens)
        self.decoder = TransformerEncoder(n_layer, n_head, d_model, d_inner, dropout, n_tokens)

        self.vq = VectorQuantizer(codebook_size, d_model, commitment_cost, sample_tokens)

        self.coords_fc = nn.Linear(d_model, 6)

    def forward(self, batch):
        """
        Forward pass of the Transformer model.

        Processes coords features with attention mechanisms and positional encoding.
        """
        x = self.encoding(batch)

        enc_output = self.encoder(x)

        z, vq_loss, perplexity = self.vq(enc_output)

        dec_output = self.decoder(z)

        coords_output = self.coords_fc(dec_output)
        coords_output = torch.sigmoid(coords_output)

        loss_coords = F.mse_loss(coords_output, batch.clone())

        return coords_output, loss_coords, vq_loss, perplexity

class DiscreteTransformer(nn.Module):
    def __init__(self, d_model, d_inner, n_layer, n_head, dropout, codebook_size, commitment_cost, n_tokens,
                 sample_tokens):
        """
        Initializes the DiscreteTransformer model.
        """

        super(DiscreteTransformer, self).__init__()

        self.d_model = d_model
        self.n_tokens = n_tokens * 6
        self.sample_tokens = sample_tokens
        self.bin = 64

        self.embed = nn.Embedding(self.bin, d_model)

        self.encoder = TransformerEncoder(n_layer, n_head, d_model, d_inner, dropout, n_tokens)
        self.decoder = TransformerEncoder(n_layer, n_head, d_model, d_inner, dropout, n_tokens)

        self.vq = VectorQuantizer(codebook_size, d_model, commitment_cost, sample_tokens)

        self.bbox_fc = nn.Linear(d_model, self.bin, bias=False)

        self.bbox_loss_fn = nn.CrossEntropyLoss()

    def forward(self, batch):
        """
        Forward pass of the Transformer model.

        Processes coords features with attention mechanisms and positional encoding.
        """
        bbox = batch.view(batch.shape[0], -1)
        bbox = self.embed(bbox)
        
        enc_output = self.encoder(bbox)

        z, vq_loss, perplexity = self.vq(enc_output)

        dec_output = self.decoder(z)

        dec_output = self.bbox_fc(dec_output)

        dec_output = torch.sigmoid(dec_output)

        bbox_output_flat = dec_output.view(-1, 64)
        bbox_labels_flat = batch.view(-1)
        bbox_loss = self.bbox_loss_fn(bbox_output_flat, bbox_labels_flat)

        return dec_output, bbox_loss, vq_loss, perplexity
    
    # def get_encoding_indices(self, batch):
    #     x = self.encoding(batch)
    #     enc_output = self.encoder(x)

    #     enc_output = enc_output.mean(dim=1, keepdim=True)  # 평균화하여 shape을 (batch, 1, feature dim)으로 변경
    #     encoding_indices = self.vq.get_encoding_indices(enc_output)

    #     return encoding_indices
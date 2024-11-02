import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision import models

import numpy as np

from layer import EncoderLayer, DecoderLayer
from diffusion_utils import *


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

class TransformerDecoder(nn.Module):
    def __init__(self, n_layer, n_head, d_model, d_inner, dropout, n_tokens, device):
        """
        Initializes the TransformerDecoder.

        Parameters are for configuring the encoder layers and the positional encoding.
        """

        super(TransformerDecoder, self).__init__()

        self.n_layer = n_layer

        self.encoding = nn.Linear(6, d_model)

        self.resnet18 = models.resnet18(pretrained=True)
        modules = list(self.resnet18.children())[:-1]  # Remove the last FC layer
        self.resnet18 = torch.nn.Sequential(*modules)
        self.resnet18.eval()

        self.pos_enc = PositionalEncoding(d_model, n_node=n_tokens)
        self.dropout = nn.Dropout(dropout)

        self.layer_stack = nn.ModuleList([
            DecoderLayer(d_model=d_model, d_inner=d_inner, n_head=n_head, dropout=dropout)
            for _ in range(n_layer)
        ])
        
        self.decoding = nn.Linear(d_model, 6)

    def forward(self, x, condition, timestep):
        """
        Forward pass for the BoundaryEncoder.

        Parameters:
        - x (Tensor): The x features.
        - enc_mask (Tensor): The encoder mask.

        Returns:
        - Tensor: The encoded features.
        """

        x = self.encoding(x)
        x = F.softplus(x)
        dec_input = x + self.pos_enc(x).expand(x.shape[0], -1, -1)
        dec_output = self.dropout(dec_input)

        condition = self.resnet18(condition).squeeze().unsqueeze(1)

        for idx, dec_layer in enumerate(self.layer_stack):
            dec_output = dec_layer(dec_output, condition, timestep)

            if idx < self.n_layer - 1:
                dec_output = F.softplus(dec_output)

        dec_output = self.decoding(dec_output)

        return dec_output

class Diffusion(nn.Module):
    def __init__(self, num_timesteps, n_head, d_model, d_inner, seq_dim, n_layer, device, ddim_num_steps, dropout,
                 beta_schedule='cosine'):
        """
        Initializes the Diffusion model.
        """

        super(Diffusion, self).__init__()

        self.network = TransformerDecoder(n_layer, n_head, d_model, d_inner, dropout, 300, device)

        self.device = device
        self.num_timesteps = num_timesteps
        betas = make_beta_schedule(schedule=beta_schedule, num_timesteps=self.num_timesteps, start=0.0001, end=0.02)
        betas = self.betas = betas.float().to(self.device)
        self.betas_sqrt = torch.sqrt(betas)
        alphas = 1.0 - betas
        self.alphas = alphas
        self.one_minus_betas_sqrt = torch.sqrt(alphas)
        self.alphas_cumprod = alphas.cumprod(dim=0)
        self.alphas_bar_sqrt = torch.sqrt(self.alphas_cumprod)
        self.one_minus_alphas_bar_sqrt = torch.sqrt(1 - self.alphas_cumprod)
        alphas_cumprod_prev = torch.cat([torch.ones(1).to(self.device), self.alphas_cumprod[:-1]], dim=0)
        self.alphas_cumprod_prev = alphas_cumprod_prev
        self.posterior_mean_coeff_1 = (betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.posterior_mean_coeff_2 = (torch.sqrt(alphas) * (1 - alphas_cumprod_prev) / (1 - self.alphas_cumprod))
        posterior_variance = (betas * (1.0 - alphas_cumprod_prev) / (1.0 - self.alphas_cumprod))
        self.posterior_variance = posterior_variance
        self.logvar = betas.log()

        self.ddim_num_steps = ddim_num_steps
        self.make_ddim_schedule(ddim_num_steps)

    def make_ddim_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0.):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.num_timesteps)

        assert self.alphas_cumprod.shape[0] == self.num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.device)

        self.register_buffer('sqrt_alphas_cumprod', to_torch(torch.sqrt(self.alphas_cumprod)))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(torch.sqrt(1. - self.alphas_cumprod)))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(torch.log(1. - self.alphas_cumprod)))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(torch.sqrt(1. / self.alphas_cumprod)))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(torch.sqrt(1. / self.alphas_cumprod - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=self.alphas_cumprod,
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', torch.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    def sample_t(self, size=(1,), t_max=None):
       """Samples batches of time steps to use."""
       if t_max is None:
           t_max = int(self.num_timesteps) - 1

       t = torch.randint(low=0, high=t_max, size=size, device=self.device)

       return t.to(self.device)

    def forward(self, layout, image_mask, t, reparam=True, train_type='generation'):
        e = torch.randn_like(layout).to(layout.device)
        l_t_noise = q_sample(layout, self.alphas_bar_sqrt,
                             self.one_minus_alphas_bar_sqrt, t, noise=e)
        
        if train_type == 'conditional':
            l_t_noise[:, :, 2:4] = layout[:, :, 2:4]
            l_t_noise[:, :, 5] = layout[:, :,5]

        eps_theta = self.network(l_t_noise, image_mask, timestep=t)

        if reparam:
            sqrt_one_minus_alpha_bar_t = extract(self.one_minus_alphas_bar_sqrt, t, l_t_noise)
            sqrt_alpha_bar_t = (1 - sqrt_one_minus_alpha_bar_t.square()).sqrt()
            l_0_generate_reparam = 1 / sqrt_alpha_bar_t * (l_t_noise - eps_theta * sqrt_one_minus_alpha_bar_t).to(self.device)

            return eps_theta, e, l_0_generate_reparam
        else:
            return eps_theta, e, None


    def reverse_ddim(self, real_layout, image_mask, stochastic=True, train_type='generation', inference_type='refine'):
        if train_type == 'generation':
            layout_t_0, intermediates = \
                ddim_sample_loop(self.network, real_layout, image_mask, self.ddim_timesteps, self.ddim_alphas,
                                    self.ddim_alphas_prev, self.ddim_sigmas, stochastic=stochastic)
        elif train_type == 'conditional':
            layout_t_0, intermediates = \
                ddim_cond_sample_loop(self.network, real_layout, image_mask, self.ddim_timesteps, self.ddim_alphas,
                                      self.ddim_alphas_prev, self.ddim_sigmas, stochastic=stochastic, inference_type=inference_type)

        return layout_t_0
from typing import *

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import Tensor, LongTensor

import numpy as np

from transformer import Transformer

################################################################


## Helper functions
def avg_except_batch(x: Tensor, num_dims=1):
    return x.reshape(*x.shape[:num_dims], -1).mean(dim=-1)


def sum_except_batch(x: Tensor, num_dims=1):
    return x.reshape(*x.shape[:num_dims], -1).sum(dim=-1)


def log_1_min_a(a: Tensor):
    return torch.log(1. - a.exp() + 1e-40)


def log_add_exp(a: Tensor, b: Tensor):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))


def extract(a: Tensor, t: LongTensor, x_shape: Tuple[int, ...]):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def log_categorical(log_x_start: Tensor, log_prob: Tensor):
    return (log_x_start.exp() * log_prob).sum(dim=1)


def index_to_log_onehot(x: LongTensor, num_classes: int):
    assert x.max().item() < num_classes, f"Error: {x.max().item()} >= {num_classes}"

    x_onehot: Tensor = F.one_hot(x, num_classes)
    permute_order = (0, -1) + tuple(range(1, len(x.shape)))
    x_onehot = x_onehot.permute(permute_order)
    log_x = torch.log(x_onehot.float().clamp(min=1e-30))
    return log_x


def log_onehot_to_index(log_x: Tensor):
    return log_x.argmax(dim=1)


def alpha_schedule(time_step: int, N: int, att_1=0.99999, att_T=0.000009, ctt_1=0.000009, ctt_T=0.99999):
    att = np.arange(0, time_step) / (time_step-1) * (att_T - att_1) + att_1
    att = np.concatenate([[1], att])
    at = att[1:] / att[:-1]

    ctt = np.arange(0, time_step) / (time_step-1) * (ctt_T - ctt_1) + ctt_1
    ctt = np.concatenate([[0], ctt])
    one_minus_ctt = 1. - ctt
    one_minus_ct = one_minus_ctt[1:] / one_minus_ctt[:-1]
    ct = 1. - one_minus_ct

    bt = (1. - at - ct) / N

    att = np.concatenate([att[1:], [1]])
    ctt = np.concatenate([ctt[1:], [0]])
    btt = (1. - att - ctt) / N

    return at, bt, ct, att, btt, ctt


################################################################

class Diffusion(nn.Module):
    def __init__(self, d_model, d_inner, n_layer, n_head, dropout, codebook_size, codebook_dim, cluster_count):
        """
        Initializes the Diffusion model.
        """

        super(Diffusion, self).__init__()

        self.auxiliary_loss_weight = 5e-4
        self.num_timesteps = 100
        self.mask_weight = [1., 1.]
        self.adaptive_auxiliary_loss = True
        self.num_codebook_size = codebook_size + 2
        self.codebook_dim = codebook_dim
        self.num_cluster_classes = cluster_count + 2

        at, bt, ct, att, btt, ctt = alpha_schedule(self.num_timesteps, N=self.num_codebook_size - 1)  # +1 for empty node

        at = torch.tensor(at.astype("float64"))
        bt = torch.tensor(bt.astype("float64"))
        ct = torch.tensor(ct.astype("float64"))
        log_at = torch.log(at)
        log_bt = torch.log(bt)
        log_ct = torch.log(ct)
        att = torch.tensor(att.astype("float64"))
        btt = torch.tensor(btt.astype("float64"))
        ctt = torch.tensor(ctt.astype("float64"))
        log_cumprod_at = torch.log(att)
        log_cumprod_bt = torch.log(btt)
        log_cumprod_ct = torch.log(ctt)

        log_1_min_ct = log_1_min_a(log_ct)
        log_1_min_cumprod_ct = log_1_min_a(log_cumprod_ct)

        assert log_add_exp(log_ct, log_1_min_ct).abs().sum().item() < 1e-5
        assert log_add_exp(log_cumprod_ct, log_1_min_cumprod_ct).abs().sum().item() < 1e-5

        # Convert to float32 and register buffers.
        self.register_buffer("log_at", log_at.float())
        self.register_buffer("log_bt", log_bt.float())
        self.register_buffer("log_ct", log_ct.float())
        self.register_buffer("log_cumprod_at", log_cumprod_at.float())
        self.register_buffer("log_cumprod_bt", log_cumprod_bt.float())
        self.register_buffer("log_cumprod_ct", log_cumprod_ct.float())
        self.register_buffer("log_1_min_ct", log_1_min_ct.float())
        self.register_buffer("log_1_min_cumprod_ct", log_1_min_cumprod_ct.float())

        self.register_buffer("Lt_history", torch.zeros(self.num_timesteps))
        self.register_buffer("Lt_count", torch.zeros(self.num_timesteps))

        self.network = Transformer(d_model, d_inner, n_layer, n_head, dropout, self.num_codebook_size, codebook_dim, self.num_cluster_classes)

    ################################################################

    def multinomial_kl(self, log_prob1: Tensor, log_prob2: Tensor):  # compute KL loss on log_prob
        kl = (log_prob1.exp() * (log_prob1 - log_prob2)).sum(dim=1)
        return kl

    def q_pred_one_timestep(self, log_x_t: Tensor, t: LongTensor, num_classes: int):  # q(xt|xt_1)
        log_at = extract(self.log_at, t, log_x_t.shape)  # at
        log_bt = extract(
            (self.log_bt.exp() * (self.num_codebook_size-1) / (num_classes-1)).log(),
            t, log_x_t.shape
        )  # bt; -1 for [mask] token
        log_ct = extract(self.log_ct, t, log_x_t.shape)  # ct
        log_1_min_ct = extract(self.log_1_min_ct, t, log_x_t.shape)  # 1-ct

        _prob_sum = log_at.exp() + log_bt.exp() * (num_classes-1) + log_ct.exp()
        assert torch.allclose(_prob_sum, torch.ones_like(_prob_sum))

        log_probs = torch.cat(
            [
                log_add_exp(log_x_t[:, :-1, ...] + log_at, log_bt),
                log_add_exp(log_x_t[:, -1:, ...] + log_1_min_ct, log_ct)
            ],
            dim=1
        )

        return log_probs

    def q_pred(self, log_x_start: Tensor, t: LongTensor, num_classes: int):  # q(xt|x0)
        t = (t + (self.num_timesteps + 1))%(self.num_timesteps + 1)
        log_cumprod_at = extract(self.log_cumprod_at, t, log_x_start.shape)  # at~
        log_cumprod_bt = extract(
            (self.log_cumprod_bt.exp() * (self.num_codebook_size-1) / (num_classes-1)).log(),
            t, log_x_start.shape
        )  # bt~; -1 for [mask] token
        log_cumprod_ct = extract(self.log_cumprod_ct, t, log_x_start.shape)  # ct~
        log_1_min_cumprod_ct = extract(self.log_1_min_cumprod_ct, t, log_x_start.shape)  # 1-ct~

        _prob_sum = log_cumprod_at.exp() + log_cumprod_bt.exp() * (num_classes-1) + log_cumprod_ct.exp()
        assert torch.allclose(_prob_sum, torch.ones_like(_prob_sum))

        log_probs = torch.cat(
            [
                log_add_exp(log_x_start[:, :-1, ...] + log_cumprod_at, log_cumprod_bt),
                log_add_exp(log_x_start[:, -1:, ...] + log_1_min_cumprod_ct, log_cumprod_ct)
            ],
            dim=1
        )

        return log_probs
    
    def predict_start(self, log_x_t, condition, t, truncation_rate=1.):
        x_t: LongTensor = log_onehot_to_index(log_x_t)

        out_x = self.network(x_t, condition, t)
        out_x: Tensor

        log_pred_x = F.log_softmax(out_x.double(), dim=1).float()  # (batch_size, num_cluster_classes, num_dimension * codebook_size)
        batch_size = log_x_t.shape[0]

        log_pred_x = self.truncate(log_pred_x, truncation_rate)

        zero_vector_x = torch.zeros(batch_size, 1, *out_x.shape[2:]).type_as(log_x_t) - 70.
        log_pred_x = torch.cat([log_pred_x, zero_vector_x], dim=1)  # (batch_size, num_cluster_classes + 1, num_dimension * codebook_size)
        log_pred_x = torch.clamp(log_pred_x, -70., 0.)

        return log_pred_x

    def q_posterior(self,
        log_x_start: Tensor, log_x_t: Tensor, t: LongTensor,
        num_classes: int
    ):  # p_theta(xt_1|xt) = sum( q(xt-1|xt,x0') * p(x0') )
        assert t.min().item() >= 0 and t.max().item() < self.num_timesteps

        batch_size = log_x_start.shape[0]
        onehot_x_t = log_onehot_to_index(log_x_t)
        mask = (onehot_x_t == num_classes-1).unsqueeze(1)  # -1 for [mask] token
        log_one_vector = torch.zeros(batch_size, 1, *([1] * (len(log_x_start.shape)-2))).type_as(log_x_t)
        log_zero_vector = torch.log(log_one_vector+1e-30).expand(-1, -1, *log_x_start.shape[2:])

        log_qt = self.q_pred(log_x_t, t, num_classes)  # q(xt|x0)
        log_qt = log_qt[:, :-1, ...]
        log_cumprod_ct = extract(self.log_cumprod_ct, t, log_x_start.shape)  # ct~
        ct_cumprod_vector = log_cumprod_ct.expand(-1, num_classes-1, *([-1]*(len(log_x_start.shape)-2)))
        log_qt = (~mask)*log_qt + mask*ct_cumprod_vector
        log_qt_one_timestep = self.q_pred_one_timestep(log_x_t, t, num_classes)  # q(xt|xt_1)
        log_qt_one_timestep = torch.cat([log_qt_one_timestep[:, :-1, ...], log_zero_vector], dim=1)
        log_ct = extract(self.log_ct, t, log_x_start.shape)  # ct
        ct_vector = log_ct.expand(-1, num_classes-1, *([-1]*(len(log_x_start.shape)-2)))
        ct_vector = torch.cat([ct_vector, log_one_vector], dim=1)
        log_qt_one_timestep = (~mask)*log_qt_one_timestep + mask*ct_vector

        q = log_x_start[:, :-1, ...] - log_qt
        q = torch.cat([q, log_zero_vector], dim=1)
        q_log_sum_exp = torch.logsumexp(q, dim=1, keepdim=True)
        q = q - q_log_sum_exp
        log_EV_xtmin_given_xt_given_xstart = self.q_pred(q, t-1, num_classes) + log_qt_one_timestep + q_log_sum_exp

        return torch.clamp(log_EV_xtmin_given_xt_given_xstart, -70., 0.)
    
    def p_pred(self,
        log_x: Tensor, t: LongTensor, condition: Tensor, truncation_rate=1.
    ):  # if x0, first p(x0|xt), than sum( q(xt-1|xt,x0) * p(x0|xt) )
        if self.parametrization == "x0":
            log_x_recon = self.predict_start(log_x, condition, t, truncation_rate)
            log_model_pred_x = self.q_posterior(
                log_x_start=log_x_recon, log_x_t=log_x, t=t,
                num_classes=self.num_codebook_size
            )
        elif self.parametrization == "direct":
            log_x_recon, log_e_recon, log_o_recon = None, None, None
            log_model_pred_x, log_model_pred_e, log_model_pred_o = self.predict_start(log_x, condition, t, truncation_rate)
        else:
            raise ValueError

        return log_model_pred_x, log_x_recon

    @torch.no_grad()
    def p_sample(self,
        log_x: Tensor, t: LongTensor, condition: Tensor, truncation_rate=1.
    ):  # sample q(xt-1) for next step from xt, actually is p(xt-1|xt)
        log_model_pred_x, _,  = self.p_pred(log_x, t, condition, truncation_rate)

        # Gumbel sample
        out_x = self.log_sample_categorical(log_model_pred_x, self.num_codebook_size)
        return out_x

    def log_sample_categorical(self, logits: Tensor, num_classes: int):  # use gumbel to sample onehot vector from log probability
        uniform = torch.rand_like(logits)
        gumbel_noise = -torch.log(-torch.log(uniform + 1e-30) + 1e-30)
        sample = (gumbel_noise + logits).argmax(dim=1)

        log_sample = index_to_log_onehot(sample, num_classes)
        return log_sample

    def q_sample(self, log_x_start: Tensor, t: LongTensor, num_classes: int):  # diffusion step, q(xt|x0) and sample xt
        log_EV_qxt_x0 = self.q_pred(log_x_start, t, num_classes)

        # Gumbel sample
        log_sample = self.log_sample_categorical(log_EV_qxt_x0, num_classes)
        return log_sample

    def sample_time(self, b: int, device: torch.device, method="uniform"):
        if method == "importance":
            if not (self.Lt_count > 10).all():
                return self.sample_time(b, device, method="uniform")

            Lt_sqrt = torch.sqrt(self.Lt_history + 1e-10) + 0.0001
            Lt_sqrt[0] = Lt_sqrt[1]  # overwrite L0 (i.e., the decoder nll) term with L1
            pt_all = Lt_sqrt / Lt_sqrt.sum()

            t = torch.multinomial(pt_all, num_samples=b, replacement=True)
            pt = pt_all.gather(dim=0, index=t)
            return t, pt
        elif method == "uniform":
            t = torch.randint(0, self.num_timesteps, (b,), device=device).long()
            pt = torch.ones_like(t).float() / self.num_timesteps
            return t, pt
        else:
            raise ValueError
    
    def truncate(self, log_p_x_0: Tensor, truncation_rate=1.):
        sorted_log_p_x_0, indices = torch.sort(log_p_x_0, dim=1, descending=True)
        sorted_p_x_0 = torch.exp(sorted_log_p_x_0)
        keep_mask = sorted_p_x_0.cumsum(dim=1) < truncation_rate

        # Ensure that at least the largest probability is not zeroed out
        all_true = torch.full_like(keep_mask[:, 0:1, :], True)
        keep_mask = torch.cat((all_true, keep_mask), dim=1)
        keep_mask = keep_mask[:, :-1, :]

        keep_mask = keep_mask.gather(1, indices.argsort(1))

        rv = log_p_x_0.clone()
        rv[~keep_mask] = -torch.inf  # -inf = log(0)
        return rv
    
    def compute_losses(self, x, condition):
        batch_size, device = x.shape[0], x.device

        x_start = x
        t, pt = self.sample_time(batch_size, device)

        log_x_start = index_to_log_onehot(x_start, self.num_codebook_size)  # (batch_size, num_cluster_classes, num_dimension * codebook_size) # (B, Cx, N)
        log_xt = self.q_sample(log_x_start=log_x_start, t=t, num_classes=self.num_codebook_size)
        xt = log_onehot_to_index(log_xt)

        # Go to p_theta function
        log_x0_recon = self.predict_start(log_xt, condition, t)    # P_theta(x0|xt)
        log_model_prob_x = self.q_posterior(log_x_start=log_x0_recon, log_x_t=log_xt, t=t, num_classes=self.num_codebook_size)  # go through q(xt_1|xt,x0)

        # Compute log_true_prob now 
        log_true_prob_x = self.q_posterior(log_x_start=log_x_start, log_x_t=log_xt, t=t, num_classes=self.num_codebook_size)
        
        # Compute loss
        kl_x = self.multinomial_kl(log_true_prob_x, log_model_prob_x)  # (B, N)
        mask_region_x = (xt == self.num_codebook_size-1).float()
        mask_weight_x = mask_region_x * self.mask_weight[0] + (1. - mask_region_x) * self.mask_weight[1]
        kl_x = kl_x * mask_weight_x    # (B, N)
        kl_x = avg_except_batch(kl_x)  # (B,)
        
        decoder_nll_x = -log_categorical(log_x_start, log_model_prob_x)  # (B, N)
        decoder_nll_x = avg_except_batch(decoder_nll_x)  # (B,)
        
        mask = (t == torch.zeros_like(t)).float()
        kl_loss_x = mask * decoder_nll_x + (1. - mask) * kl_x  # (B,)

        # Record for importance sampling
        Lt2 = kl_loss_x.pow(2)  # (B,);
        Lt2_prev = self.Lt_history.gather(dim=0, index=t)
        new_Lt_history = (0.1 * Lt2 + 0.9 * Lt2_prev).detach()
        self.Lt_history.scatter_(dim=0, index=t, src=new_Lt_history)
        self.Lt_count.scatter_add_(dim=0, index=t, src=torch.ones_like(Lt2))
        
        # Upweigh loss term of the kl
        loss1_x  = kl_loss_x / pt
        vb_loss_x = loss1_x
        if self.auxiliary_loss_weight != 0.:
            kl_aux_x = self.multinomial_kl(log_x_start[:, :-1, ...], log_x0_recon[:, :-1, ...])
            kl_aux_x = kl_aux_x * mask_weight_x    # (B, N)
            kl_aux_x = avg_except_batch(kl_aux_x)  # (B,)
            kl_aux_loss_x = mask * decoder_nll_x + (1. - mask) * kl_aux_x

            if self.adaptive_auxiliary_loss:
                addition_loss_weight = (1. - t/self.num_timesteps) + 1.
            else:
                addition_loss_weight = 1.

            loss2_x = addition_loss_weight * self.auxiliary_loss_weight * kl_aux_loss_x / pt
            vb_loss_x += loss2_x  # (B,)

        return vb_loss_x.mean() / x.shape[-1]
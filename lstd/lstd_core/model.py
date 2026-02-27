# lstd_core/model.py

from __future__ import annotations

from typing import Dict, Any, Optional, cast

import torch
from torch import nn
import torch.distributions as D

from .config import LSTDModelConfig
from .components import (
    LongTermEncoder,
    ShortTermEncoder,
    VariationalBlock,
    LongPredictor,
    ShortPredictor,
    HistoricalDecoder,
    FuturePredictor,
    NPTransitionPrior,
)


class LSTDNet(nn.Module):
    """
    Paper-faithful LSTD model based on Table 3.

    Forward returns:
        x_rec    : reconstructed history       [B, seq_len, enc_in]
        y_flat   : flattened future forecast   [B, pred_len * enc_in]
        other    : KL + smooth + sparse losses
    """

    def __init__(self, cfg: LSTDModelConfig, device: Optional[torch.device] = None):
        super().__init__()
        self.cfg = cfg

        if cfg.mode != "feature":
            raise ValueError(
                "This rewritten architecture matches the paper's feature-based Table 3 path. "
                "Use model.mode='feature'."
            )

        # 1) Encoders phi^s, phi^d
        self.long_encoder = LongTermEncoder(
            seq_len=cfg.seq_len,
            enc_in=cfg.enc_in,
            conv_hidden=cfg.long_conv_hidden,
        )
        self.short_encoder = ShortTermEncoder(
            seq_len=cfg.seq_len,
            hidden_dim=cfg.short_mlp_hidden,
        )

        # Variational blocks for historical latents
        self.long_variational = VariationalBlock(cfg.enc_in)
        self.short_variational = VariationalBlock(cfg.enc_in)

        # 2) Latent transition modules T^s, T^d
        self.long_predictor = LongPredictor(
            seq_len=cfg.seq_len,
            pred_len=cfg.pred_len,
            hidden_dim=cfg.short_mlp_hidden,
        )
        self.short_predictor = ShortPredictor(
            seq_len=cfg.seq_len,
            pred_len=cfg.pred_len,
        )

        # Variational blocks for future latent states
        self.long_future_variational = VariationalBlock(cfg.enc_in)
        self.short_future_variational = VariationalBlock(cfg.enc_in)

        # 3) Decoders F_x, F_y
        self.history_decoder = HistoricalDecoder(cfg.enc_in)
        self.future_predictor = FuturePredictor(
            enc_in=cfg.enc_in,
            hidden_dim=cfg.future_mlp_hidden,
        )

        # 4) Prior networks r_i^s, r_i^d
        self.long_prior = NPTransitionPrior(
            lags=cfg.lags,
            latent_size=cfg.enc_in,
            num_hidden_layers=cfg.prior_num_hidden_layers,
            hidden_dim=cfg.prior_hidden_dim,
        )
        self.short_prior = NPTransitionPrior(
            lags=cfg.lags,
            latent_size=cfg.enc_in,
            num_hidden_layers=cfg.prior_num_hidden_layers,
            hidden_dim=cfg.prior_hidden_dim,
        )

        # Long-term smoothness constraint
        self.attention = nn.MultiheadAttention(
            embed_dim=cfg.enc_in,
            num_heads=1,
            batch_first=True,
        )

        self.register_buffer("stationary_dist_mean", torch.zeros(cfg.enc_in))
        self.register_buffer("stationary_dist_var", torch.eye(cfg.enc_in))

        if device is not None:
            self.to(device)

    from typing import cast

    @property
    def stationary_dist(self) -> D.MultivariateNormal:
        mean = cast(torch.Tensor, self.stationary_dist_mean)
        cov = cast(torch.Tensor, self.stationary_dist_var)
        return D.MultivariateNormal(mean, cov)

    def _kl_from_prior(
        self,
        mu_full: torch.Tensor,
        logvar_full: torch.Tensor,
        z_full: torch.Tensor,
        prior_net: NPTransitionPrior,
    ) -> torch.Tensor:
        """
        KL(q(z) || p(z)) where p(z) is defined through the modular
        inverse transition prior network.
        """
        q_dist = D.Normal(mu_full, torch.exp(0.5 * logvar_full))
        log_qz = q_dist.log_prob(z_full)

        lags = self.cfg.lags

        # Initial prior for the first 'lags' time steps
        p0 = D.Normal(
            torch.zeros_like(mu_full[:, :lags]),
            torch.ones_like(logvar_full[:, :lags]),
        )
        log_p0 = p0.log_prob(z_full[:, :lags]).sum(dim=(-1, -2))
        log_q0 = log_qz[:, :lags].sum(dim=(-1, -2))
        kld_init = (log_q0 - log_p0).mean()

        # Transition prior for the remaining steps
        residuals, logabsdet = prior_net(z_full)
        log_q_future = log_qz[:, lags:].sum(dim=(-1, -2))
        log_p_future = self.stationary_dist.log_prob(residuals).sum(dim=1) + logabsdet.sum(dim=1)

        denom = max(1, z_full.shape[1] - lags)
        kld_future = ((log_q_future - log_p_future) / denom).mean()

        return kld_init + kld_future

    def _smooth_constraint(self, z_s_hist: torch.Tensor) -> torch.Tensor:
        """
        Paper long-term smoothness constraint:
        compare association matrices of early vs late long-term latents.
        """
        half = max(1, z_s_hist.shape[1] // 2)

        z_s_early = z_s_hist[:, :half, :]
        z_s_late = z_s_hist[:, -half:, :]

        _, w1 = self.attention(z_s_early, z_s_early, z_s_early)
        _, w2 = self.attention(z_s_late, z_s_late, z_s_late)

        return torch.mean((w1 - w2) ** 2)

    def _sparse_dependency_constraint(self, z_d_full: torch.Tensor) -> torch.Tensor:
        """
        Paper-style short-term interrupted dependency penalty.

        We penalize the dependence of the final short-term residual epsilon_H^d
        on earlier short-term states z_{tau-1}^d.

        NOTE:
        This is second-order and can be expensive. That is expected.
        """
        if z_d_full.shape[1] <= self.cfg.lags + 1:
            return z_d_full.new_tensor(0.0)

        # We need higher-order gradients so do NOT detach.
        z_req = z_d_full.requires_grad_(True)
        residuals, _ = self.short_prior(z_req)

        final_eps = residuals[:, -1, :].sum()

        grads = torch.autograd.grad(
            final_eps,
            z_req,
            retain_graph=True,
            create_graph=True,
            allow_unused=False,
        )[0]

        # Penalize dependency on earlier short-term states
        early_grads = grads[:, :-1, :]
        return early_grads.abs().mean()

    def forward(self, x: torch.Tensor, is_training: bool = True, return_latents: bool = False):
        x = x.float()

        # 1) Encode historical states
        h_s_hist = self.long_encoder(x)     # [B, L, C]
        h_d_hist = self.short_encoder(x)    # [B, L, C]

        z_s_hist, mu_s_hist, logvar_s_hist = self.long_variational(
            h_s_hist,
            is_training=is_training,
        )
        z_d_hist, mu_d_hist, logvar_d_hist = self.short_variational(
            h_d_hist,
            is_training=is_training,
        )

        # 2) Predict future latent states
        h_s_future = self.long_predictor(z_s_hist)    # [B, pred_len, C]
        h_d_future = self.short_predictor(z_d_hist)   # [B, pred_len, C]

        z_s_future, mu_s_future, logvar_s_future = self.long_future_variational(
            h_s_future,
            is_training=is_training,
        )
        z_d_future, mu_d_future, logvar_d_future = self.short_future_variational(
            h_d_future,
            is_training=is_training,
        )

        # 3) Decode history and forecast future
        x_rec = self.history_decoder(z_s_hist, z_d_hist)        # [B, L, C]
        y = self.future_predictor(z_s_future, z_d_future)       # [B, pred_len, C]

         
        # 4) Auxiliary losses
        z_s_full = torch.cat([z_s_hist, z_s_future], dim=1)
        z_d_full = torch.cat([z_d_hist, z_d_future], dim=1)

        mu_s_full = torch.cat([mu_s_hist, mu_s_future], dim=1)
        mu_d_full = torch.cat([mu_d_hist, mu_d_future], dim=1)

        logvar_s_full = torch.cat([logvar_s_hist, logvar_s_future], dim=1)
        logvar_d_full = torch.cat([logvar_d_hist, logvar_d_future], dim=1)

        smooth_loss = self._smooth_constraint(z_s_hist) if self.cfg.L2_weight != 0 else x.new_tensor(0.0)

        # The sparse dependency constraint needs autograd.
        # Do not compute it when its weight is zero, or when gradients are globally disabled.
        if self.cfg.L1_weight != 0.0 and torch.is_grad_enabled():
            sparse_loss = self._sparse_dependency_constraint(z_d_full)
        else:
            sparse_loss = x.new_tensor(0.0)

        if is_training:
            zs_kl_loss = self._kl_from_prior(
                mu_full=mu_s_full,
                logvar_full=logvar_s_full,
                z_full=z_s_full,
                prior_net=self.long_prior,
            )
            zd_kl_loss = self._kl_from_prior(
                mu_full=mu_d_full,
                logvar_full=logvar_d_full,
                z_full=z_d_full,
                prior_net=self.short_prior,
            )

            other_loss = (
                self.cfg.zc_kl_weight * zs_kl_loss
                + self.cfg.zd_kl_weight * zd_kl_loss
                + self.cfg.L2_weight * smooth_loss
                + self.cfg.L1_weight * sparse_loss
            )
        else:
            other_loss = (
                self.cfg.L2_weight * smooth_loss
                + self.cfg.L1_weight * sparse_loss
            )
        y_flat = y.reshape(y.shape[0], -1)

        if return_latents:
            extras: Dict[str, Any] = {
                "y": y,
                "z_s_hist": z_s_hist,
                "z_d_hist": z_d_hist,
                "z_s_future": z_s_future,
                "z_d_future": z_d_future,
                "smooth_loss": smooth_loss.detach(),
                "sparse_loss": sparse_loss.detach(),
            }
            return x_rec, y_flat, other_loss, extras

        return x_rec, y_flat, other_loss

    @torch.no_grad()
    def store_grad(self):
        """
        Kept only for trainer compatibility.
        The old TS2Vec/FSNet hybrid needed this; the rewritten Table-3 model does not.
        """
        return None
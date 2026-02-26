# lstd_core/model.py

from __future__ import annotations

from typing import Dict, Any, Optional

import torch
from torch import nn
import torch.distributions as D

from .config import LSTDModelConfig
from .components import BaseNet, NPTransitionPrior
from .ts2vec.fsnet import TSEncoder


class LSTDNet(nn.Module):
    """
    Core LSTD network, packaged from exp_LSTD.py::net with bug fixes and cleanup.

    forward() returns:
        x_rec:    reconstructed input sequence [B, seq_len, enc_in]
        y_flat:   flattened forecast [B, pred_len * enc_in]
        other:    KL + regularization terms
    """
    def __init__(self, cfg: LSTDModelConfig, device: Optional[torch.device] = None):
        super().__init__()
        self.cfg = cfg
        self.device_ref = device

        # Transition prior on "dynamic" latent
        self.stationary_transition_prior = NPTransitionPrior(
            lags=cfg.lags,
            latent_size=cfg.enc_in,
            num_layers=1,
            hidden_dim=cfg.hidden_dim,
        )

        # TS2Vec encoder setup mirrors repo's mode-specific setup
        if cfg.mode == "time":
            ts_encoder = TSEncoder(
                input_dims=cfg.seq_len,                 # time axis treated as feature axis
                output_dims=cfg.ts_output_dims,
                hidden_dims=cfg.ts_hidden_dims,
                depth=cfg.depth,
                use_adaptive_memory_conv=cfg.use_adaptive_memory_conv,
                gamma=cfg.gamma,
                tau=cfg.tau,
                mask_mode="all_true",
            )

            self.encoder = ts_encoder

            self.regressor = BaseNet(
                input_len=2 * cfg.pred_len,
                out_len=cfg.pred_len,
                hidden_dim=cfg.hidden_dim,
                input_dim=cfg.enc_in,
                out_dim=cfg.enc_in,
                is_mean_std=False,
                activation=cfg.activation,
                layer_norm=False,
                c_type="None",
                drop_out=cfg.dropout,
                layer_nums=cfg.hidden_layers,
            )

            self.zs_rec = nn.Linear(cfg.ts_output_dims, 2 * cfg.seq_len)
            self.x_rec = nn.Linear(2 * cfg.seq_len, cfg.seq_len)
            self.zs_pred = nn.Linear(cfg.seq_len, 2 * cfg.pred_len)

            self.zd_rec = BaseNet(
                input_len=cfg.seq_len,
                out_len=2 * cfg.seq_len,
                hidden_dim=cfg.hidden_dim,
                input_dim=cfg.enc_in,
                out_dim=cfg.enc_in,
                is_mean_std=False,
                activation=cfg.activation,
                layer_norm=True,
                c_type="None",
                drop_out=cfg.dropout,
                layer_nums=cfg.hidden_layers,
            )
            self.zd_pred = nn.Linear(cfg.seq_len, 2 * cfg.pred_len)
        else:
            ts_encoder = TSEncoder(
                input_dims=cfg.enc_in,
                output_dims=cfg.ts_output_dims,
                hidden_dims=cfg.ts_hidden_dims,
                depth=cfg.depth,
                use_adaptive_memory_conv=cfg.use_adaptive_memory_conv,
                gamma=cfg.gamma,
                tau=cfg.tau,
                mask_mode="all_true",
            )

            self.encoder = ts_encoder

            self.regressor = BaseNet(
                input_len=2 * cfg.enc_in,
                out_len=cfg.enc_in,
                hidden_dim=cfg.hidden_dim,
                input_dim=2 * cfg.enc_in,
                out_dim=cfg.enc_in,
                is_mean_std=False,
                activation=cfg.activation,
                layer_norm=False,
                c_type="type1",
                drop_out=cfg.dropout,
                layer_nums=cfg.hidden_layers,
            )

            self.zs_rec = nn.Linear(cfg.ts_output_dims, 2 * cfg.enc_in)
            self.x_rec = nn.Linear(2 * cfg.enc_in, cfg.enc_in)
            self.zs_pred = nn.Linear(cfg.seq_len, 2 * cfg.pred_len)

            self.zd_rec = BaseNet(
                input_len=cfg.enc_in,
                out_len=2 * cfg.enc_in,
                hidden_dim=cfg.hidden_dim,
                input_dim=cfg.enc_in,
                out_dim=2 * cfg.enc_in,
                is_mean_std=False,
                activation=cfg.activation,
                layer_norm=True,
                c_type="type1",
                drop_out=cfg.dropout,
                layer_nums=cfg.hidden_layers,
            )
            self.zd_pred = nn.Linear(cfg.seq_len, 2 * cfg.pred_len)

        self.attention = nn.MultiheadAttention(embed_dim=cfg.enc_in, num_heads=1, batch_first=True)
        self.sigmoid = nn.Sigmoid()
        
        self.stationary_dist_mean: torch.Tensor
        self.stationary_dist_var: torch.Tensor
        self.register_buffer("stationary_dist_mean", torch.zeros(cfg.enc_in))
        self.register_buffer("stationary_dist_var", torch.eye(cfg.enc_in))

        if device is not None:
            self.to(device)

    @property
    def stationary_dist(self) -> D.MultivariateNormal:
        return D.MultivariateNormal(self.stationary_dist_mean, self.stationary_dist_var)

    def _reparametrize(self, mu: torch.Tensor, logvar_like: torch.Tensor) -> torch.Tensor:
        # Repo uses sigmoid outputs but still treats them like logvar; preserved for behavior compatibility
        std = torch.exp(0.5 * logvar_like)
        eps = torch.randn_like(std)
        return mu + std * eps

    def kl_loss(self, mus: torch.Tensor, logvars: torch.Tensor, z_est: torch.Tensor) -> torch.Tensor:
        lags_and_length = z_est.shape[1]

        q_dist = D.Normal(mus, torch.exp(logvars / 2))
        log_qz = q_dist.log_prob(z_est)

        # Past KLD
        p_dist = D.Normal(torch.zeros_like(mus[:, : self.cfg.lags]), torch.ones_like(logvars[:, : self.cfg.lags]))
        log_pz_normal = torch.sum(torch.sum(p_dist.log_prob(z_est[:, : self.cfg.lags]), dim=-1), dim=-1)
        log_qz_normal = torch.sum(torch.sum(log_qz[:, : self.cfg.lags], dim=-1), dim=-1)
        kld_normal = (log_qz_normal - log_pz_normal).mean()

        # Future KLD through transition prior
        log_qz_laplace = log_qz[:, self.cfg.lags :]
        residuals, logabsdet = self.stationary_transition_prior(z_est)
        log_pz_laplace = torch.sum(self.stationary_dist.log_prob(residuals), dim=1) + logabsdet.sum(dim=1)

        kld_laplace = (
            torch.sum(torch.sum(log_qz_laplace, dim=-1), dim=-1) - log_pz_laplace
        ) / (lags_and_length - self.cfg.lags)
        kld_laplace = kld_laplace.mean()

        return kld_normal + kld_laplace

    def forward(self, x: torch.Tensor, is_training: bool = True, return_latents: bool = False):
        cfg = self.cfg
        x = x.float()

        if cfg.mode == "time":
            # Encoder in "time mode"
            zs_backbone = self.encoder.forward_time(x, mask="all_true")        # [B, enc_in, ts_out]
            zs_ = self.zs_rec(zs_backbone)                                     # [B, enc_in, 2*seq_len]

            zd_ = self.zd_rec(x).permute(0, 2, 1)                              # [B, enc_in, 2*seq_len]

            zs_rec_mean = zs_[:, :, : cfg.seq_len]
            zs_rec_std  = self.sigmoid(zs_[:, :, cfg.seq_len :])

            zd_rec_mean = zd_[:, :, : cfg.seq_len]
            zd_rec_std  = self.sigmoid(zd_[:, :, cfg.seq_len :])

            if is_training:
                zs_rec = self._reparametrize(zs_rec_mean, zs_rec_std)
                zd_rec = self._reparametrize(zd_rec_mean, zd_rec_std)
            else:
                zs_rec, zd_rec = zs_rec_mean, zd_rec_mean

            zs_pred = self.zs_pred(zs_rec)                                     # [B, enc_in, 2*pred_len]
            zd_pred = self.zd_pred(zd_rec)                                     # [B, enc_in, 2*pred_len]

            # Fixed bug from repo: std must come from zs_pred/zd_pred, not zs_/zd_
            zs_pred_mean = zs_pred[:, :, : cfg.pred_len]
            zs_pred_std  = self.sigmoid(zs_pred[:, :, cfg.pred_len :])

            zd_pred_mean = zd_pred[:, :, : cfg.pred_len]
            zd_pred_std  = self.sigmoid(zd_pred[:, :, cfg.pred_len :])

            if is_training:
                zs_pred_rec = self._reparametrize(zs_pred_mean, zs_pred_std)
                zd_pred_rec = self._reparametrize(zd_pred_mean, zd_pred_std)
            else:
                zs_pred_rec, zd_pred_rec = zs_pred_mean, zd_pred_mean

            # Reconstruction
            x_rec = self.x_rec(torch.cat([zs_rec, zd_rec], dim=-1)).transpose(1, 2)  # [B, seq_len, enc_in]

            # Forecast
            y = self.regressor(torch.cat([zs_pred_rec, zd_pred_rec], dim=-1).transpose(1, 2))  # [B, pred_len, enc_in]

            zs_q1 = zs_rec[:, :, : (cfg.seq_len // 2)].permute(0, 2, 1)       # [B, half, enc_in]
            zs_q2 = zs_rec[:, :, - (cfg.seq_len // 2) :].permute(0, 2, 1)
        else:
            # Standard feature mode
            zs_backbone = self.encoder.forward(x, mask="all_true")             # [B, seq_len, ts_out]
            zs_ = self.zs_rec(zs_backbone)                                     # [B, seq_len, 2*enc_in]
            zd_ = self.zd_rec(x)                                               # [B, seq_len, 2*enc_in]

            zs_rec_mean = zs_[:, :, : cfg.enc_in]
            zs_rec_std  = self.sigmoid(zs_[:, :, cfg.enc_in :])

            zd_rec_mean = zd_[:, :, : cfg.enc_in]
            zd_rec_std  = self.sigmoid(zd_[:, :, cfg.enc_in :])

            if is_training:
                zs_rec = self._reparametrize(zs_rec_mean, zs_rec_std)
                zd_rec = self._reparametrize(zd_rec_mean, zd_rec_std)
            else:
                zs_rec, zd_rec = zs_rec_mean, zd_rec_mean

            zs_pred = self.zs_pred(zs_rec.permute(0, 2, 1))                    # [B, enc_in, 2*pred_len]
            zd_pred = self.zd_pred(zd_rec.permute(0, 2, 1))                    # [B, enc_in, 2*pred_len]

            zs_pred_mean = zs_pred[:, :, : cfg.pred_len].permute(0, 2, 1)      # [B, pred_len, enc_in]
            zs_pred_std  = self.sigmoid(zs_pred[:, :, cfg.pred_len :].permute(0, 2, 1))

            zd_pred_mean = zd_pred[:, :, : cfg.pred_len].permute(0, 2, 1)
            zd_pred_std  = self.sigmoid(zd_pred[:, :, cfg.pred_len :].permute(0, 2, 1))

            if is_training:
                zs_pred_rec = self._reparametrize(zs_pred_mean, zs_pred_std)
                zd_pred_rec = self._reparametrize(zd_pred_mean, zd_pred_std)
            else:
                zs_pred_rec, zd_pred_rec = zs_pred_mean, zd_pred_mean

            x_rec = self.x_rec(torch.cat([zs_rec, zd_rec], dim=-1))            # [B, seq_len, enc_in]
            y = self.regressor(torch.cat([zs_pred_rec, zd_pred_rec], dim=-1))  # [B, pred_len, enc_in]

            zs_q1 = zs_rec[:, : (cfg.seq_len // 2), :]
            zs_q2 = zs_rec[:, - (cfg.seq_len // 2) :, :]

        # Attention consistency regularizer (repo L2 term)
        _, weights1 = self.attention(zs_q1, zs_q1, zs_q1)
        _, weights2 = self.attention(zs_q2, zs_q2, zs_q2)
        L2_loss = torch.mean((weights1 - weights2) ** 2)

        # L1 sparsity on zd_pred params (fixed: sum over all params instead of overwriting)
        L1_loss = torch.tensor(0.0, device=x.device)
        for p in self.zd_pred.parameters():
            L1_loss = L1_loss + torch.abs(p).sum()

        if is_training:
            if cfg.mode == "time":
                zs_kl_loss = self.kl_loss(
                    torch.cat([zs_rec_mean, zs_pred_mean], dim=2).permute(0, 2, 1),
                    torch.cat([zs_rec_std, zs_pred_std], dim=2).permute(0, 2, 1),
                    torch.cat([zs_rec, zs_pred_rec], dim=2).permute(0, 2, 1),
                )
                zd_kl_loss = self.kl_loss(
                    torch.cat([zd_rec_mean, zd_pred_mean], dim=2).permute(0, 2, 1),
                    torch.cat([zd_rec_std, zd_pred_std], dim=2).permute(0, 2, 1),
                    torch.cat([zd_rec, zd_pred_rec], dim=2).permute(0, 2, 1),
                )
            else:
                zs_kl_loss = self.kl_loss(
                    torch.cat([zs_rec_mean, zs_pred_mean], dim=1),
                    torch.cat([zs_rec_std, zs_pred_std], dim=1),
                    torch.cat([zs_rec, zs_pred_rec], dim=1),
                )
                zd_kl_loss = self.kl_loss(
                    torch.cat([zd_rec_mean, zd_pred_mean], dim=1),
                    torch.cat([zd_rec_std, zd_pred_std], dim=1),
                    torch.cat([zd_rec, zd_pred_rec], dim=1),
                )

            other_loss = (
                zs_kl_loss * cfg.zc_kl_weight
                + zd_kl_loss * cfg.zd_kl_weight
                + cfg.L1_weight * L1_loss
                + cfg.L2_weight * L2_loss
            )
        else:
            other_loss = cfg.L1_weight * L1_loss + cfg.L2_weight * L2_loss

        y_flat = y.reshape(y.shape[0], -1)

        if return_latents:
            extras: Dict[str, Any] = {
                "y": y,
                "zs_q1": zs_q1,
                "zs_q2": zs_q2,
                "L1_loss": L1_loss.detach(),
                "L2_loss": L2_loss.detach(),
            }
            return x_rec, y_flat, other_loss, extras

        return x_rec, y_flat, other_loss

    @torch.no_grad()
    def store_grad(self):
        # Call after backward() and before zero_grad()
        self.encoder.store_grad()
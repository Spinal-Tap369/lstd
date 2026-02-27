# lstd_core/components.py

from __future__ import annotations

import torch
from torch import nn


class TimeLinear(nn.Module):
    """
    Apply Linear(in_len -> out_len) independently to each feature channel.

    Input : [B, T, C]
    Output: [B, out_len, C]
    """

    def __init__(self, in_len: int, out_len: int):
        super().__init__()
        self.linear = nn.Linear(in_len, out_len)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_t = x.transpose(1, 2)      # [B, C, T]
        y_t = self.linear(x_t)       # [B, C, out_len]
        return y_t.transpose(1, 2)   # [B, out_len, C]


class FeatureLinear(nn.Module):
    """
    Apply Linear(in_dim -> out_dim) independently to each time step.

    Input : [B, T, C]
    Output: [B, T, out_dim]
    """

    def __init__(self, in_dim: int, out_dim: int):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class LongTermEncoder(nn.Module):
    """
    Paper Table 3 / Appendix C long-term encoder phi^s.

    Practical faithful interpretation:
      x [B, L, C]
        -> Conv1d over time with 640 hidden channels
        -> 1x1 projection back to C channels
        -> output [B, L, C]

    This is the closest clean implementation of the paper's
    "FSNet backbone for long latent variables" while staying
    consistent with Table 3's 640-hidden description.
    """

    def __init__(
        self,
        seq_len: int,
        enc_in: int,
        conv_hidden: int = 640,
        kernel_size: int = 3,
    ):
        super().__init__()
        _ = seq_len  # kept for API symmetry

        padding = kernel_size // 2
        self.conv = nn.Conv1d(
            in_channels=enc_in,
            out_channels=conv_hidden,
            kernel_size=kernel_size,
            padding=padding,
        )
        self.proj = nn.Conv1d(
            in_channels=conv_hidden,
            out_channels=enc_in,
            kernel_size=1,
        )
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = x.transpose(1, 2)        # [B, C, L]
        h = self.act(self.conv(h))   # [B, 640, L]
        h = self.act(self.proj(h))   # [B, C, L]
        return h.transpose(1, 2)     # [B, L, C]


class ShortTermEncoder(nn.Module):
    """
    Paper Table 3 short-term encoder phi^d.

      x [B, L, C]
        -> TimeLinear(L -> 512), LeakyReLU
        -> TimeLinear(512 -> L), LeakyReLU
        -> [B, L, C]
    """

    def __init__(self, seq_len: int, hidden_dim: int = 512):
        super().__init__()
        self.fc1 = TimeLinear(seq_len, hidden_dim)
        self.fc2 = TimeLinear(hidden_dim, seq_len)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.act(self.fc1(x))    # [B, 512, C]
        h = self.act(self.fc2(h))    # [B, L, C]
        return h


class VariationalBlock(nn.Module):
    """
    Hidden feature -> mean/logvar -> reparameterized sample
    """

    def __init__(self, enc_in: int):
        super().__init__()
        self.mu = FeatureLinear(enc_in, enc_in)
        self.logvar = FeatureLinear(enc_in, enc_in)

    def forward(self, h: torch.Tensor, is_training: bool = True):
        mu = self.mu(h)
        logvar = self.logvar(h).clamp(min=-10.0, max=10.0)

        if is_training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            z = mu + eps * std
        else:
            z = mu

        return z, mu, logvar


class LongPredictor(nn.Module):
    """
    Paper Table 3 T^s.

      z_s [B, L, C]
        -> TimeLinear(L -> 512), LeakyReLU
        -> TimeLinear(512 -> pred_len), LeakyReLU
        -> [B, pred_len, C]
    """

    def __init__(self, seq_len: int, pred_len: int, hidden_dim: int = 512):
        super().__init__()
        self.fc1 = TimeLinear(seq_len, hidden_dim)
        self.fc2 = TimeLinear(hidden_dim, pred_len)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        h = self.act(self.fc1(z))
        h = self.act(self.fc2(h))
        return h


class ShortPredictor(nn.Module):
    """
    Paper Table 3 T^d.

      z_d [B, L, C]
        -> TimeLinear(L -> pred_len), LeakyReLU
        -> [B, pred_len, C]
    """

    def __init__(self, seq_len: int, pred_len: int):
        super().__init__()
        self.fc = TimeLinear(seq_len, pred_len)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.act(self.fc(z))


class HistoricalDecoder(nn.Module):
    """
    Paper Table 3 F_x.

      concat(z_s_hist, z_d_hist) : [B, L, 2C]
      Dense(2C -> C), LeakyReLU  : [B, L, C]
    """

    def __init__(self, enc_in: int):
        super().__init__()
        self.fc = nn.Linear(2 * enc_in, enc_in)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, z_s: torch.Tensor, z_d: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z_s, z_d], dim=-1)
        return self.act(self.fc(x))


class FuturePredictor(nn.Module):
    """
    Paper Table 3 F_y.

      concat(z_s_future, z_d_future) : [B, pred_len, 2C]
      Dense(2C -> 512), LeakyReLU    : [B, pred_len, 512]
      Dense(512 -> C), LeakyReLU     : [B, pred_len, C]
    """

    def __init__(self, enc_in: int, hidden_dim: int = 512):
        super().__init__()
        self.fc1 = nn.Linear(2 * enc_in, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, enc_in)
        self.act = nn.LeakyReLU(0.2)

    def forward(self, z_s: torch.Tensor, z_d: torch.Tensor) -> torch.Tensor:
        x = torch.cat([z_s, z_d], dim=-1)
        x = self.act(self.fc1(x))
        x = self.act(self.fc2(x))
        return x


class MLP2(nn.Module):
    """
    Modular prior-network MLP used for each r_i.

    Table 3:
      input dim = n* + 1 (with lags=1, that is latent_size + 1)
      hidden    = 128
      hidden    = 128
      hidden    = 128
      output    = 1
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: int = 128,
        output_dim: int = 1,
        num_hidden_layers: int = 3,
    ):
        super().__init__()

        layers = []
        in_dim = input_dim
        for _ in range(num_hidden_layers):
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.LeakyReLU(0.2))
            in_dim = hidden_dim
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NPTransitionPrior(nn.Module):
    """
    Paper-style modular prior network.

    For each latent dimension i and time t, we model:
        eps_{t,i} = r_i([z_{t-1}, z_{t,i}])

    with first-order Markov by default (lags=1), so the input size
    is latent_size * lags + 1, which matches the paper's n* + 1.

    Returns:
        residuals            : [B, length, D]
        log_abs_det_jacobian : [B, length]
    """

    def __init__(
        self,
        lags: int,
        latent_size: int,
        num_hidden_layers: int = 3,
        hidden_dim: int = 128,
    ):
        super().__init__()
        self.lags = lags
        self.latent_size = latent_size

        self.gs = nn.ModuleList(
            [
                MLP2(
                    input_dim=lags * latent_size + 1,
                    hidden_dim=hidden_dim,
                    output_dim=1,
                    num_hidden_layers=num_hidden_layers,
                )
                for _ in range(latent_size)
            ]
        )

    def forward(self, x: torch.Tensor):
        # x: [B, lags + length, D]
        batch_size, lags_and_length, x_dim = x.shape
        if x_dim != self.latent_size:
            raise ValueError(f"Expected latent_size={self.latent_size}, got {x_dim}")

        length = lags_and_length - self.lags
        if length <= 0:
            raise ValueError("Input does not contain enough time steps for the configured lags")

        # [B, length, lags+1, D]
        windows = x.unfold(dimension=1, size=self.lags + 1, step=1)
        windows = windows.permute(0, 1, 3, 2).contiguous()
        windows = windows.view(batch_size * length, self.lags + 1, x_dim)

        # [B*length, lags*D]
        x_lags = windows[:, :-1].reshape(batch_size * length, self.lags * x_dim)
        # [B*length, D]
        x_t = windows[:, -1]

        residuals = []
        sum_log_abs_det_jacobian = torch.zeros(
            batch_size * length,
            device=x.device,
            dtype=x.dtype,
        )

        for i in range(self.latent_size):
            # Each r_i sees all lagged latent dims plus current scalar z_{t,i}
            inputs = torch.cat([x_lags, x_t[:, i:i + 1]], dim=-1)
            inputs = inputs.requires_grad_(True)

            residual = self.gs[i](inputs)  # [B*length, 1]

            deriv = torch.autograd.grad(
                residual.sum(),
                inputs,
                retain_graph=True,
                create_graph=True,
            )[0][:, -1]

            logabsdet = torch.log(torch.abs(deriv) + 1e-8)

            residuals.append(residual)
            sum_log_abs_det_jacobian = sum_log_abs_det_jacobian + logabsdet

        residuals = torch.cat(residuals, dim=-1).reshape(batch_size, length, x_dim)
        log_abs_det_jacobian = sum_log_abs_det_jacobian.reshape(batch_size, length)

        return residuals, log_abs_det_jacobian
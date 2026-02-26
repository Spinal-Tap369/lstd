# lstd_core/components.py

from __future__ import annotations

from typing import Optional

import torch
from torch import nn

try:
    from torch.func import vmap, jacfwd
except Exception:  # pragma: no cover
    from functorch import vmap, jacfwd


class MLP(nn.Module):
    def __init__(
        self,
        layer_nums: int,
        in_dim: int,
        hid_dim: Optional[int] = None,
        out_dim: Optional[int] = None,
        activation: str = "gelu",
        layer_norm: bool = True,
    ):
        super().__init__()
        if out_dim is None:
            out_dim = in_dim

        if activation == "gelu":
            act = nn.GELU()
        elif activation == "relu":
            act = nn.ReLU()
        elif activation == "tanh":
            act = nn.Tanh()
        else:
            act = nn.Identity()

        if layer_nums < 1:
            raise ValueError("layer_nums must be >= 1")

        if layer_nums == 1:
            self.net = nn.Sequential(nn.Linear(in_dim, out_dim))
            return

        if hid_dim is None:
            raise ValueError("hid_dim is required when layer_nums > 1")

        layers = [nn.Linear(in_dim, hid_dim), act]
        if layer_norm:
            layers.append(nn.LayerNorm(hid_dim))

        for _ in range(layer_nums - 2):  # fixed bug from repo
            layers.append(nn.Linear(hid_dim, hid_dim))
            layers.append(act)
            if layer_norm:
                layers.append(nn.LayerNorm(hid_dim))

        layers.append(nn.Linear(hid_dim, out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class BaseNet(nn.Module):
    """
    Cleaned version of Base_Net from exp_LSTD.py.
    """
    def __init__(
        self,
        input_len: int,
        out_len: int,
        hidden_dim: int,
        input_dim: int,
        out_dim: int,
        is_mean_std: bool = True,
        activation: str = "gelu",
        layer_norm: bool = True,
        c_type: str = "None",
        drop_out: float = 0.0,
        layer_nums: int = 2,
    ):
        super().__init__()

        self.input_dim = input_dim
        self.out_len = out_len
        self.out_dim = out_dim
        self.c_type = "type1" if (out_dim != input_dim and c_type == "None") else c_type
        self.radio = 2 if is_mean_std else 1

        if self.c_type == "None":
            self.net = MLP(
                layer_nums,
                in_dim=input_len,
                hid_dim=hidden_dim,
                out_dim=out_len * self.radio,
                activation=activation,
                layer_norm=layer_norm,
            )
        elif self.c_type == "type1":
            self.net = MLP(
                layer_nums,
                in_dim=self.input_dim,
                hid_dim=hidden_dim,
                out_dim=self.out_dim * self.radio,
                activation=activation,
                layer_norm=layer_norm,
            )
        elif self.c_type == "type2":
            self.net = MLP(
                layer_nums,
                in_dim=self.input_dim * input_len,
                hid_dim=hidden_dim,
                out_dim=self.out_dim * input_len * self.radio,
                activation=activation,
                layer_norm=layer_norm,
            )
        else:
            raise ValueError(f"Unknown c_type: {self.c_type}")

        self.dropout_net = nn.Dropout(drop_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if self.c_type == "type1":
            x = self.net(x)
        elif self.c_type == "type2":
            x = self.net(x.reshape(x.shape[0], -1)).reshape(x.shape[0], -1, self.out_dim * self.radio)
        else:
            x = self.net(x.permute(0, 2, 1)).permute(0, 2, 1)

        x = self.dropout_net(x)

        if self.radio == 2:
            split_dim = 2 if self.c_type in {"type1", "type2"} else 1
            return torch.chunk(x, chunks=2, dim=split_dim)

        return x


class MLP2(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, num_layers: int, leaky_relu_slope: float = 0.2):
        super().__init__()
        layers = []
        for i in range(num_layers):
            if i == 0:
                layers.extend([nn.Linear(input_dim, hidden_dim), nn.LeakyReLU(leaky_relu_slope)])
            else:
                layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.LeakyReLU(leaky_relu_slope)])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class NPTransitionPrior(nn.Module):
    """
    Same idea as repo: non-parametric transition prior with jacobian term.
    """
    def __init__(self, lags: int, latent_size: int, num_layers: int = 3, hidden_dim: int = 64, compress_dim: int = 10):
        super().__init__()
        self.lags = lags
        self.latent_size = latent_size
        self.compress_dim = compress_dim

        if latent_size > 100:
            self.gs = nn.ModuleList([
                MLP2(input_dim=compress_dim + 1, hidden_dim=hidden_dim, output_dim=1, num_layers=num_layers)
                for _ in range(latent_size)
            ])
            self.compress = nn.Linear(lags * latent_size, compress_dim)
        else:
            self.gs = nn.ModuleList([
                MLP2(input_dim=lags * latent_size + 1, hidden_dim=hidden_dim, output_dim=1, num_layers=num_layers)
                for _ in range(latent_size)
            ])
            self.compress = None

    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None):
        # x: [B, lags + length, D]
        batch_size, lags_and_length, x_dim = x.shape
        length = lags_and_length - self.lags

        batch_x = x.unfold(dimension=1, size=self.lags + 1, step=1).transpose(2, 3)  # [B, length, lags+1, D]
        batch_x = batch_x.reshape(-1, self.lags + 1, x_dim)

        batch_x_lags = batch_x[:, :-1].reshape(-1, self.lags * x_dim)  # [B*length, lags*D]
        batch_x_t = batch_x[:, -1]                                      # [B*length, D]

        if self.compress is not None:
            batch_x_lags = self.compress(batch_x_lags)

        sum_log_abs_det_jacobian = torch.zeros(
            batch_x_lags.shape[0], device=x.device, dtype=x.dtype
        )
        residuals = []

        for i in range(self.latent_size):
            if mask is not None:
                batch_inputs = torch.cat((batch_x_lags * mask[i], batch_x_t[:, i:i + 1]), dim=-1)
            else:
                batch_inputs = torch.cat((batch_x_lags, batch_x_t[:, i:i + 1]), dim=-1)

            residual = self.gs[i](batch_inputs)  # [B*length, 1]

            J = jacfwd(self.gs[i])
            data_J = vmap(J)(batch_inputs).squeeze()   # [B*length, input_dim]
            logabsdet = torch.log(torch.abs(data_J[:, -1]) + 1e-8)

            sum_log_abs_det_jacobian = sum_log_abs_det_jacobian + logabsdet
            residuals.append(residual)

        residuals = torch.cat(residuals, dim=-1).reshape(batch_size, length, x_dim)
        log_abs_det_jacobian = sum_log_abs_det_jacobian.reshape(batch_size, length)
        return residuals, log_abs_det_jacobian
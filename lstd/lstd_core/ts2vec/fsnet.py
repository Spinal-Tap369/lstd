# lstd_core/ts2vec/fsnet.py

from __future__ import annotations

import math
from typing import List, Optional

import torch
from torch import nn
import torch.nn.functional as F
import torch.fft as fft

from .masks import generate_binomial_mask, generate_continuous_mask
from .dilated_conv import DilatedConvEncoder as PlainDilatedConvEncoder
from .fsnet_blocks import AdaptiveDilatedConvEncoder


class TSEncoder(nn.Module):
    """
    TS2Vec encoder wrapper used by the LSTD model.
    - forward(): standard mode (B, T, C)
    - forward_time(): repo's 'time' mode (treats time axis as feature axis)
    """
    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        hidden_dims: int = 64,
        depth: int = 10,
        mask_mode: str = "binomial",
        use_adaptive_memory_conv: bool = True,
        gamma: float = 0.9,
        tau: float = 0.5,
    ):
        super().__init__()
        self.input_dims = input_dims
        self.output_dims = output_dims
        self.hidden_dims = hidden_dims
        self.mask_mode = mask_mode

        self.input_fc = nn.Linear(input_dims, hidden_dims)

        channels = [hidden_dims] * depth + [output_dims]
        if use_adaptive_memory_conv:
            self.feature_extractor = AdaptiveDilatedConvEncoder(
                hidden_dims, channels, kernel_size=3, gamma=gamma, tau=tau
            )
        else:
            self.feature_extractor = PlainDilatedConvEncoder(hidden_dims, channels, kernel_size=3)

        self.repr_dropout = nn.Dropout(p=0.1)

    def ctrl_params(self):
        yield from self.feature_extractor.ctrl_params()

    def store_grad(self):
        if hasattr(self.feature_extractor, "store_grad"):
            self.feature_extractor.store_grad()

    def _build_mask(self, x: torch.Tensor, mask: Optional[str]) -> torch.Tensor:
        # x shape: [B, T, C]
        if mask is None:
            mask = self.mask_mode if self.training else "all_true"

        if mask == "binomial":
            m = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == "continuous":
            m = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == "all_true":
            m = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        elif mask == "all_false":
            m = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
        elif mask == "mask_last":
            m = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            m[:, -1] = False
        else:
            raise ValueError(f"Unknown mask mode: {mask}")
        return m

    def forward(self, x: torch.Tensor, mask: Optional[str] = None) -> torch.Tensor:
        # x: [B, T, input_dims]
        nan_mask = ~x.isnan().any(dim=-1)
        x = x.clone()
        x[~nan_mask] = 0
        x = self.input_fc(x.float())  # [B, T, hidden]

        m = self._build_mask(x, mask)
        m = m & nan_mask
        x[~m] = 0

        x = x.transpose(1, 2)                   # [B, hidden, T]
        x = self.repr_dropout(self.feature_extractor(x))  # [B, out, T]
        x = x.transpose(1, 2)                   # [B, T, out]
        return x

    def forward_time(self, x: torch.Tensor, mask: Optional[str] = None) -> torch.Tensor:
        """
        Repo-compatible 'time' mode behavior:
        input x is [B, T, C], but we transpose to [B, C, T] and apply Linear over T.
        """
        x = x.transpose(1, 2)                   # [B, C, T]
        nan_mask = ~x.isnan().any(dim=-1)       # [B, C]
        x = x.clone()
        x[~nan_mask] = 0

        x = self.input_fc(x.float())            # Linear over last dim (T -> hidden): [B, C, hidden]

        # Build mask over the "token axis" (now C)
        if mask is None:
            mask = self.mask_mode if self.training else "all_true"

        if mask == "binomial":
            m = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == "continuous":
            m = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == "all_true":
            m = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        elif mask == "all_false":
            m = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
        elif mask == "mask_last":
            m = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
            m[:, -1] = False
        else:
            raise ValueError(f"Unknown mask mode: {mask}")

        m = m & nan_mask
        x[~m] = 0

        x = x.transpose(1, 2)                   # [B, hidden, C]
        x = self.repr_dropout(self.feature_extractor(x))  # [B, out, C]
        x = x.transpose(1, 2)                   # [B, C, out]
        return x


class TS2VecEncoderWrapper(nn.Module):
    def __init__(self, encoder: TSEncoder, mask: str = "all_true"):
        super().__init__()
        self.encoder = encoder
        self.mask = mask

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x, mask=self.mask)[:, -1]


class BandedFourierLayer(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        band: int,
        num_bands: int,
        freq_mixing: bool = False,
        bias: bool = True,
        length: int = 201,
    ):
        super().__init__()
        self.length = length
        self.total_freqs = (self.length // 2) + 1
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.freq_mixing = freq_mixing
        self.band = band
        self.num_bands = num_bands

        self.num_freqs = self.total_freqs // self.num_bands + (
            self.total_freqs % self.num_bands if self.band == self.num_bands - 1 else 0
        )
        self.start = self.band * (self.total_freqs // self.num_bands)
        self.end = self.start + self.num_freqs

        if self.freq_mixing:
            self.weight = nn.Parameter(torch.empty((self.num_freqs, in_channels, self.total_freqs, out_channels), dtype=torch.cfloat))
        else:
            self.weight = nn.Parameter(torch.empty((self.num_freqs, in_channels, out_channels), dtype=torch.cfloat))

        self.bias = nn.Parameter(torch.empty((self.num_freqs, out_channels), dtype=torch.cfloat)) if bias else None
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1.0 / math.sqrt(fan_in) if fan_in > 0 else 0.0
            nn.init.uniform_(self.bias, -bound, bound)

    def _forward(self, x_fft: torch.Tensor) -> torch.Tensor:
        if self.freq_mixing:
            out = torch.einsum("bai,tiao->bto", x_fft, self.weight)
        else:
            out = torch.einsum("bti,tio->bto", x_fft[:, self.start:self.end], self.weight)
        return out if self.bias is None else (out + self.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, D]
        b, t, _ = x.shape
        x_fft = fft.rfft(x, dim=1)
        out_fft = torch.zeros(b, t // 2 + 1, self.out_channels, device=x.device, dtype=torch.cfloat)
        out_fft[:, self.start:self.end] = self._forward(x_fft)
        return fft.irfft(out_fft, n=t, dim=1)


class GlobalLocalMultiscaleTSEncoder(nn.Module):
    """
    Included for completeness because it exists in the repo, although exp_LSTD.py
    mainly uses the plain TSEncoder path.
    """
    def __init__(
        self,
        input_dims: int,
        output_dims: int,
        kernels: List[int],
        num_bands: int,
        freq_mixing: bool,
        length: int,
        hidden_dims: int = 64,
        depth: int = 10,
        mask_mode: str = "binomial",
        use_adaptive_memory_conv: bool = True,
        gamma: float = 0.9,
        tau: float = 0.5,
    ):
        super().__init__()
        self.kernels = kernels
        self.num_bands = num_bands
        self.input_fc = nn.Linear(input_dims, hidden_dims)
        self.mask_mode = mask_mode

        channels = [hidden_dims] * depth + [output_dims]
        if use_adaptive_memory_conv:
            self.feature_extractor = AdaptiveDilatedConvEncoder(hidden_dims, channels, kernel_size=3, gamma=gamma, tau=tau)
        else:
            self.feature_extractor = PlainDilatedConvEncoder(hidden_dims, channels, kernel_size=3)

        self.convs = nn.ModuleList(
            [nn.Conv1d(output_dims, output_dims // 2, k, padding=k - 1) for k in kernels]
        )
        self.fouriers = nn.ModuleList(
            [BandedFourierLayer(output_dims, output_dims // 2, b, num_bands, freq_mixing=freq_mixing, length=length)
             for b in range(num_bands)]
        )

    def forward(self, x: torch.Tensor, tcn_output: bool = False, mask: str = "all_true") -> torch.Tensor:
        nan_mask = ~x.isnan().any(dim=-1)
        x = x.clone()
        x[~nan_mask] = 0
        x = self.input_fc(x.float())

        if mask == "all_true":
            m = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
        elif mask == "all_false":
            m = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
        elif mask == "binomial":
            m = generate_binomial_mask(x.size(0), x.size(1)).to(x.device)
        elif mask == "continuous":
            m = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
        else:
            raise ValueError(f"Unsupported mask: {mask}")

        m = m & nan_mask
        x[~m] = 0

        x = x.transpose(1, 2)            # [B, H, T]
        x = self.feature_extractor(x)    # [B, O, T]

        if tcn_output:
            return x.transpose(1, 2)

        local_multiscale: Optional[torch.Tensor] = None
        if len(self.kernels) > 0:
            local_parts = []
            for k, mod in zip(self.kernels, self.convs):
                out = mod(x)
                if k != 1:
                    out = out[..., :-(k - 1)]
                local_parts.append(out.transpose(1, 2))
            local_multiscale = torch.stack(local_parts, dim=0).mean(dim=0)

        x_time = x.transpose(1, 2)

        global_multiscale: Optional[torch.Tensor] = None
        if self.num_bands > 0:
            global_parts = [layer(x_time) for layer in self.fouriers]
            global_multiscale = global_parts[0]

        if local_multiscale is None and global_multiscale is None:
            raise RuntimeError("No local or global multiscale branch is enabled.")

        if local_multiscale is None:
            return global_multiscale  # type: ignore[return-value]

        if global_multiscale is None:
            return local_multiscale

        return torch.cat([local_multiscale, global_multiscale], dim=-1)
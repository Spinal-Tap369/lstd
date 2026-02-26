# lstd_core/ts2vec/dilated_conv.py

import torch
from torch import nn
import torch.nn.functional as F
from typing import Iterator


class SamePadConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int = 1, groups: int = 1):
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        self.padding = self.receptive_field // 2
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation,
            groups=groups,
        )
        self.remove = 1 if self.receptive_field % 2 == 0 else 0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.remove > 0:
            out = out[:, :, :-self.remove]
        return out


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, dilation: int, final: bool = False):
        super().__init__()
        self.conv1 = SamePadConv(in_channels, out_channels, kernel_size, dilation=dilation)
        self.conv2 = SamePadConv(out_channels, out_channels, kernel_size, dilation=dilation)
        self.projector = nn.Conv1d(in_channels, out_channels, 1) if (in_channels != out_channels or final) else None

    def ctrl_params(self) -> Iterator[nn.Parameter]:
        return iter(())

    def store_grad(self) -> None:
        return None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.projector is None else self.projector(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        return x + residual


class DilatedConvEncoder(nn.Module):
    def __init__(self, in_channels: int, channels: list[int], kernel_size: int):
        super().__init__()
        self.blocks: list[ConvBlock] = [
            ConvBlock(
                channels[i - 1] if i > 0 else in_channels,
                channels[i],
                kernel_size=kernel_size,
                dilation=2 ** i,
                final=(i == len(channels) - 1),
            )
            for i in range(len(channels))
        ]
        self.net = nn.Sequential(*self.blocks)

    def ctrl_params(self) -> Iterator[nn.Parameter]:
        for layer in self.blocks:
            yield from layer.ctrl_params()

    def store_grad(self) -> None:
        for layer in self.blocks:
            layer.store_grad()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)
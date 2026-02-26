# lstd_core/ts2vec/fsnet_blocks.py

from __future__ import annotations

from itertools import chain
from typing import Iterator

import torch
from torch import nn
import torch.nn.functional as F


def _normalize_memory(W: torch.Tensor) -> torch.Tensor:
    norm = torch.norm(W)
    norm = torch.relu(norm - 1.0) + 1.0
    return W / norm


class AdaptiveSamePadConv(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        groups: int = 1,
        gamma: float = 0.9,
        tau: float = 0.5,
        memory_slots: int = 32,
    ):
        super().__init__()
        self.receptive_field = (kernel_size - 1) * dilation + 1
        self.padding = self.receptive_field // 2
        self.dilation = dilation
        self.kernel_size = kernel_size
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.gamma = gamma
        self.fast_gamma = 0.3
        self.tau = tau
        self.memory_slots = memory_slots

        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=self.padding,
            dilation=dilation,
            groups=groups,
            bias=False,
        )
        self.bias = nn.Parameter(torch.zeros(out_channels), requires_grad=True)

        self.grad_dim = self.conv.weight.numel()
        self.n_chunks = in_channels
        self.chunk_in_d = self.grad_dim // self.n_chunks
        self.chunk_out_d = int(in_channels * kernel_size // self.n_chunks)
        self.group_out = out_channels // in_channels

        if out_channels % in_channels != 0:
            raise ValueError(
                f"out_channels ({out_channels}) must be divisible by in_channels ({in_channels}) "
                "for FSNet-style chunked calibration."
            )

        nh = 64
        self.controller = nn.Sequential(nn.Linear(self.chunk_in_d, nh), nn.SiLU())
        self.calib_w = nn.Linear(nh, self.chunk_out_d)
        self.calib_b = nn.Linear(nh, self.group_out)
        self.calib_f = nn.Linear(nh, self.group_out)

        self.q_dim = self.n_chunks * (self.chunk_out_d + 2 * self.group_out)

        # Explicit annotations for Pyright/Pylance
        self.W: nn.Parameter = nn.Parameter(torch.empty(self.q_dim, memory_slots), requires_grad=False)
        nn.init.xavier_uniform_(self.W.data)
        self.W.data = _normalize_memory(self.W.data)

        self.grads: torch.Tensor
        self.fast_grads: torch.Tensor
        self.q_ema: torch.Tensor
        self.register_buffer("grads", torch.zeros(self.grad_dim))
        self.register_buffer("fast_grads", torch.zeros(self.grad_dim))
        self.register_buffer("q_ema", torch.zeros(self.q_dim))

        self.trigger = False
        self.cos = nn.CosineSimilarity(dim=0, eps=1e-6)
        self.remove = 1 if self.receptive_field % 2 == 0 else 0

    def ctrl_params(self) -> Iterator[nn.Parameter]:
        yield from chain(
            self.controller.parameters(),
            self.calib_w.parameters(),
            self.calib_b.parameters(),
            self.calib_f.parameters(),
        )

    @torch.no_grad()
    def store_grad(self) -> None:
        if self.conv.weight.grad is None:
            return

        grad = self.conv.weight.grad.detach().clone()
        grad = F.normalize(grad, dim=0).reshape(-1)

        self.fast_grads.mul_(self.fast_gamma).add_(grad * (1.0 - self.fast_gamma))

        if not self.training:
            sim = self.cos(self.fast_grads, self.grads)
            if sim.item() < -self.tau:
                self.trigger = True

        self.grads.mul_(self.gamma).add_(grad * (1.0 - self.gamma))

    def _pack_calibration(self, w: torch.Tensor, b: torch.Tensor, f: torch.Tensor) -> torch.Tensor:
        return torch.cat([w.reshape(-1), b.reshape(-1), f.reshape(-1)], dim=0)

    def _unpack_calibration(self, q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        a = self.n_chunks * self.chunk_out_d
        bdim = self.n_chunks * self.group_out
        w = q[:a].view(self.n_chunks, self.chunk_out_d)
        b = q[a:a + bdim].view(self.n_chunks, self.group_out)
        f = q[a + bdim:a + bdim + bdim].view(self.n_chunks, self.group_out)
        return w, b, f

    def _memory_retrieve_and_update(self, q: torch.Tensor) -> torch.Tensor:
        att = torch.softmax(q @ self.W / 0.5, dim=0)
        k = min(2, att.numel())
        vals, idx = torch.topk(att, k=k)
        weights = vals / (vals.sum() + 1e-8)

        retrieved = self.W[:, idx] @ weights

        for slot, w_slot in zip(idx.tolist(), weights.tolist()):
            self.W.data[:, slot] = self.tau * self.W.data[:, slot] + (1.0 - self.tau) * (q * w_slot)

        self.W.data = _normalize_memory(self.W.data)
        q = self.tau * q + (1.0 - self.tau) * retrieved
        return q

    def _fw_chunks(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x = self.grads.view(self.n_chunks, -1)
        rep = self.controller(x)
        w = self.calib_w(rep)
        b = self.calib_b(rep)
        f = self.calib_f(rep)

        q = self._pack_calibration(w, b, f)

        if torch.count_nonzero(self.q_ema).item() == 0:
            self.q_ema.copy_(q.detach())
        else:
            self.q_ema.mul_(self.fast_gamma).add_(q.detach() * (1.0 - self.fast_gamma))
            q = self.q_ema

        if self.trigger:
            q = self._memory_retrieve_and_update(q)
            self.trigger = False

        w, b, f = self._unpack_calibration(q)

        w = w.unsqueeze(0)
        b = b.reshape(-1)
        f = f.reshape(-1).unsqueeze(0).unsqueeze(2)
        return w, b, f

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        w, b, f = self._fw_chunks()
        cw = self.conv.weight * w
        conv_out = F.conv1d(
            x,
            cw,
            padding=self.padding,
            dilation=self.dilation,
            bias=self.bias * b,
        )
        out = f * conv_out
        if self.remove > 0:
            out = out[:, :, :-self.remove]
        return out

    def representation(self, x: torch.Tensor) -> torch.Tensor:
        out = self.conv(x)
        if self.remove > 0:
            out = out[:, :, :-self.remove]
        return out


class AdaptiveConvBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int,
        final: bool = False,
        gamma: float = 0.9,
        tau: float = 0.5,
    ):
        super().__init__()
        self.conv1 = AdaptiveSamePadConv(in_channels, out_channels, kernel_size, dilation=dilation, gamma=gamma, tau=tau)
        self.conv2 = AdaptiveSamePadConv(out_channels, out_channels, kernel_size, dilation=dilation, gamma=gamma, tau=tau)
        self.projector = nn.Conv1d(in_channels, out_channels, 1) if (in_channels != out_channels or final) else None

    def ctrl_params(self) -> Iterator[nn.Parameter]:
        yield from self.conv1.ctrl_params()
        yield from self.conv2.ctrl_params()

    def store_grad(self) -> None:
        self.conv1.store_grad()
        self.conv2.store_grad()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x if self.projector is None else self.projector(x)
        x = F.gelu(x)
        x = self.conv1(x)
        x = F.gelu(x)
        x = self.conv2(x)
        return x + residual


class AdaptiveDilatedConvEncoder(nn.Module):
    def __init__(self, in_channels: int, channels: list[int], kernel_size: int, gamma: float = 0.9, tau: float = 0.5):
        super().__init__()
        self.blocks: list[AdaptiveConvBlock] = [
            AdaptiveConvBlock(
                channels[i - 1] if i > 0 else in_channels,
                channels[i],
                kernel_size=kernel_size,
                dilation=2 ** i,
                final=(i == len(channels) - 1),
                gamma=gamma,
                tau=tau,
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
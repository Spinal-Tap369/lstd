# lstd_core/ts2vec/losses.py

import torch
import torch.nn.functional as F


def instance_contrastive_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    B, T = z1.size(0), z1.size(1)
    if B == 1:
        return z1.new_tensor(0.0)

    z = torch.cat([z1, z2], dim=0)       # [2B, T, C]
    z = z.transpose(0, 1)                # [T, 2B, C]
    sim = torch.matmul(z, z.transpose(1, 2))  # [T, 2B, 2B]

    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]
    logits = logits + torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)

    i = torch.arange(B, device=z1.device)
    loss = (logits[:, i, B + i - 1].mean() + logits[:, B + i, i].mean()) / 2
    return loss


def temporal_contrastive_loss(z1: torch.Tensor, z2: torch.Tensor) -> torch.Tensor:
    B, T = z1.size(0), z1.size(1)
    if T == 1:
        return z1.new_tensor(0.0)

    z = torch.cat([z1, z2], dim=1)       # [B, 2T, C]
    sim = torch.matmul(z, z.transpose(1, 2))  # [B, 2T, 2T]

    logits = torch.tril(sim, diagonal=-1)[:, :, :-1]
    logits = logits + torch.triu(sim, diagonal=1)[:, :, 1:]
    logits = -F.log_softmax(logits, dim=-1)

    t = torch.arange(T, device=z1.device)
    loss = (logits[:, t, T + t - 1].mean() + logits[:, T + t, t].mean()) / 2
    return loss


def hierarchical_contrastive_loss(
    z1: torch.Tensor,
    z2: torch.Tensor,
    alpha: float = 0.5,
    temporal_unit: int = 0,
) -> torch.Tensor:
    loss = torch.tensor(0.0, device=z1.device)
    d = 0
    while z1.size(1) > 1:
        if alpha != 0:
            loss = loss + alpha * instance_contrastive_loss(z1, z2)
        if d >= temporal_unit and (1 - alpha) != 0:
            loss = loss + (1 - alpha) * temporal_contrastive_loss(z1, z2)

        d += 1
        z1 = F.max_pool1d(z1.transpose(1, 2), kernel_size=2).transpose(1, 2)
        z2 = F.max_pool1d(z2.transpose(1, 2), kernel_size=2).transpose(1, 2)

    if z1.size(1) == 1:
        if alpha != 0:
            loss = loss + alpha * instance_contrastive_loss(z1, z2)
        d += 1

    return loss / d
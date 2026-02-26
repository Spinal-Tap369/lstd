# lstd_core/ts2vec/masks.py

from typing import Union

import numpy as np
import torch


def generate_continuous_mask(B: int, T: int, n: Union[int, float] = 5, l: Union[int, float] = 0.1) -> torch.Tensor:
    res = torch.full((B, T), True, dtype=torch.bool)
    if isinstance(n, float):
        n = int(n * T)
    n = max(min(n, T // 2), 1)

    if isinstance(l, float):
        l = int(l * T)
    l = max(l, 1)

    for i in range(B):
        for _ in range(n):
            t = np.random.randint(0, max(1, T - l + 1))
            res[i, t:t + l] = False
    return res


def generate_binomial_mask(B: int, T: int, p: float = 0.5) -> torch.Tensor:
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T))).to(torch.bool)
# lstd_train/utils.py

import json
import os
import random
from datetime import datetime
from typing import Any, Dict

import numpy as np
import torch


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    # Safe defaults for reproducibility
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def choose_device(device_str: str = "auto") -> torch.device:
    d = (device_str or "auto").lower()

    if d == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if d == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("device='cuda' requested but CUDA is not available.")
        return torch.device("cuda")
    if d == "cpu":
        return torch.device("cpu")

    raise ValueError("device must be one of: 'auto', 'cpu', 'cuda'")


def timestamp_tag() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def save_json(path: str, payload: Dict[str, Any]) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=_json_fallback)


def _json_fallback(obj: Any):
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if torch.is_tensor(obj):
        return obj.detach().cpu().tolist()
    return str(obj)
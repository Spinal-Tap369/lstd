# lstd_train/early_stopping.py

import os
from typing import Optional

import torch


class EarlyStopping:
    """
    Simple early stopping on a scalar metric (lower is better).
    Saves best checkpoint to `checkpoint_path`.
    """

    def __init__(self, patience: int = 5, min_delta: float = 0.0, verbose: bool = True):
        self.patience = int(patience)
        self.min_delta = float(min_delta)
        self.verbose = bool(verbose)

        self.best_score: Optional[float] = None
        self.counter: int = 0
        self.early_stop: bool = False

    def step(self, score: float, model: torch.nn.Module, checkpoint_path: str) -> None:
        if self.best_score is None:
            self.best_score = float(score)
            self._save_checkpoint(model, checkpoint_path, score)
            return

        improved = float(score) < (self.best_score - self.min_delta)

        if improved:
            self.best_score = float(score)
            self.counter = 0
            self._save_checkpoint(model, checkpoint_path, score)
        else:
            self.counter += 1
            if self.verbose:
                print(f"EarlyStopping counter: {self.counter}/{self.patience} (best={self.best_score:.6f})")
            if self.counter >= self.patience:
                self.early_stop = True

    def _save_checkpoint(self, model: torch.nn.Module, path: str, score: float) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(model.state_dict(), path)
        if self.verbose:
            print(f"Saved best checkpoint (monitor={score:.6f}) -> {path}")
# lstd_train/metrics.py

from typing import Dict

import numpy as np


def regression_metrics(y_pred: np.ndarray, y_true: np.ndarray, eps: float = 1e-8) -> Dict[str, float]:
    """
    y_pred, y_true: same shape, e.g. [N, pred_len, C] or [N, D]
    """
    yp = np.asarray(y_pred, dtype=np.float64)
    yt = np.asarray(y_true, dtype=np.float64)

    if yp.shape != yt.shape:
        raise ValueError(f"Shape mismatch: y_pred={yp.shape}, y_true={yt.shape}")

    err = yp - yt
    abs_err = np.abs(err)

    mse = float(np.mean(err ** 2))
    mae = float(np.mean(abs_err))
    rmse = float(np.sqrt(mse))

    denom = np.where(np.abs(yt) < eps, eps, np.abs(yt))
    mape = float(np.mean(abs_err / denom))
    mspe = float(np.mean((err / denom) ** 2))

    return {
        "mae": mae,
        "mse": mse,
        "rmse": rmse,
        "mape": mape,
        "mspe": mspe,
    }
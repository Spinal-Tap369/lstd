# binance_ts/time_features.py

import numpy as np
import pandas as pd


def time_features(df_stamp: pd.DataFrame, timeenc: int = 0, freq: str = "15min") -> np.ndarray:
    """
    Lightweight replacement for the repo's utils.timefeatures.time_features.

    Returns arrays shaped [N, F] suitable for seq_x_mark / seq_y_mark.
    """
    if "date" not in df_stamp.columns:
        raise ValueError("df_stamp must contain a 'date' column")

    dt = pd.to_datetime(df_stamp["date"])

    # Convert pandas datetime components to plain NumPy arrays for type-checker friendliness
    month = np.asarray(dt.dt.month.to_numpy(), dtype=np.int64)
    day = np.asarray(dt.dt.day.to_numpy(), dtype=np.int64)
    weekday = np.asarray(dt.dt.weekday.to_numpy(), dtype=np.int64)
    hour = np.asarray(dt.dt.hour.to_numpy(), dtype=np.int64)
    minute = np.asarray(dt.dt.minute.to_numpy(), dtype=np.int64)

    if timeenc == 0:
        # Informer-style discrete-ish normalized ints
        # minute bucket for 15m data
        minute_bucket = (minute // 15).astype(np.int64)
        feats = np.stack([month, day, weekday, hour, minute_bucket], axis=1).astype(np.float32)
        return feats

    # Continuous normalized version (similar spirit to repo's timeenc=2 path)
    feats = np.stack(
        [
            (month - 1).astype(np.float32) / 11.0,
            (day - 1).astype(np.float32) / 30.0,
            weekday.astype(np.float32) / 6.0,
            hour.astype(np.float32) / 23.0,
            minute.astype(np.float32) / 59.0,
        ],
        axis=1,
    ).astype(np.float32)

    return feats
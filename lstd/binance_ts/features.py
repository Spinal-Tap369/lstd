# binance_ts/features.py

from typing import List, Tuple, Optional

import numpy as np
import pandas as pd

from .config import FeatureConfig, LSTDExportConfig


def _safe_div(a: pd.Series, b: pd.Series, eps: float = 1e-12) -> pd.Series:
    return a / (b.replace(0, np.nan) + eps)


def add_basic_price_features(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()

    out["hl2"] = (out["high"] + out["low"]) / 2.0
    out["ohlc4"] = (out["open"] + out["high"] + out["low"] + out["close"]) / 4.0

    out["ret_1"] = out["close"].pct_change(1)
    out["ret_4"] = out["close"].pct_change(4)
    out["ret_16"] = out["close"].pct_change(16)
    out["log_ret_1"] = np.log(out["close"]).diff(1)

    out["range_abs"] = out["high"] - out["low"]
    out["range_pct"] = _safe_div(out["high"] - out["low"], out["close"])
    out["body_abs"] = (out["close"] - out["open"]).abs()
    out["body_pct"] = _safe_div((out["close"] - out["open"]).abs(), out["close"])
    out["upper_wick_abs"] = out["high"] - out[["open", "close"]].max(axis=1)
    out["lower_wick_abs"] = out[["open", "close"]].min(axis=1) - out["low"]

    out["volume_log1p"] = np.log1p(out["volume"])
    out["quote_volume_log1p"] = np.log1p(out["quote_asset_volume"])
    out["taker_buy_ratio"] = _safe_div(out["taker_buy_base_asset_volume"], out["volume"])

    return out


def add_instance_norm_features(df: pd.DataFrame, window: int = 96) -> pd.DataFrame:
    if window < 2:
        raise ValueError("instance_norm window must be >= 2")

    out = df.copy()
    cols = ["open", "high", "low", "close", "volume"]

    for col in cols:
        mu = out[col].rolling(window=window, min_periods=window).mean()
        sigma = out[col].rolling(window=window, min_periods=window).std(ddof=0)
        out[f"{col}_zin_{window}"] = (out[col] - mu) / (sigma + 1e-8)

    return out


def add_seasonal_trend_features(df: pd.DataFrame, window: int = 96) -> pd.DataFrame:
    if window < 2:
        raise ValueError("seasonal_trend window must be >= 2")

    out = df.copy()
    for col in ["close", "volume"]:
        trend = out[col].rolling(window=window, min_periods=window).mean()
        seasonal = out[col] - trend
        out[f"{col}_trend_{window}"] = trend
        out[f"{col}_seasonal_{window}"] = seasonal
        out[f"{col}_seasonal_abs_{window}"] = seasonal.abs()

    out[f"close_trend_slope_{window}"] = out[f"close_trend_{window}"].diff()
    return out


def _rolling_fft_features(arr: np.ndarray, window: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = len(arr)
    dom_bin_norm = np.full(n, np.nan, dtype=float)
    low_ratio = np.full(n, np.nan, dtype=float)
    high_ratio = np.full(n, np.nan, dtype=float)

    if window < 8:
        return dom_bin_norm, low_ratio, high_ratio

    for i in range(window - 1, n):
        w = arr[i - window + 1:i + 1]
        if np.any(np.isnan(w)):
            continue

        w = w - np.mean(w)
        spec = np.fft.rfft(w)
        power = (np.abs(spec) ** 2)

        if len(power) <= 2:
            continue

        power_no_dc = power[1:]
        total = power_no_dc.sum()
        if total <= 1e-12:
            continue

        dom_idx = int(np.argmax(power_no_dc)) + 1
        dom_bin_norm[i] = dom_idx / max(1, len(power) - 1)

        split = max(1, len(power_no_dc) // 4)
        low = power_no_dc[:split].sum()
        high = power_no_dc[split:].sum() if split < len(power_no_dc) else 0.0

        low_ratio[i] = low / total
        high_ratio[i] = high / total

    return dom_bin_norm, low_ratio, high_ratio


def add_frequency_features(df: pd.DataFrame, window: int = 128) -> pd.DataFrame:
    if window < 8:
        raise ValueError("frequency window must be >= 8")

    out = df.copy()
    close_values = out["close"].astype(float).to_numpy()

    dom, low, high = _rolling_fft_features(close_values, window=window)
    out[f"close_fft_dom_bin_{window}"] = dom
    out[f"close_fft_low_ratio_{window}"] = low
    out[f"close_fft_high_ratio_{window}"] = high

    return out


def add_long_short_regime_features(df: pd.DataFrame, short_window: int = 16, long_window: int = 96) -> pd.DataFrame:
    if short_window < 2 or long_window < 2:
        raise ValueError("windows must be >= 2")
    if short_window >= long_window:
        raise ValueError("short_window should be smaller than long_window")

    out = df.copy()

    if "ret_1" not in out.columns:
        out["ret_1"] = out["close"].pct_change(1)

    out[f"ema_close_short_{short_window}"] = out["close"].ewm(span=short_window, adjust=False).mean()
    out[f"ema_close_long_{long_window}"] = out["close"].ewm(span=long_window, adjust=False).mean()
    out["ema_gap_short_long"] = out[f"ema_close_short_{short_window}"] - out[f"ema_close_long_{long_window}"]

    out[f"ret_mean_short_{short_window}"] = out["ret_1"].rolling(short_window, min_periods=short_window).mean()
    out[f"ret_mean_long_{long_window}"] = out["ret_1"].rolling(long_window, min_periods=long_window).mean()

    out[f"ret_vol_short_{short_window}"] = out["ret_1"].rolling(short_window, min_periods=short_window).std(ddof=0)
    out[f"ret_vol_long_{long_window}"] = out["ret_1"].rolling(long_window, min_periods=long_window).std(ddof=0)

    out["drift_proxy_mean_gap"] = (
        out[f"ret_mean_short_{short_window}"] - out[f"ret_mean_long_{long_window}"]
    ).abs()

    out["drift_proxy_vol_ratio"] = _safe_div(
        out[f"ret_vol_short_{short_window}"],
        out[f"ret_vol_long_{long_window}"],
    )

    return out


def add_targets(df: pd.DataFrame, horizon_bars: int = 1) -> pd.DataFrame:
    if horizon_bars < 1:
        raise ValueError("target_horizon_bars must be >= 1")

    out = df.copy()
    out[f"target_close_t_plus_{horizon_bars}"] = out["close"].shift(-horizon_bars)
    out[f"target_ret_t_plus_{horizon_bars}"] = (out["close"].shift(-horizon_bars) / out["close"]) - 1.0
    return out


def apply_feature_pipeline(df: pd.DataFrame, cfg: FeatureConfig) -> pd.DataFrame:
    out = df.copy()

    if cfg.add_basic_price_features:
        out = add_basic_price_features(out)

    if cfg.add_instance_norm_features:
        out = add_instance_norm_features(out, window=cfg.instance_norm_window)

    if cfg.add_seasonal_trend_features:
        out = add_seasonal_trend_features(out, window=cfg.seasonal_trend_window)

    if cfg.add_frequency_features:
        out = add_frequency_features(out, window=cfg.frequency_window)

    if cfg.add_long_short_regime_features:
        out = add_long_short_regime_features(out, short_window=cfg.short_window, long_window=cfg.long_window)

    if cfg.add_explicit_targets:
        out = add_targets(out, horizon_bars=cfg.target_horizon_bars)

    if cfg.drop_na_rows:
        out = out.dropna().reset_index(drop=True)

    return out


def build_lstd_ready_frame(df: pd.DataFrame, export_cfg: LSTDExportConfig) -> tuple[pd.DataFrame, List[str]]:
    """
    Build a CSV matching LSTD repo Dataset_Custom expectations:
      columns = ['date', ...(features), target]
    """
    out = df.copy()

    if "open_dt" not in out.columns:
        raise ValueError("Expected 'open_dt' column in engineered dataframe.")

    # Standard LSTD-style date column
    out["date"] = pd.to_datetime(out["open_dt"], utc=True).dt.strftime("%Y-%m-%d %H:%M:%S")

    # Candidate numeric columns
    numeric_cols = list(out.select_dtypes(include=["number"]).columns)

    exclude = set(export_cfg.exclude_columns)
    exclude.add("open_dt")
    exclude.add("close_dt")

    if export_cfg.drop_explicit_target_columns:
        for c in list(numeric_cols):
            if c.startswith("target_"):
                exclude.add(c)

    if export_cfg.target_column not in out.columns:
        raise ValueError(f"target_column '{export_cfg.target_column}' not found in dataframe")

    if export_cfg.feature_columns is None:
        feature_cols = [c for c in numeric_cols if c not in exclude]
    else:
        missing = [c for c in export_cfg.feature_columns if c not in out.columns]
        if missing:
            raise ValueError(f"feature_columns missing from dataframe: {missing}")
        feature_cols = [c for c in export_cfg.feature_columns if c not in exclude]

    # Ensure target is handled last (repo convention)
    if export_cfg.target_column in feature_cols:
        feature_cols.remove(export_cfg.target_column)

    mode = export_cfg.feature_mode.upper()
    if mode == "S":
        final_cols = ["date", export_cfg.target_column]
        used_feature_cols = [export_cfg.target_column]
    elif mode in {"M", "MS"}:
        final_cols = ["date"] + feature_cols + [export_cfg.target_column]
        used_feature_cols = feature_cols + [export_cfg.target_column]
    else:
        raise ValueError("feature_mode must be one of: S, M, MS")

    lstd_df = out[final_cols].copy()
    return lstd_df, used_feature_cols
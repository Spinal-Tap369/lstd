# binance_ts/run_example.py

from __future__ import annotations

import argparse
from copy import deepcopy
from pprint import pprint
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from .config import PipelineConfig, DEFAULT_CONFIG
from .pipeline import download_and_prepare
from .utils import align_ms_to_interval, dt_to_ms


def _to_utc_ts(x: str) -> pd.Timestamp:
    ts = pd.to_datetime(x, utc=True)
    if pd.isna(ts):
        raise ValueError(f"Could not parse timestamp: {x}")
    return ts


def validate_download_result(cfg: PipelineConfig, result: dict) -> None:
    raw_path = Path(result["raw_path"])
    features_path = Path(result["features_path"])

    # 1) File existence
    if not raw_path.exists():
        raise FileNotFoundError(f"Raw file not found: {raw_path}")
    if not features_path.exists():
        raise FileNotFoundError(f"Features file not found: {features_path}")

    # 2) Filename sanity (symbol + interval)
    expected_stub = f"{cfg.download.symbol}_{cfg.download.interval}"
    if expected_stub not in raw_path.name:
        raise ValueError(f"Raw filename does not match requested symbol/interval: {raw_path.name}")
    if expected_stub not in features_path.name:
        raise ValueError(f"Features filename does not match requested symbol/interval: {features_path.name}")

    # 3) Load files
    raw_df = pd.read_csv(raw_path)
    feat_df = pd.read_csv(features_path)

    # 4) Row count checks against pipeline return values
    if len(raw_df) != int(result["raw_rows"]):
        raise ValueError(f"raw_rows mismatch: result={result['raw_rows']} actual={len(raw_df)}")
    if len(feat_df) != int(result["feature_rows"]):
        raise ValueError(f"feature_rows mismatch: result={result['feature_rows']} actual={len(feat_df)}")

    # 5) Required raw columns
    raw_required = {
        "open_time", "open", "high", "low", "close", "volume",
        "close_time", "open_dt", "close_dt",
    }
    missing_raw = raw_required - set(raw_df.columns)
    if missing_raw:
        raise ValueError(f"Raw CSV missing required columns: {sorted(missing_raw)}")

    # 6) Parse and validate timestamps
    raw_open_ts = pd.to_datetime(raw_df["open_dt"], utc=True, errors="coerce")
    raw_close_ts = pd.to_datetime(raw_df["close_dt"], utc=True, errors="coerce")

    if raw_open_ts.isna().any():
        bad = raw_df.loc[raw_open_ts.isna(), "open_dt"].head(3).tolist()
        raise ValueError(f"Invalid open_dt timestamps found (examples): {bad}")
    if raw_close_ts.isna().any():
        bad = raw_df.loc[raw_close_ts.isna(), "close_dt"].head(3).tolist()
        raise ValueError(f"Invalid close_dt timestamps found (examples): {bad}")

    if not raw_open_ts.is_monotonic_increasing:
        raise ValueError("Raw timestamps are not sorted ascending (open_dt).")

    dup_count = int(raw_open_ts.duplicated().sum())
    if dup_count > 0:
        raise ValueError(f"Duplicate open_dt rows found: {dup_count}")

    # 7) Interval spacing check
    expected_delta = pd.to_timedelta(cfg.download.interval)
    deltas = raw_open_ts.diff().dropna()
    bad_spacing = deltas[deltas != expected_delta]
    if len(bad_spacing) > 0:
        ex_idx = bad_spacing.index[:5].tolist()
        ex_vals = [str(v) for v in bad_spacing.iloc[:5].tolist()]
        raise ValueError(
            f"Found {len(bad_spacing)} rows with wrong interval spacing "
            f"(expected {expected_delta}). Example indices={ex_idx}, deltas={ex_vals}"
        )

    # 8) Requested time coverage checks
    if cfg.download.start is not None:
        requested_start = _to_utc_ts(cfg.download.start)
        actual_start = raw_open_ts.iloc[0]
        if actual_start > (requested_start + expected_delta):
            raise ValueError(
                f"Raw data starts too late. requested_start={requested_start}, actual_start={actual_start}"
            )

    if cfg.download.end is not None:
        requested_end = _to_utc_ts(cfg.download.end)
        aligned_end_ms = align_ms_to_interval(
            dt_to_ms(requested_end.to_pydatetime()),
            cfg.download.interval,
            "floor",
        )
        aligned_end = pd.to_datetime(aligned_end_ms, unit="ms", utc=True)

        # pipeline keeps open_time < end_ms, so last expected open is aligned_end - interval
        expected_last_open = aligned_end - expected_delta
        actual_end = raw_open_ts.iloc[-1]

        if actual_end < expected_last_open:
            raise ValueError(
                f"Raw data ends too early. "
                f"requested_end={requested_end}, aligned_end={aligned_end}, "
                f"expected_last_open>={expected_last_open}, actual_last_open={actual_end}"
            )

    # 9) Numeric sanity on raw OHLCV
    ohlcv_cols = ["open", "high", "low", "close", "volume"]
    for col in ohlcv_cols:
        raw_df[col] = pd.to_numeric(raw_df[col], errors="coerce")

    if raw_df[ohlcv_cols].isna().any().any():
        bad_cols = raw_df[ohlcv_cols].isna().sum()
        raise ValueError(f"NaN values in raw OHLCV columns:\n{bad_cols}")

    raw_vals = raw_df[ohlcv_cols].to_numpy(dtype=np.float64)
    if not np.isfinite(raw_vals).all():
        raise ValueError("Raw OHLCV contains inf/-inf values.")

    if (raw_df["high"] < raw_df["low"]).any():
        raise ValueError("Found rows where high < low.")

    if (raw_df["high"] < raw_df[["open", "close"]].max(axis=1)).any():
        raise ValueError("Found rows where high < max(open, close).")

    if (raw_df["low"] > raw_df[["open", "close"]].min(axis=1)).any():
        raise ValueError("Found rows where low > min(open, close).")

    if (raw_df["close"] <= 0).any() or (raw_df["open"] <= 0).any():
        raise ValueError("Found non-positive prices in raw data.")

    if (raw_df["volume"] < 0).any():
        raise ValueError("Found negative volume values in raw data.")

    # 10) Feature-file sanity
    if "open_dt" not in feat_df.columns:
        raise ValueError("Feature CSV missing 'open_dt' column.")

    feat_open_ts = pd.to_datetime(feat_df["open_dt"], utc=True, errors="coerce")
    if feat_open_ts.isna().any():
        raise ValueError("Feature CSV has invalid 'open_dt' values.")

    if not feat_open_ts.is_monotonic_increasing:
        raise ValueError("Feature timestamps are not sorted ascending.")

    if int(feat_open_ts.duplicated().sum()) > 0:
        raise ValueError("Feature CSV contains duplicate timestamps.")

    if len(feat_df) > len(raw_df):
        raise ValueError("Feature CSV has more rows than raw CSV (unexpected).")

    returned_cols = list(result.get("feature_columns", []))
    if returned_cols:
        actual_cols = feat_df.columns.tolist()
        if returned_cols != actual_cols:
            raise ValueError("feature_columns returned by pipeline do not match actual CSV columns.")

    feat_num = feat_df.select_dtypes(include=[np.number])
    if feat_num.empty:
        raise ValueError("No numeric columns found in feature CSV.")

    if feat_num.isna().any().any():
        bad = feat_num.isna().sum()
        bad = bad[bad > 0].sort_values(ascending=False)
        raise ValueError(f"NaN values found in feature numeric columns:\n{bad}")

    feat_vals = feat_num.to_numpy(dtype=np.float64)
    if not np.isfinite(feat_vals).all():
        raise ValueError("Feature CSV contains inf/-inf values.")

    # 11) Target column check
    if cfg.export.target_column not in feat_df.columns:
        raise ValueError(
            f"Target column '{cfg.export.target_column}' not found in feature CSV. "
            f"Available columns include: {feat_df.columns[:20].tolist()}..."
        )

    print("\nValidation checks passed ✓")
    print(f"  Raw time range     : {raw_open_ts.iloc[0]} -> {raw_open_ts.iloc[-1]}")
    print(f"  Feature time range : {feat_open_ts.iloc[0]} -> {feat_open_ts.iloc[-1]}")
    print(f"  Raw rows           : {len(raw_df)}")
    print(f"  Feature rows       : {len(feat_df)}")
    print(f"  Target column      : {cfg.export.target_column}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download Binance klines and build LSTD-ready training data."
    )

    # Download params
    parser.add_argument("--symbol", type=str, default=None)
    parser.add_argument("--interval", type=str, default=None)
    parser.add_argument("--start", type=str, default=None)
    parser.add_argument("--end", type=str, default=None)
    parser.add_argument("--lookback-days", type=int, default=None)
    parser.add_argument("--output-dir", type=str, default=None)
    parser.add_argument("--request-limit", type=int, default=None)
    parser.add_argument("--sleep-seconds", type=float, default=None)
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--max-retries", type=int, default=None)
    parser.add_argument("--base-url", type=str, default=None)

    # Quality checks
    parser.add_argument("--allow-missing-candles", action="store_true")
    parser.add_argument("--disable-contiguity-check", action="store_true")

    # Feature toggles
    parser.add_argument("--add-instance-norm-features", action="store_true")
    parser.add_argument("--disable-seasonal-trend-features", action="store_true")
    parser.add_argument("--disable-frequency-features", action="store_true")
    parser.add_argument("--disable-long-short-regime-features", action="store_true")

    parser.add_argument("--instance-norm-window", type=int, default=None)
    parser.add_argument("--seasonal-trend-window", type=int, default=None)
    parser.add_argument("--frequency-window", type=int, default=None)
    parser.add_argument("--short-window", type=int, default=None)
    parser.add_argument("--long-window", type=int, default=None)

    parser.add_argument("--add-explicit-targets", action="store_true")
    parser.add_argument("--target-horizon-bars", type=int, default=None)
    parser.add_argument("--keep-na-rows", action="store_true")

    # Export params
    parser.add_argument("--feature-mode", type=str, choices=["S", "M", "MS"], default=None)
    parser.add_argument("--target-column", type=str, default=None)
    parser.add_argument("--disable-lstd-export", action="store_true")

    # Control
    parser.add_argument("--skip-validate", action="store_true")

    return parser


def apply_args_to_config(cfg: PipelineConfig, args: argparse.Namespace) -> PipelineConfig:
    # Download
    if args.symbol is not None:
        cfg.download.symbol = args.symbol
    if args.interval is not None:
        cfg.download.interval = args.interval
    if args.start is not None:
        cfg.download.start = args.start
    if args.end is not None:
        cfg.download.end = args.end
    if args.lookback_days is not None:
        cfg.download.lookback_days = args.lookback_days
    if args.output_dir is not None:
        cfg.download.output_dir = args.output_dir
    if args.request_limit is not None:
        cfg.download.request_limit = args.request_limit
    if args.sleep_seconds is not None:
        cfg.download.sleep_seconds = args.sleep_seconds
    if args.timeout is not None:
        cfg.download.timeout = args.timeout
    if args.max_retries is not None:
        cfg.download.max_retries = args.max_retries
    if args.base_url is not None:
        cfg.download.base_url = args.base_url

    if args.allow_missing_candles:
        cfg.download.allow_missing_candles = True
    if args.disable_contiguity_check:
        cfg.download.validate_contiguity = False

    # Features
    if args.add_instance_norm_features:
        cfg.features.add_instance_norm_features = True
    if args.disable_seasonal_trend_features:
        cfg.features.add_seasonal_trend_features = False
    if args.disable_frequency_features:
        cfg.features.add_frequency_features = False
    if args.disable_long_short_regime_features:
        cfg.features.add_long_short_regime_features = False

    if args.instance_norm_window is not None:
        cfg.features.instance_norm_window = args.instance_norm_window
    if args.seasonal_trend_window is not None:
        cfg.features.seasonal_trend_window = args.seasonal_trend_window
    if args.frequency_window is not None:
        cfg.features.frequency_window = args.frequency_window
    if args.short_window is not None:
        cfg.features.short_window = args.short_window
    if args.long_window is not None:
        cfg.features.long_window = args.long_window

    if args.add_explicit_targets:
        cfg.features.add_explicit_targets = True
    if args.target_horizon_bars is not None:
        cfg.features.target_horizon_bars = args.target_horizon_bars
    if args.keep_na_rows:
        cfg.features.drop_na_rows = False

    # Export
    if args.feature_mode is not None:
        cfg.export.feature_mode = args.feature_mode
    if args.target_column is not None:
        cfg.export.target_column = args.target_column
    if args.disable_lstd_export:
        cfg.export.enabled = False

    return cfg


def run_with_config(cfg: PipelineConfig, validate: bool = True) -> dict:
    result = download_and_prepare(cfg)

    print("\nSaved files:")
    print("  raw      :", result["raw_path"])
    print("  features :", result["features_path"])
    if "lstd_csv_path" in result:
        print("  lstd csv  :", result["lstd_csv_path"])
    if "metadata_path" in result:
        print("  metadata  :", result["metadata_path"])

    print("\nShapes:")
    print("  raw rows      :", result["raw_rows"])
    print("  feature rows  :", result["feature_rows"])
    if "lstd_rows" in result:
        print("  lstd rows     :", result["lstd_rows"])

    print("\nFeature columns (first 30):")
    pprint(result["feature_columns"][:30])

    if validate:
        validate_download_result(cfg, result)

    print("\nHead of engineered data:")
    df = pd.read_csv(result["features_path"])
    print(df.head(5).to_string())

    return result


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = deepcopy(DEFAULT_CONFIG)
    cfg = apply_args_to_config(cfg, args)

    run_with_config(cfg, validate=not args.skip_validate)


if __name__ == "__main__":
    main()
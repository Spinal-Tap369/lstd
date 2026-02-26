# binance_ts/pipeline.py

import json
import os
from typing import Dict, Any

from .binance_client import BinanceKlinesClient
from .config import PipelineConfig
from .features import apply_feature_pipeline, build_lstd_ready_frame
from .utils import (
    ensure_dir,
    resolve_time_range_ms,
    compact_time_str,
    validate_kline_contiguity,
)


def download_and_prepare(cfg: PipelineConfig) -> Dict[str, Any]:
    """
    Orchestrates:
    1) download raw klines
    2) save raw csv
    3) apply feature engineering
    4) save engineered csv
    5) (optional) save LSTD-ready csv: date + features + target
    """
    dcfg = cfg.download
    fcfg = cfg.features
    ecfg = cfg.export

    ensure_dir(dcfg.output_dir)

    start_ms, end_ms = resolve_time_range_ms(
        start=dcfg.start,
        end=dcfg.end,
        lookback_days=dcfg.lookback_days,
        interval=dcfg.interval,
    )

    client = BinanceKlinesClient(
        base_url=dcfg.base_url,
        timeout=dcfg.timeout,
        max_retries=dcfg.max_retries,
    )

    raw_df = client.fetch_historical_klines(
        symbol=dcfg.symbol,
        interval=dcfg.interval,
        start_ms=start_ms,
        end_ms=end_ms,
        request_limit=dcfg.request_limit,
        sleep_seconds=dcfg.sleep_seconds,
    )

    if raw_df.empty:
        raise RuntimeError("No kline data returned. Check symbol/interval/date range.")

    quality = validate_kline_contiguity(raw_df, dcfg.interval)
    if dcfg.validate_contiguity and (not quality["ok"]) and (not dcfg.allow_missing_candles):
        raise RuntimeError(f"Kline continuity check failed: {quality}")

    stem = f"{dcfg.symbol}_{dcfg.interval}_{compact_time_str(start_ms)}_{compact_time_str(end_ms)}"
    raw_path = os.path.join(dcfg.output_dir, f"{stem}_raw.csv")
    feat_path = os.path.join(dcfg.output_dir, f"{stem}_features.csv")

    raw_df.to_csv(raw_path, index=False)

    feat_df = apply_feature_pipeline(raw_df, fcfg)
    feat_df.to_csv(feat_path, index=False)

    result = {
        "raw_path": raw_path,
        "features_path": feat_path,
        "raw_rows": len(raw_df),
        "feature_rows": len(feat_df),
        "feature_columns": list(feat_df.columns),
        "quality": quality,
    }

    if ecfg.enabled:
        lstd_df, used_feature_cols = build_lstd_ready_frame(feat_df, ecfg)
        lstd_path = os.path.join(dcfg.output_dir, f"{stem}{ecfg.lstd_csv_suffix}")
        meta_path = os.path.join(dcfg.output_dir, f"{stem}{ecfg.metadata_suffix}")

        lstd_df.to_csv(lstd_path, index=False)

        meta = {
            "symbol": dcfg.symbol,
            "interval": dcfg.interval,
            "start_ms": start_ms,
            "end_ms": end_ms,
            "quality": quality,
            "feature_mode": ecfg.feature_mode,
            "target_column": ecfg.target_column,
            "used_feature_columns": used_feature_cols,
            "lstd_csv_path": lstd_path,
        }
        with open(meta_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)

        result["lstd_csv_path"] = lstd_path
        result["metadata_path"] = meta_path
        result["lstd_rows"] = len(lstd_df)
        result["lstd_columns"] = list(lstd_df.columns)

    return result
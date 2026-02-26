# binance_ts/utils.py

import os
import time
from datetime import datetime, timezone, timedelta
from typing import Optional, Tuple

import pandas as pd


_INTERVAL_MS = {
    "1m": 60_000,
    "3m": 3 * 60_000,
    "5m": 5 * 60_000,
    "15m": 15 * 60_000,
    "30m": 30 * 60_000,
    "1h": 60 * 60_000,
    "2h": 2 * 60 * 60_000,
    "4h": 4 * 60 * 60_000,
    "6h": 6 * 60 * 60_000,
    "8h": 8 * 60 * 60_000,
    "12h": 12 * 60 * 60_000,
    "1d": 24 * 60 * 60_000,
}

_INTERVAL_FREQ_STR = {
    "1m": "1min",
    "3m": "3min",
    "5m": "5min",
    "15m": "15min",
    "30m": "30min",
    "1h": "1H",
    "2h": "2H",
    "4h": "4H",
    "6h": "6H",
    "8h": "8H",
    "12h": "12H",
    "1d": "1D",
}


def interval_to_millis(interval: str) -> int:
    if interval not in _INTERVAL_MS:
        raise ValueError(f"Unsupported interval: {interval}")
    return _INTERVAL_MS[interval]


def interval_to_pandas_freq(interval: str) -> str:
    if interval not in _INTERVAL_FREQ_STR:
        raise ValueError(f"Unsupported interval: {interval}")
    return _INTERVAL_FREQ_STR[interval]


def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def parse_utc_datetime(dt_str: str) -> datetime:
    if not dt_str:
        raise ValueError("Empty datetime string")

    cleaned = dt_str.strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(cleaned)
    except ValueError:
        for fmt in ("%Y-%m-%d", "%Y-%m-%d %H:%M:%S"):
            try:
                dt = datetime.strptime(cleaned, fmt)
                break
            except ValueError:
                continue
        else:
            raise ValueError(f"Could not parse datetime: {dt_str}")

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    else:
        dt = dt.astimezone(timezone.utc)
    return dt


def dt_to_ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)


def align_ms_to_interval(ms: int, interval: str, mode: str = "floor") -> int:
    step = interval_to_millis(interval)
    if mode == "floor":
        return (ms // step) * step
    if mode == "ceil":
        return ((ms + step - 1) // step) * step
    raise ValueError("mode must be 'floor' or 'ceil'")


def resolve_time_range_ms(
    start: Optional[str],
    end: Optional[str],
    lookback_days: Optional[int],
    interval: str,
) -> Tuple[int, Optional[int]]:
    """
    Returns (start_ms, end_ms). end_ms can be None.
    Timestamps are aligned to interval boundaries.
    """
    now = datetime.now(timezone.utc)

    if start and end:
        start_dt = parse_utc_datetime(start)
        end_dt = parse_utc_datetime(end)
        if end_dt <= start_dt:
            raise ValueError("end must be after start")
        start_ms = align_ms_to_interval(dt_to_ms(start_dt), interval, "floor")
        end_ms = align_ms_to_interval(dt_to_ms(end_dt), interval, "floor")
        return start_ms, end_ms

    if lookback_days is not None:
        if lookback_days <= 0:
            raise ValueError("lookback_days must be positive")
        end_dt = parse_utc_datetime(end) if end else now
        start_dt = end_dt - timedelta(days=lookback_days)
        start_ms = align_ms_to_interval(dt_to_ms(start_dt), interval, "floor")
        end_ms = align_ms_to_interval(dt_to_ms(end_dt), interval, "floor")
        return start_ms, end_ms

    if start and not end:
        start_dt = parse_utc_datetime(start)
        start_ms = align_ms_to_interval(dt_to_ms(start_dt), interval, "floor")
        return start_ms, None

    raise ValueError("Provide (start and end), or lookback_days (+ optional end), or start only.")


def compact_time_str(ms: Optional[int]) -> str:
    if ms is None:
        return "open_end"
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y%m%d_%H%M")


def validate_kline_contiguity(df: pd.DataFrame, interval: str) -> dict:
    """
    Checks missing or duplicated bar spacing using open_time.
    """
    if df.empty or "open_time" not in df.columns:
        return {"ok": True, "missing_steps": 0, "duplicate_steps": 0}

    step = interval_to_millis(interval)
    diffs = df["open_time"].astype("int64").diff().dropna()

    missing = int((diffs > step).sum())
    duplicate_or_overlap = int((diffs < step).sum())

    return {
        "ok": (missing == 0 and duplicate_or_overlap == 0),
        "missing_steps": missing,
        "duplicate_steps": duplicate_or_overlap,
        "expected_step_ms": step,
    }
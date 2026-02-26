# binance_ts/binance_client.py

import time
from typing import List, Optional

import pandas as pd
import requests

from .utils import interval_to_millis


BINANCE_KLINE_COLUMNS = [
    "open_time",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "close_time",
    "quote_asset_volume",
    "number_of_trades",
    "taker_buy_base_asset_volume",
    "taker_buy_quote_asset_volume",
    "ignore",
]


class BinanceKlinesClient:
    def __init__(
        self,
        base_url: str = "https://api.binance.com",
        timeout: int = 20,
        max_retries: int = 5,
    ):
        self.base_url = base_url.rstrip("/")
        self.timeout = timeout
        self.max_retries = max_retries
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "binance-ts-lstd/1.0"})

    def _get_klines(
        self,
        symbol: str,
        interval: str,
        limit: int,
        start_time: Optional[int] = None,
        end_time: Optional[int] = None,
    ) -> List[list]:
        url = f"{self.base_url}/api/v3/klines"
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "limit": limit,
        }
        if start_time is not None:
            params["startTime"] = int(start_time)
        if end_time is not None:
            params["endTime"] = int(end_time)

        last_err = None
        for attempt in range(self.max_retries):
            try:
                resp = self.session.get(url, params=params, timeout=self.timeout)

                # Basic retry behavior for transient errors / rate limits
                if resp.status_code in (429, 418, 500, 502, 503, 504):
                    retry_after = resp.headers.get("Retry-After")
                    sleep_s = float(retry_after) if retry_after else (1.0 + attempt)
                    time.sleep(sleep_s)
                    continue

                resp.raise_for_status()
                data = resp.json()
                if not isinstance(data, list):
                    raise RuntimeError(f"Unexpected Binance response: {data}")
                return data
            except (requests.RequestException, ValueError, RuntimeError) as e:
                last_err = e
                time.sleep(1.0 + attempt)

        raise RuntimeError(f"Failed to fetch klines after retries: {last_err}")

    def fetch_historical_klines(
        self,
        symbol: str,
        interval: str,
        start_ms: int,
        end_ms: Optional[int] = None,
        request_limit: int = 1000,
        sleep_seconds: float = 0.15,
    ) -> pd.DataFrame:
        if request_limit <= 0 or request_limit > 1000:
            raise ValueError("request_limit must be between 1 and 1000")

        step_ms = interval_to_millis(interval)
        all_rows: List[list] = []
        current_start = int(start_ms)

        while True:
            chunk = self._get_klines(
                symbol=symbol,
                interval=interval,
                limit=request_limit,
                start_time=current_start,
                end_time=end_ms,
            )

            if not chunk:
                break

            all_rows.extend(chunk)

            last_open_time = int(chunk[-1][0])
            next_start = last_open_time + step_ms

            if next_start <= current_start:
                break

            current_start = next_start

            if len(chunk) < request_limit:
                break

            if end_ms is not None and current_start >= end_ms:
                break

            time.sleep(sleep_seconds)

        if not all_rows:
            return pd.DataFrame(columns=BINANCE_KLINE_COLUMNS)

        df = pd.DataFrame(all_rows, columns=BINANCE_KLINE_COLUMNS)

        # Types
        numeric_cols = [
            "open", "high", "low", "close", "volume",
            "quote_asset_volume",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
        ]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        int_cols = ["open_time", "close_time", "number_of_trades"]
        for col in int_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

        # Datetime columns (UTC)
        df["open_dt"] = pd.to_datetime(df["open_time"], unit="ms", utc=True)
        df["close_dt"] = pd.to_datetime(df["close_time"], unit="ms", utc=True)

        df = df.drop_duplicates(subset=["open_time"]).sort_values("open_time").reset_index(drop=True)

        if end_ms is not None:
            df = df[df["open_time"] < end_ms].reset_index(drop=True)

        return df
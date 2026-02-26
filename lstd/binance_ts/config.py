# binance_ts/config.py

from dataclasses import dataclass, field
from typing import Optional, List


@dataclass
class DownloadConfig:
    symbol: str = "BTCUSDT"
    interval: str = "15m"

    # Choose ONE style:
    # 1) explicit start/end (UTC)
    start: Optional[str] = "2024-01-01 00:00:00"
    end: Optional[str] = "2024-03-01 00:00:00"

    # 2) or use lookback_days if start/end not provided
    lookback_days: Optional[int] = None

    # API / IO
    base_url: str = "https://api.binance.com"
    request_limit: int = 1000
    sleep_seconds: float = 0.15
    timeout: int = 20
    max_retries: int = 5
    output_dir: str = "data"

    # Data quality checks
    validate_contiguity: bool = True
    allow_missing_candles: bool = False


@dataclass
class FeatureConfig:
    # Core features
    add_basic_price_features: bool = True

    add_instance_norm_features: bool = False
    instance_norm_window: int = 96

    add_seasonal_trend_features: bool = True
    seasonal_trend_window: int = 96

    add_frequency_features: bool = True
    frequency_window: int = 128

    add_long_short_regime_features: bool = True
    short_window: int = 16
    long_window: int = 96

    # IMPORTANT:
    # LSTD windowing already creates future targets by slicing seq_y,
    # so explicit shifted targets are usually OFF for LSTD training CSVs.
    add_explicit_targets: bool = False
    target_horizon_bars: int = 1

    drop_na_rows: bool = True


@dataclass
class LSTDExportConfig:
    enabled: bool = True

    # LSTD-style CSV format: date + [features...] + target
    # features mode semantics follow the repo:
    #   S  -> only target column
    #   M  -> multivariate features (target included at end)
    #   MS -> multivariate with target at end (same exported shape,
    #         different training semantics)
    feature_mode: str = "M"  # "S", "M", "MS"
    target_column: str = "close"

    # If None -> auto-select all numeric columns except excluded
    feature_columns: Optional[List[str]] = None

    # Columns to exclude from LSTD exported CSV
    exclude_columns: List[str] = field(default_factory=lambda: [
        "ignore",
        "open_time",
        "close_time",
    ])

    # Drop explicit shifted target columns (target_*), because LSTD creates
    # future targets via windows
    drop_explicit_target_columns: bool = True

    # File names
    lstd_csv_suffix: str = "_lstd.csv"
    metadata_suffix: str = "_meta.json"


@dataclass
class WindowConfig:
    # LSTD Dataset windowing params
    seq_len: int = 96
    label_len: int = 48
    pred_len: int = 16

    # Split ratios (chronological)
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15

    # Repo-like dataset flags
    features: str = "M"    # "S", "M", "MS"
    target: str = "close"
    scale: bool = True
    inverse: bool = False
    timeenc: int = 0       # 0 or 2 supported here
    freq: str = "15min"

    # Test rolling behavior
    delay_fb: bool = False


@dataclass
class PipelineConfig:
    download: DownloadConfig = field(default_factory=DownloadConfig)
    features: FeatureConfig = field(default_factory=FeatureConfig)
    export: LSTDExportConfig = field(default_factory=LSTDExportConfig)
    windows: WindowConfig = field(default_factory=WindowConfig)


DEFAULT_CONFIG = PipelineConfig()
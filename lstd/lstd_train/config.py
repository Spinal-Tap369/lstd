# lstd_train/config.py

from dataclasses import dataclass, field
from typing import Optional

from binance_ts.config import WindowConfig
from lstd_core.config import LSTDModelConfig


@dataclass
class OptimizerConfig:
    train_epochs: int = 20
    batch_size: int = 32
    val_batch_size: int = 64

    # IMPORTANT: online replay is most faithful with batch_size=1
    test_batch_size: int = 1

    learning_rate: float = 1e-3
    weight_decay: float = 1e-4

    # If set, gradients are clipped by norm
    grad_clip_norm: Optional[float] = None

    # DataLoader
    num_workers: int = 0
    pin_memory: bool = False

    # Mixed precision (CUDA only)
    use_amp: bool = False

    # Logging cadence fallback (tqdm is used now, but keeping this field is fine)
    print_every: int = 50


@dataclass
class RuntimeConfig:
    seed: int = 42
    device: str = "auto"  # "auto" | "cpu" | "cuda"

    checkpoints_dir: str = "checkpoints"
    outputs_dir: str = "outputs"
    experiment_name: str = "lstd_btc_offline"

    # Save the split CSVs (optional, for inspection/debugging)
    export_split_csvs: bool = True

    # Save preds/trues after test
    save_test_arrays: bool = True


@dataclass
class EvaluationConfig:
    # If True and scaling is enabled, also compute metrics in original (unscaled) units
    compute_unscaled_metrics: bool = True

    # Repo behavior:
    # - MS -> evaluate target channel only
    # - M  -> evaluate all channels
    # - S  -> single channel
    follow_repo_target_slice: bool = True


@dataclass
class OnlineAdaptConfig:
    """
    Online replay settings for validation/test stream:
    predict first -> reveal label -> update.
    """
    # "none"      = no online updates (static evaluation)
    # "full"      = update entire model
    # "regressor" = freeze encoder, update the rest (repo-style option)
    mode: str = "full"

    # Number of gradient updates after each revealed batch/window
    n_inner: int = 1

    # Enforce batch_size=1 for true sequential online replay
    require_batch_size_one: bool = True

    # Save final adapted model after online replay
    save_adapted_checkpoint: bool = True

    # If True, validation metric inside fit() is measured with online replay
    # on a temporary cloned model (so training weights are not contaminated).
    online_validate_during_fit: bool = False

    # Validation replay mode (if enabled above)
    val_mode: str = "full"
    val_n_inner: int = 1


@dataclass
class OfflineTrainConfig:
    """
    Offline training config for LSTD on your exported LSTD CSV:
    CSV must look like: ['date', ...(features), target]
    """
    root_path: str = "data"
    data_path: str = ""  # REQUIRED (e.g. BTCUSDT_15m_..._lstd.csv)

    windows: WindowConfig = field(default_factory=WindowConfig)
    model: LSTDModelConfig = field(default_factory=LSTDModelConfig)

    optim: OptimizerConfig = field(default_factory=OptimizerConfig)
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    eval: EvaluationConfig = field(default_factory=EvaluationConfig)
    online: OnlineAdaptConfig = field(default_factory=OnlineAdaptConfig)

    # Early stopping
    patience: int = 5
# lstd_core/config.py

from dataclasses import dataclass


@dataclass
class LSTDModelConfig:
    # Core forecasting dimensions
    seq_len: int = 96
    pred_len: int = 16
    enc_in: int = 1  # number of input variables/channels

    # LSTD mode:
    # - "feature": standard time-series mode (B, T, C)
    # - "time":    time-axis-as-feature mode from the repo
    mode: str = "feature"

    # Backbone (TS2Vec / FSNet)
    ts_hidden_dims: int = 64
    ts_output_dims: int = 320
    depth: int = 10
    gamma: float = 0.9      # slow EMA for gradient state
    tau: float = 0.5        # memory trigger / blending control
    use_adaptive_memory_conv: bool = True  # fsnet_ vs plain dilated conv

    # LSTD MLP blocks
    hidden_dim: int = 128
    hidden_layers: int = 2
    dropout: float = 0.1
    activation: str = "gelu"

    # Transition prior
    lags: int = 1

    # Loss weights (as in repo)
    zc_kl_weight: float = 1.0
    zd_kl_weight: float = 1.0
    L1_weight: float = 0.0
    L2_weight: float = 0.0
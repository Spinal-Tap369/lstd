# lstd_core/config.py

from dataclasses import dataclass


@dataclass
class LSTDModelConfig:
    # Core forecasting dimensions
    seq_len: int = 96
    pred_len: int = 16
    enc_in: int = 1  # number of variables/channels

    # Paper mentions a "mode" hyperparameter, but this rewritten
    # Table-3-faithful implementation only supports the standard feature path.
    mode: str = "feature"

    # ------------------------------------------------------------------
    # Paper-faithful architecture params (Table 3 / Appendix C)
    # ------------------------------------------------------------------
    long_conv_hidden: int = 640      # phi^s hidden conv channels
    short_mlp_hidden: int = 512      # phi^d and T^s hidden width
    future_mlp_hidden: int = 512     # F_y hidden width

    # Prior network r_i
    lags: int = 1
    prior_hidden_dim: int = 128
    prior_num_hidden_layers: int = 3

    # Variational/prior weights
    zc_kl_weight: float = 1.0
    zd_kl_weight: float = 1.0

    # IMPORTANT:
    # L1_weight now means the paper-style sparse dependency penalty (L_s)
    # L2_weight now means the paper-style smooth/attention penalty (L_m)
    L1_weight: float = 1.0
    L2_weight: float = 1.0

    # Kept for trainer / CLI compatibility. Not used by the rewritten model.
    hidden_dim: int = 128
    hidden_layers: int = 2
    dropout: float = 0.0
    activation: str = "leaky_relu"

    # Kept only so your existing CLI/trainer does not break.
    ts_hidden_dims: int = 64
    ts_output_dims: int = 320
    depth: int = 10
    gamma: float = 0.9
    tau: float = 0.5
    use_adaptive_memory_conv: bool = True
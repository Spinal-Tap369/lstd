# lstd_train/run_example.py

from __future__ import annotations

import argparse
from pprint import pprint
from typing import Optional

from .config import OfflineTrainConfig
from .trainer import OfflineLSTDTrainer


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Paper-style LSTD: warm-up pretraining + online replay evaluation."
    )

    # Required-ish
    parser.add_argument("--root-path", type=str, default="data")
    parser.add_argument("--data-path", type=str, required=True)


    # Windowing / split

    parser.add_argument("--seq-len", type=int, default=96)
    parser.add_argument("--label-len", type=int, default=48)
    parser.add_argument("--pred-len", type=int, default=16)

    # 20% train + 5% val = 25% warm-up, 75% online/test
    parser.add_argument("--train-ratio", type=float, default=0.20)
    parser.add_argument("--val-ratio", type=float, default=0.05)
    parser.add_argument("--test-ratio", type=float, default=0.75)

    # For BTC multivariate context with single target supervision
    parser.add_argument("--features", type=str, choices=["S", "M", "MS"], default="MS")
    parser.add_argument("--target", type=str, default="close")
    parser.add_argument("--scale", action="store_true")
    parser.add_argument("--no-scale", action="store_true")
    parser.add_argument("--timeenc", type=int, default=2)
    parser.add_argument("--freq", type=str, default="15min")
    parser.add_argument("--delay-fb", action="store_true")


    # Model 

    parser.add_argument("--model-mode", type=str, choices=["feature"], default="feature")

    # Table 3 architecture widths
    parser.add_argument("--long-conv-hidden", type=int, default=640)
    parser.add_argument("--short-mlp-hidden", type=int, default=512)
    parser.add_argument("--future-mlp-hidden", type=int, default=512)

    # Prior networks r_i
    parser.add_argument("--lags", type=int, default=1)
    parser.add_argument("--prior-hidden-dim", type=int, default=128)
    parser.add_argument("--prior-num-hidden-layers", type=int, default=3)

    # Loss weights
    parser.add_argument("--zc-kl-weight", type=float, default=1.0)
    parser.add_argument("--zd-kl-weight", type=float, default=1.0)

    # Start conservative:
    # L1 = sparse dependency constraint (expensive)
    # L2 = long smoothness constraint
    parser.add_argument("--l1-weight", type=float, default=0.0)
    parser.add_argument("--l2-weight", type=float, default=1e-2)


    # Optimization
    parser.add_argument("--train-epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--val-batch-size", type=int, default=1)
    parser.add_argument("--test-batch-size", type=int, default=1)

    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--grad-clip-norm", type=float, default=1.0)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--pin-memory", action="store_true")
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--patience", type=int, default=1)

    # Runtime
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", type=str, choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--experiment-name", type=str, default="btc_lstd_paper_faithful")
    parser.add_argument("--checkpoints-dir", type=str, default="checkpoints")
    parser.add_argument("--outputs-dir", type=str, default="outputs")
    parser.add_argument("--no-export-split-csvs", action="store_true")
    parser.add_argument("--no-save-test-arrays", action="store_true")
    parser.add_argument("--no-unscaled-metrics", action="store_true")

    # Online replay
    parser.add_argument("--online-mode", type=str, choices=["none", "full", "regressor"], default="full")
    parser.add_argument("--n-inner", type=int, default=1)
    parser.add_argument("--disable-batch-size-one-check", action="store_true")
    parser.add_argument("--no-save-adapted-checkpoint", action="store_true")
    parser.add_argument("--online-validate-during-fit", action="store_true")
    parser.add_argument("--val-online-mode", type=str, choices=["none", "full", "regressor"], default="full")
    parser.add_argument("--val-n-inner", type=int, default=1)

    # Run mode
    parser.add_argument("--skip-train", action="store_true")
    parser.add_argument("--eval-only", action="store_true")
    parser.add_argument("--test-protocol", type=str, choices=["online", "static"], default="online")

    return parser


def build_config_from_args(args: argparse.Namespace) -> OfflineTrainConfig:
    cfg = OfflineTrainConfig()

    # Dataset
    cfg.root_path = args.root_path
    cfg.data_path = args.data_path

    # Windows
    cfg.windows.seq_len = args.seq_len
    cfg.windows.label_len = args.label_len
    cfg.windows.pred_len = args.pred_len
    cfg.windows.train_ratio = args.train_ratio
    cfg.windows.val_ratio = args.val_ratio
    cfg.windows.test_ratio = args.test_ratio
    cfg.windows.features = args.features
    cfg.windows.target = args.target
    cfg.windows.timeenc = args.timeenc
    cfg.windows.freq = args.freq
    cfg.windows.delay_fb = bool(args.delay_fb)

    if args.no_scale:
        cfg.windows.scale = False
    elif args.scale:
        cfg.windows.scale = True


    # Model
    cfg.model.mode = args.model_mode

    # Table 3 architecture
    cfg.model.long_conv_hidden = args.long_conv_hidden
    cfg.model.short_mlp_hidden = args.short_mlp_hidden
    cfg.model.future_mlp_hidden = args.future_mlp_hidden

    # Prior nets
    cfg.model.lags = args.lags
    cfg.model.prior_hidden_dim = args.prior_hidden_dim
    cfg.model.prior_num_hidden_layers = args.prior_num_hidden_layers

    # Loss weights
    cfg.model.zc_kl_weight = args.zc_kl_weight
    cfg.model.zd_kl_weight = args.zd_kl_weight
    cfg.model.L1_weight = args.l1_weight
    cfg.model.L2_weight = args.l2_weight

    # Optimization
    cfg.optim.train_epochs = args.train_epochs
    cfg.optim.batch_size = args.batch_size
    cfg.optim.val_batch_size = args.val_batch_size
    cfg.optim.test_batch_size = args.test_batch_size
    cfg.optim.learning_rate = args.learning_rate
    cfg.optim.weight_decay = args.weight_decay
    cfg.optim.grad_clip_norm = args.grad_clip_norm
    cfg.optim.num_workers = args.num_workers
    cfg.optim.pin_memory = bool(args.pin_memory)
    cfg.optim.use_amp = bool(args.use_amp)

    # Runtime
    cfg.runtime.seed = args.seed
    cfg.runtime.device = args.device
    cfg.runtime.experiment_name = args.experiment_name
    cfg.runtime.checkpoints_dir = args.checkpoints_dir
    cfg.runtime.outputs_dir = args.outputs_dir
    cfg.runtime.export_split_csvs = not args.no_export_split_csvs
    cfg.runtime.save_test_arrays = not args.no_save_test_arrays

    # Eval
    cfg.eval.compute_unscaled_metrics = not args.no_unscaled_metrics

    # Online replay
    cfg.online.mode = args.online_mode
    cfg.online.n_inner = args.n_inner
    cfg.online.require_batch_size_one = not args.disable_batch_size_one_check
    cfg.online.save_adapted_checkpoint = not args.no_save_adapted_checkpoint
    cfg.online.online_validate_during_fit = bool(args.online_validate_during_fit)
    cfg.online.val_mode = args.val_online_mode
    cfg.online.val_n_inner = args.val_n_inner

    # Early stopping
    cfg.patience = args.patience

    return cfg


def run_with_config(
    cfg: OfflineTrainConfig,
    *,
    skip_train: bool = False,
    eval_only: bool = False,
    test_protocol: str = "online",
) -> dict:
    trainer = OfflineLSTDTrainer(cfg)

    train_summary = None
    if not skip_train and not eval_only:
        print("\n[1] Offline pretraining / warm-up...")
        train_summary = trainer.fit()
        pprint(train_summary["history"])

    print("\n[2] Evaluation...")
    if test_protocol == "static":
        test_summary = trainer.test_static(load_best=True)
    else:
        test_summary = trainer.test(load_best=True)

    pprint(test_summary)

    return {
        "train_summary": train_summary,
        "test_summary": test_summary,
    }


def main(argv: Optional[list[str]] = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)

    cfg = build_config_from_args(args)
    run_with_config(
        cfg,
        skip_train=bool(args.skip_train),
        eval_only=bool(args.eval_only),
        test_protocol=args.test_protocol,
    )


if __name__ == "__main__":
    main()
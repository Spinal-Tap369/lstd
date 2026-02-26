# lstd_train/data.py

import os
from dataclasses import dataclass
from typing import Optional

import pandas as pd
from torch.utils.data import DataLoader

from binance_ts.dataset import BinanceLSTDDataset
from .config import OfflineTrainConfig
from .utils import ensure_dir


@dataclass
class LoaderBundle:
    train_dataset: BinanceLSTDDataset
    train_loader: DataLoader

    val_dataset: Optional[BinanceLSTDDataset]
    val_loader: Optional[DataLoader]

    test_dataset: BinanceLSTDDataset
    test_loader: DataLoader

    enc_in: int
    target_index: int  # target is last channel by LSTD CSV convention
    csv_columns: list[str]


def export_chronological_splits(cfg: OfflineTrainConfig) -> dict:
    """
    Optional utility to write explicit train/val/test CSV files for inspection.
    Training still uses the in-dataset chronological split (with overlap logic).
    """
    full_path = os.path.join(cfg.root_path, cfg.data_path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"CSV not found: {full_path}")

    df = pd.read_csv(full_path)
    if "date" not in df.columns:
        raise ValueError("Expected a 'date' column in LSTD CSV")

    n = len(df)
    tr = cfg.windows.train_ratio
    vr = cfg.windows.val_ratio
    te = cfg.windows.test_ratio
    if abs((tr + vr + te) - 1.0) > 1e-6:
        raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

    n_train = int(n * tr)
    n_val = int(n * vr)
    n_test = n - n_train - n_val

    out_dir = os.path.join(cfg.runtime.outputs_dir, cfg.runtime.experiment_name, "splits")
    ensure_dir(out_dir)

    stem = os.path.splitext(os.path.basename(cfg.data_path))[0]

    train_df = df.iloc[:n_train].copy()
    val_df = df.iloc[n_train:n_train + n_val].copy()
    test_df = df.iloc[n_train + n_val:].copy()

    paths = {
        "train_csv": os.path.join(out_dir, f"{stem}_train.csv"),
        "val_csv": os.path.join(out_dir, f"{stem}_val.csv"),
        "test_csv": os.path.join(out_dir, f"{stem}_test.csv"),
    }

    train_df.to_csv(paths["train_csv"], index=False)
    val_df.to_csv(paths["val_csv"], index=False)
    test_df.to_csv(paths["test_csv"], index=False)

    meta = {
        "rows_total": n,
        "rows_train": len(train_df),
        "rows_val": len(val_df),
        "rows_test": len(test_df),
        "train_ratio": tr,
        "val_ratio": vr,
        "test_ratio": te,
    }
    return {"paths": paths, "meta": meta}


def build_loaders(cfg: OfflineTrainConfig) -> LoaderBundle:
    wc = cfg.windows
    oc = cfg.optim

    if not cfg.data_path:
        raise ValueError("OfflineTrainConfig.data_path is required (path to your *_lstd.csv file).")

    full_path = os.path.join(cfg.root_path, cfg.data_path)
    if not os.path.exists(full_path):
        raise FileNotFoundError(f"LSTD CSV not found: {full_path}")

    size = [wc.seq_len, wc.label_len, wc.pred_len]

    # Train
    train_ds = BinanceLSTDDataset(
        root_path=cfg.root_path,
        data_path=cfg.data_path,
        flag="train",
        delay_fb=wc.delay_fb,
        size=size,
        features=wc.features,
        target=wc.target,
        scale=wc.scale,
        inverse=wc.inverse,
        timeenc=wc.timeenc,
        freq=wc.freq,
        cols=None,
        train_ratio=wc.train_ratio,
        val_ratio=wc.val_ratio,
        test_ratio=wc.test_ratio,
    )

    # Validation (optional if val_ratio == 0)
    val_ds = None
    val_loader = None
    if wc.val_ratio > 0:
        val_ds = BinanceLSTDDataset(
            root_path=cfg.root_path,
            data_path=cfg.data_path,
            flag="val",
            delay_fb=wc.delay_fb,
            size=size,
            features=wc.features,
            target=wc.target,
            scale=wc.scale,
            inverse=wc.inverse,
            timeenc=wc.timeenc,
            freq=wc.freq,
            cols=None,
            train_ratio=wc.train_ratio,
            val_ratio=wc.val_ratio,
            test_ratio=wc.test_ratio,
        )

    # Test
    test_ds = BinanceLSTDDataset(
        root_path=cfg.root_path,
        data_path=cfg.data_path,
        flag="test",
        delay_fb=wc.delay_fb,
        size=size,
        features=wc.features,
        target=wc.target,
        scale=wc.scale,
        inverse=wc.inverse,
        timeenc=wc.timeenc,
        freq=wc.freq,
        cols=None,
        train_ratio=wc.train_ratio,
        val_ratio=wc.val_ratio,
        test_ratio=wc.test_ratio,
    )

    if len(train_ds) <= 0:
        raise RuntimeError(
            f"Train dataset length is {len(train_ds)}. "
            f"Increase data range or reduce seq_len/pred_len."
        )
    if val_ds is not None and len(val_ds) <= 0:
        raise RuntimeError(
            f"Val dataset length is {len(val_ds)}. "
            f"Increase data range or reduce seq_len/pred_len (or set val_ratio=0)."
        )
    if len(test_ds) <= 0:
        raise RuntimeError(
            f"Test dataset length is {len(test_ds)}. "
            f"Increase data range or reduce seq_len/pred_len."
        )

    pin_memory = bool(oc.pin_memory)

    train_loader = DataLoader(
        train_ds,
        batch_size=oc.batch_size,
        shuffle=True,
        num_workers=oc.num_workers,
        drop_last=True,
        pin_memory=pin_memory,
    )

    if val_ds is not None:
        val_loader = DataLoader(
            val_ds,
            batch_size=oc.val_batch_size,
            shuffle=False,
            num_workers=oc.num_workers,
            drop_last=False,
            pin_memory=pin_memory,
        )

    test_loader = DataLoader(
        test_ds,
        batch_size=oc.test_batch_size,
        shuffle=False,
        num_workers=oc.num_workers,
        drop_last=False,
        pin_memory=pin_memory,
    )

    # Infer input channels directly from dataset data_x
    enc_in = int(train_ds.data_x.shape[1])

    # Target column is last column in exported LSTD CSV by your build_lstd_ready_frame()
    # For "M"/"MS": all non-date columns are in data_x and target is last.
    # For "S": only one channel exists, index 0.
    target_index = enc_in - 1 if enc_in > 0 else 0

    # Read CSV columns for metadata/debug
    csv_cols = pd.read_csv(full_path, nrows=1).columns.tolist()

    return LoaderBundle(
        train_dataset=train_ds,
        train_loader=train_loader,
        val_dataset=val_ds,
        val_loader=val_loader,
        test_dataset=test_ds,
        test_loader=test_loader,
        enc_in=enc_in,
        target_index=target_index,
        csv_columns=csv_cols,
    )
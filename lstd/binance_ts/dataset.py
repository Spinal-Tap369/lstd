# binance_ts/dataset.py

import os
from dataclasses import dataclass
from typing import Optional, List

import numpy as np
import pandas as pd
from torch.utils.data import Dataset

from .time_features import time_features


class StandardScalerNumpy:
    def __init__(self):
        self.mean_: Optional[np.ndarray] = None
        self.std_: Optional[np.ndarray] = None

    def fit(self, x: np.ndarray) -> "StandardScalerNumpy":
        mean = np.asarray(np.mean(x, axis=0, keepdims=True), dtype=np.float32)
        std = np.asarray(np.std(x, axis=0, keepdims=True), dtype=np.float32)

        std[std < 1e-12] = 1.0

        self.mean_ = mean
        self.std_ = std
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        mean = self.mean_
        std = self.std_
        if mean is None or std is None:
            raise RuntimeError("Scaler not fitted")
        return (x - mean) / std

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        mean = self.mean_
        std = self.std_
        if mean is None or std is None:
            raise RuntimeError("Scaler not fitted")
        return x * std + mean


class BinanceLSTDDataset(Dataset):
    """
    LSTD-style dataset built from the exported CSV:
      ['date', ...(features), target]
    """
    def __init__(
        self,
        root_path: str,
        data_path: str,
        flag: str = "train",           # train / val / test
        delay_fb: bool = False,
        size: Optional[List[int]] = None,   # [seq_len, label_len, pred_len]
        features: str = "M",           # S / M / MS
        target: str = "close",
        scale: bool = True,
        inverse: bool = False,
        timeenc: int = 0,
        freq: str = "15min",
        cols: Optional[List[str]] = None,
        train_ratio: float = 0.70,
        val_ratio: float = 0.15,
        test_ratio: float = 0.15,
    ):
        if size is None:
            self.seq_len = 96
            self.label_len = 48
            self.pred_len = 16
        else:
            self.seq_len, self.label_len, self.pred_len = size

        assert flag in {"train", "val", "test"}
        self.flag = flag
        self.set_type = {"train": 0, "val": 1, "test": 2}[flag]

        if abs((train_ratio + val_ratio + test_ratio) - 1.0) > 1e-6:
            raise ValueError("train_ratio + val_ratio + test_ratio must equal 1.0")

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.delay_fb = delay_fb
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path

        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.test_ratio = test_ratio

        self.scaler = StandardScalerNumpy()
        self.date_scaler = StandardScalerNumpy()

        self.__read_data__()

    def __read_data__(self):
        full_path = os.path.join(self.root_path, self.data_path)
        df_raw = pd.read_csv(full_path)

        if "date" not in df_raw.columns:
            raise ValueError("Dataset CSV must contain 'date' column")

        if self.cols:
            cols = self.cols.copy()
            if self.target in cols:
                cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove("date")
            if self.target not in cols:
                raise ValueError(f"Target '{self.target}' not found in CSV")
            cols.remove(self.target)

        # LSTD repo convention: target at end
        df_raw = df_raw[["date"] + cols + [self.target]]

        n = len(df_raw)
        n_train = int(n * self.train_ratio)
        n_val = int(n * self.val_ratio)
        n_test = n - n_train - n_val

        # Chronological borders with seq overlap like repo
        border1s = [0, n_train - self.seq_len, n_train + n_val - self.seq_len]
        border2s = [n_train, n_train + n_val, n]

        border1 = border1s[self.set_type]
        border2 = border2s[self.set_type]

        if self.features in {"M", "MS"}:
            cols_data = df_raw.columns[1:]  # all except date
            df_data = df_raw[cols_data]
        elif self.features == "S":
            df_data = df_raw[[self.target]]
        else:
            raise ValueError("features must be one of: S, M, MS")

        if self.scale:
            train_data = df_data.iloc[border1s[0]:border2s[0]].values.astype(np.float32)
            self.scaler.fit(train_data)
            data = self.scaler.transform(df_data.values.astype(np.float32))
        else:
            data = df_data.values.astype(np.float32)

        # Time marks
        train_stamp = pd.DataFrame({"date": pd.to_datetime(df_raw["date"].iloc[border1s[0]:border2s[0]])})
        train_mark = time_features(train_stamp, timeenc=self.timeenc, freq=self.freq)
        if self.timeenc == 2:
            self.date_scaler.fit(train_mark)

        df_stamp = pd.DataFrame({"date": pd.to_datetime(df_raw["date"].iloc[border1:border2])})
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)
        if self.timeenc == 2:
            data_stamp = self.date_scaler.transform(data_stamp)

        self.data_x = data[border1:border2]
        if self.inverse:
            self.data_y = df_data.values.astype(np.float32)[border1:border2]
        else:
            self.data_y = data[border1:border2]
        self.data_stamp = data_stamp.astype(np.float32)

    def __getitem__(self, index: int):
        if self.delay_fb and self.set_type == 2:
            s_begin = index * self.pred_len
            s_end = s_begin + self.seq_len
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len
        else:
            s_begin = index
            s_end = s_begin + self.seq_len
            r_begin = s_end - self.label_len
            r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        if self.inverse:
            seq_y = np.concatenate(
                [self.data_x[r_begin:r_begin + self.label_len], self.data_y[r_begin + self.label_len:r_end]],
                axis=0
            )
        else:
            seq_y = self.data_y[r_begin:r_end]

        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]

        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        if self.delay_fb and self.set_type == 2:
            return (len(self.data_x) - self.seq_len - self.pred_len) // self.pred_len
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform(data)


class BinanceLSTDPredDataset(Dataset):
    """
    Prediction dataset similar to repo Dataset_Pred:
    uses the last seq_len rows and generates future time marks.
    """
    def __init__(
        self,
        root_path: str,
        data_path: str,
        size: Optional[List[int]] = None,
        features: str = "M",
        target: str = "close",
        scale: bool = True,
        inverse: bool = False,
        timeenc: int = 0,
        freq: str = "15min",
        cols: Optional[List[str]] = None,
    ):
        if size is None:
            self.seq_len = 96
            self.label_len = 48
            self.pred_len = 16
        else:
            self.seq_len, self.label_len, self.pred_len = size

        self.features = features
        self.target = target
        self.scale = scale
        self.inverse = inverse
        self.timeenc = timeenc
        self.freq = freq
        self.cols = cols
        self.root_path = root_path
        self.data_path = data_path
        self.scaler = StandardScalerNumpy()

        self.__read_data__()

    def __read_data__(self):
        df_raw = pd.read_csv(os.path.join(self.root_path, self.data_path))
        if self.cols:
            cols = self.cols.copy()
            if self.target in cols:
                cols.remove(self.target)
        else:
            cols = list(df_raw.columns)
            cols.remove("date")
            cols.remove(self.target)
        df_raw = df_raw[["date"] + cols + [self.target]]

        border1 = len(df_raw) - self.seq_len
        border2 = len(df_raw)

        if self.features in {"M", "MS"}:
            df_data = df_raw[df_raw.columns[1:]]
        else:
            df_data = df_raw[[self.target]]

        if self.scale:
            self.scaler.fit(df_data.values.astype(np.float32))
            data = self.scaler.transform(df_data.values.astype(np.float32))
        else:
            data = df_data.values.astype(np.float32)

        tmp_stamp = pd.DataFrame({"date": pd.to_datetime(df_raw["date"].iloc[border1:border2])})
        last_ts = tmp_stamp["date"].iloc[-1]
        pred_dates = pd.date_range(last_ts, periods=self.pred_len + 1, freq=self.freq)

        df_stamp = pd.DataFrame({"date": list(tmp_stamp["date"].values) + list(pred_dates[1:].values)})
        data_stamp = time_features(df_stamp, timeenc=self.timeenc, freq=self.freq)

        self.data_x = data[border1:border2]
        self.data_y = (df_data.values.astype(np.float32) if self.inverse else data)[border1:border2]
        self.data_stamp = data_stamp.astype(np.float32)

    def __getitem__(self, index: int):
        s_begin = index
        s_end = s_begin + self.seq_len
        r_begin = s_end - self.label_len
        r_end = r_begin + self.label_len + self.pred_len

        seq_x = self.data_x[s_begin:s_end]
        seq_y = self.data_x[r_begin:r_begin + self.label_len] if self.inverse else self.data_y[r_begin:r_begin + self.label_len]
        seq_x_mark = self.data_stamp[s_begin:s_end]
        seq_y_mark = self.data_stamp[r_begin:r_end]
        return seq_x, seq_y, seq_x_mark, seq_y_mark

    def __len__(self):
        return len(self.data_x) - self.seq_len - self.pred_len + 1

    def inverse_transform(self, data: np.ndarray) -> np.ndarray:
        return self.scaler.inverse_transform(data)
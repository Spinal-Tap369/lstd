# lstd_train/trainer.py

from __future__ import annotations

import os
import time
from copy import deepcopy
from typing import Dict, Any, Optional, Tuple, List

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from tqdm.auto import tqdm

from lstd_core.model import LSTDNet

from .config import OfflineTrainConfig
from .data import build_loaders, export_chronological_splits, LoaderBundle
from .early_stopping import EarlyStopping
from .metrics import regression_metrics
from .utils import ensure_dir, save_json, set_seed, choose_device, timestamp_tag


class OfflineLSTDTrainer:
    """
    - fit(): offline pretraining (standard supervised optimization)
    - run_offline()/test(): online replay evaluation on a stream
      (predict -> reveal label -> update), matching paper protocol.
    """
    def __init__(self, cfg: OfflineTrainConfig):
        self.cfg = deepcopy(cfg)

        set_seed(self.cfg.runtime.seed)
        self.device = choose_device(self.cfg.runtime.device)

        self.bundle: LoaderBundle = build_loaders(self.cfg)

        # Sync model dimensions to dataset/window settings
        self.cfg.model.seq_len = self.cfg.windows.seq_len
        self.cfg.model.pred_len = self.cfg.windows.pred_len
        self.cfg.model.enc_in = self.bundle.enc_in

        self.model = LSTDNet(self.cfg.model, device=self.device)
        self.criterion = nn.MSELoss()
        self.optimizer = AdamW(
            self.model.parameters(),
            lr=self.cfg.optim.learning_rate,
            weight_decay=self.cfg.optim.weight_decay,
        )

        self.use_amp = bool(self.cfg.optim.use_amp) and (self.device.type == "cuda")
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.use_amp)

        # Run directories
        self.run_id = f"{self.cfg.runtime.experiment_name}_{timestamp_tag()}"
        self.ckpt_dir = os.path.join(self.cfg.runtime.checkpoints_dir, self.run_id)
        self.out_dir = os.path.join(self.cfg.runtime.outputs_dir, self.run_id)
        ensure_dir(self.ckpt_dir)
        ensure_dir(self.out_dir)

        self.best_ckpt_path = os.path.join(self.ckpt_dir, "checkpoint.pth")

        # Optional explicit split CSVs (for human inspection)
        self.split_export_info: Optional[dict] = None
        if self.cfg.runtime.export_split_csvs:
            self.split_export_info = export_chronological_splits(self.cfg)

    # -----------------------------
    # Core tensor helpers
    # -----------------------------
    def _extract_pred_true(
        self,
        outputs_flat: torch.Tensor,
        batch_y: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Repo-like behavior:
        - features == 'MS' -> evaluate only target channel (last column)
        - features == 'M'  -> evaluate all channels
        - features == 'S'  -> single channel
        Returns flattened tensors [B, pred_len * channels_eval]
        """
        pred_len = self.cfg.windows.pred_len
        enc_in = self.bundle.enc_in
        B = outputs_flat.shape[0]

        pred_seq = outputs_flat.view(B, pred_len, enc_in)

        # batch_y shape from dataset is [B, label_len + pred_len, C]
        true_seq_all = batch_y[:, -pred_len:, :].to(self.device).float()

        if self.cfg.windows.features.upper() == "MS":
            pred_seq = pred_seq[:, :, -1:]       # target is last
            true_seq = true_seq_all[:, :, -1:]
        else:
            true_seq = true_seq_all

        pred_flat = pred_seq.reshape(B, -1)
        true_flat = true_seq.reshape(B, -1)

        return pred_flat, true_flat

    def _forward_losses(
        self,
        batch_x: torch.Tensor,
        batch_y: torch.Tensor,
        is_training_forward: bool,
    ) -> Dict[str, torch.Tensor]:
        x = batch_x.to(self.device).float()

        x_rec, outputs_flat, other_loss = self.model(x, is_training=is_training_forward)
        pred_flat, true_flat = self._extract_pred_true(outputs_flat, batch_y)

        pred_loss = self.criterion(pred_flat, true_flat)
        rec_loss = self.criterion(x_rec, x)
        total_loss = pred_loss + rec_loss + other_loss

        return {
            "x": x,
            "x_rec": x_rec,
            "outputs_flat": outputs_flat,
            "pred_flat": pred_flat,
            "true_flat": true_flat,
            "pred_loss": pred_loss,
            "rec_loss": rec_loss,
            "other_loss": other_loss,
            "total_loss": total_loss,
        }

    def _inverse_scale_for_metrics(
        self,
        pred_np: np.ndarray,   # [N, pred_len, C_eval]
        true_np: np.ndarray,   # [N, pred_len, C_eval]
        dataset,
    ) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Convert scaled predictions back to original units using dataset scaler.
        Works for:
          - S (single channel)
          - M (all channels)
          - MS (target-only; assumed target is last channel in LSTD CSV)
        Returns None if scaling disabled or scaler not available.
        """
        if not self.cfg.windows.scale:
            return None

        scaler = getattr(dataset, "scaler", None)
        if scaler is None or scaler.mean_ is None or scaler.std_ is None:
            return None

        mean = np.asarray(scaler.mean_, dtype=np.float32)  # [1, C_all]
        std = np.asarray(scaler.std_, dtype=np.float32)    # [1, C_all]

        mode = self.cfg.windows.features.upper()
        if mode in {"M", "S"}:
            return (
                pred_np * std.reshape(1, 1, -1) + mean.reshape(1, 1, -1),
                true_np * std.reshape(1, 1, -1) + mean.reshape(1, 1, -1),
            )

        if mode == "MS":
            target_mean = float(mean[0, -1])
            target_std = float(std[0, -1])
            return pred_np * target_std + target_mean, true_np * target_std + target_mean

        return None

    # -----------------------------
    # Offline training (pretraining)
    # -----------------------------
    def _train_one_epoch(self, epoch_idx: int) -> float:
        self.model.train()
        losses: List[float] = []

        pbar = tqdm(
            self.bundle.train_loader,
            total=len(self.bundle.train_loader),
            desc=f"Train {epoch_idx + 1}/{self.cfg.optim.train_epochs}",
            leave=False,
            dynamic_ncols=True,
        )

        for step_idx, batch in enumerate(pbar, start=1):
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch
            del batch_x_mark, batch_y_mark

            self.optimizer.zero_grad(set_to_none=True)

            if self.use_amp:
                with torch.cuda.amp.autocast():
                    out = self._forward_losses(batch_x, batch_y, is_training_forward=True)

                self.scaler.scale(out["total_loss"]).backward()

                if self.cfg.optim.grad_clip_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.optim.grad_clip_norm)

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                out = self._forward_losses(batch_x, batch_y, is_training_forward=True)
                out["total_loss"].backward()

                if self.cfg.optim.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.optim.grad_clip_norm)

                self.optimizer.step()

            # Important for FSNet adaptive-memory behavior
            self.model.store_grad()

            total_loss_val = float(out["total_loss"].detach().cpu().item())
            pred_loss_val = float(out["pred_loss"].detach().cpu().item())
            rec_loss_val = float(out["rec_loss"].detach().cpu().item())
            other_loss_val = float(out["other_loss"].detach().cpu().item())
            losses.append(total_loss_val)

            pbar.set_postfix(
                step=f"{step_idx}/{len(self.bundle.train_loader)}",
                loss=f"{total_loss_val:.4f}",
                pred=f"{pred_loss_val:.4f}",
                rec=f"{rec_loss_val:.4f}",
                oth=f"{other_loss_val:.4f}",
                avg=f"{np.mean(losses):.4f}",
                lr=f"{self.optimizer.param_groups[0]['lr']:.2e}",
            )

        return float(np.mean(losses)) if losses else float("inf")

    @torch.no_grad()
    def _validate_static(self) -> Optional[float]:
        if self.bundle.val_loader is None:
            return None

        self.model.eval()
        losses = []

        pbar = tqdm(
            self.bundle.val_loader,
            total=len(self.bundle.val_loader),
            desc="Validate",
            leave=False,
            dynamic_ncols=True,
        )

        for batch in pbar:
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch
            del batch_x_mark, batch_y_mark

            out = self._forward_losses(batch_x, batch_y, is_training_forward=False)

            # Keep validation monitor clean: prediction loss only
            loss = float(out["pred_loss"].detach().cpu().item())
            losses.append(loss)

            pbar.set_postfix(avg_pred=f"{np.mean(losses):.4f}")

        self.model.train()
        return float(np.mean(losses)) if losses else None

    def _validate_online_clone(self) -> Optional[float]:
        """
        Optional online validation replay on a cloned model/optimizer so that
        validation updates do NOT contaminate the training model.
        """
        if self.bundle.val_loader is None:
            return None

        val_mode = self.cfg.online.val_mode.lower()
        if val_mode not in {"none", "full", "regressor"}:
            raise ValueError("cfg.online.val_mode must be one of: none, full, regressor")

        # Clone model and optimizer state
        model_clone = LSTDNet(self.cfg.model, device=self.device)
        model_clone.load_state_dict(deepcopy(self.model.state_dict()))
        model_clone.eval()

        opt_clone = AdamW(
            model_clone.parameters(),
            lr=self.cfg.optim.learning_rate,
            weight_decay=self.cfg.optim.weight_decay,
        )
        opt_clone.load_state_dict(deepcopy(self.optimizer.state_dict()))

        losses = []

        pbar = tqdm(
            self.bundle.val_loader,
            total=len(self.bundle.val_loader),
            desc=f"Val-Online[{val_mode}]",
            leave=False,
            dynamic_ncols=True,
        )

        # freeze encoder if regressor-only
        if val_mode == "regressor":
            for p in model_clone.encoder.parameters():
                p.requires_grad_(False)

        for batch in pbar:
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch
            del batch_x_mark, batch_y_mark

            # 1) Prediction BEFORE update (this is what should be scored)
            with torch.no_grad():
                x = batch_x.to(self.device).float()
                x_rec, outputs_flat, other_loss = model_clone(x, is_training=False)

                pred_flat, true_flat = self._extract_pred_true(outputs_flat, batch_y)
                pred_loss = self.criterion(pred_flat, true_flat)
                losses.append(float(pred_loss.detach().cpu().item()))

            # 2) Online update after label reveal
            if val_mode != "none":
                for _ in range(int(self.cfg.online.val_n_inner)):
                    opt_clone.zero_grad(set_to_none=True)

                    x = batch_x.to(self.device).float()
                    x_rec_u, outputs_flat_u, other_loss_u = model_clone(x, is_training=False)
                    pred_flat_u, true_flat_u = self._extract_pred_true(outputs_flat_u, batch_y)

                    loss_u = self.criterion(pred_flat_u, true_flat_u) + self.criterion(x_rec_u, x) + other_loss_u
                    loss_u.backward()
                    opt_clone.step()
                    model_clone.store_grad()

            pbar.set_postfix(avg_pred=f"{np.mean(losses):.4f}")

        return float(np.mean(losses)) if losses else None

    def fit(self) -> Dict[str, Any]:
        early_stopping = EarlyStopping(patience=self.cfg.patience, min_delta=0.0, verbose=True)

        history = {
            "train_loss": [],
            "val_loss": [],
        }

        print("\n=== LSTD Offline Pretraining ===")
        print(f"Run ID       : {self.run_id}")
        print(f"Device       : {self.device}")
        print(f"CSV          : {os.path.join(self.cfg.root_path, self.cfg.data_path)}")
        print(f"Enc channels : {self.bundle.enc_in}")
        print(f"Train windows: {len(self.bundle.train_dataset)}")
        print(f"Val windows  : {len(self.bundle.val_dataset) if self.bundle.val_dataset is not None else 0}")
        print(f"Test windows : {len(self.bundle.test_dataset)}")
        print(f"Online test  : mode={self.cfg.online.mode}, n_inner={self.cfg.online.n_inner}")
        print()

        epoch_bar = tqdm(
            range(self.cfg.optim.train_epochs),
            desc="Epochs",
            total=self.cfg.optim.train_epochs,
            dynamic_ncols=True,
        )

        for epoch in epoch_bar:
            t0 = time.time()
            train_loss = self._train_one_epoch(epoch)

            if self.cfg.online.online_validate_during_fit:
                val_loss = self._validate_online_clone()
            else:
                val_loss = self._validate_static()

            dt = time.time() - t0

            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)

            if val_loss is None:
                monitor = train_loss
                epoch_msg = f"train={train_loss:.6f} | val=SKIPPED | {dt:.1f}s"
            else:
                monitor = val_loss
                epoch_msg = f"train={train_loss:.6f} | val={val_loss:.6f} | {dt:.1f}s"

            print(f"Epoch {epoch+1}/{self.cfg.optim.train_epochs} | {epoch_msg}")

            early_stopping.step(monitor, self.model, self.best_ckpt_path)

            epoch_bar.set_postfix(
                train=f"{train_loss:.4f}",
                val=("NA" if val_loss is None else f"{val_loss:.4f}"),
                best=("NA" if early_stopping.best_score is None else f"{early_stopping.best_score:.4f}"),
                es=f"{early_stopping.counter}/{self.cfg.patience}",
            )

            if early_stopping.early_stop:
                print("Early stopping triggered.")
                break

        # Load best checkpoint after pretraining
        self.model.load_state_dict(torch.load(self.best_ckpt_path, map_location=self.device))

        summary = {
            "run_id": self.run_id,
            "best_checkpoint": self.best_ckpt_path,
            "best_monitor": early_stopping.best_score,
            "device": str(self.device),
            "csv_path": os.path.join(self.cfg.root_path, self.cfg.data_path),
            "enc_in": self.bundle.enc_in,
            "target_index": self.bundle.target_index,
            "history": history,
            "window_config": {
                "seq_len": self.cfg.windows.seq_len,
                "label_len": self.cfg.windows.label_len,
                "pred_len": self.cfg.windows.pred_len,
                "features": self.cfg.windows.features,
                "target": self.cfg.windows.target,
                "scale": self.cfg.windows.scale,
                "timeenc": self.cfg.windows.timeenc,
                "freq": self.cfg.windows.freq,
                "train_ratio": self.cfg.windows.train_ratio,
                "val_ratio": self.cfg.windows.val_ratio,
                "test_ratio": self.cfg.windows.test_ratio,
            },
            "model_config": {
                "mode": self.cfg.model.mode,
                "enc_in": self.cfg.model.enc_in,
                "seq_len": self.cfg.model.seq_len,
                "pred_len": self.cfg.model.pred_len,
                "hidden_dim": self.cfg.model.hidden_dim,
                "hidden_layers": self.cfg.model.hidden_layers,
                "depth": self.cfg.model.depth,
            },
            "online_config": {
                "mode": self.cfg.online.mode,
                "n_inner": self.cfg.online.n_inner,
                "require_batch_size_one": self.cfg.online.require_batch_size_one,
                "online_validate_during_fit": self.cfg.online.online_validate_during_fit,
                "val_mode": self.cfg.online.val_mode,
                "val_n_inner": self.cfg.online.val_n_inner,
            },
            "split_export_info": self.split_export_info,
        }

        save_json(os.path.join(self.out_dir, "train_summary.json"), summary)
        return summary

    # -----------------------------
    # Online replay (predict->label->update)
    # -----------------------------
    def _set_online_trainable_mode(self, mode: str) -> None:
        mode = mode.lower()
        if mode not in {"none", "full", "regressor"}:
            raise ValueError("online mode must be one of: none, full, regressor")

        # Reset all params trainable first
        for p in self.model.parameters():
            p.requires_grad_(True)

        # Repo-style regressor mode freezes encoder only
        if mode == "regressor":
            for p in self.model.encoder.parameters():
                p.requires_grad_(False)

    def _online_update_after_label(
        self,
        batch_x: torch.Tensor,
        batch_y: torch.Tensor,
        n_inner: int,
    ) -> None:
        """
        Model is expected to be in eval() mode (repo test-time behavior).
        We still compute grads and step optimizer for adaptation.
        """
        for _ in range(int(n_inner)):
            self.optimizer.zero_grad(set_to_none=True)

            if self.use_amp:
                with torch.cuda.amp.autocast():
                    out = self._forward_losses(batch_x, batch_y, is_training_forward=False)

                self.scaler.scale(out["total_loss"]).backward()

                if self.cfg.optim.grad_clip_norm is not None:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.optim.grad_clip_norm)

                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                out = self._forward_losses(batch_x, batch_y, is_training_forward=False)
                out["total_loss"].backward()

                if self.cfg.optim.grad_clip_norm is not None:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.cfg.optim.grad_clip_norm)

                self.optimizer.step()

            # Important for FSNet adaptive-memory mechanism
            self.model.store_grad()

    def _resolve_split_loader(self, split: str):
        s = split.lower()
        if s == "train":
            return self.bundle.train_dataset, self.bundle.train_loader
        if s == "val":
            if self.bundle.val_loader is None or self.bundle.val_dataset is None:
                raise RuntimeError("Validation split is unavailable (val_ratio may be 0).")
            return self.bundle.val_dataset, self.bundle.val_loader
        if s == "test":
            return self.bundle.test_dataset, self.bundle.test_loader
        raise ValueError("split must be one of: train, val, test")

    def run_offline(
        self,
        split: str = "test",
        load_best: bool = True,
        online_mode: Optional[str] = None,
        n_inner: Optional[int] = None,
    ) -> Dict[str, Any]:
        """
        Online replay evaluation:
          1) predict (no update)
          2) compute metric on that prediction
          3) reveal label and update model (if mode != 'none')
        This leaves the model in an ADAPTED state after the stream.
        """
        mode = (online_mode or self.cfg.online.mode).lower()
        if mode not in {"none", "full", "regressor"}:
            raise ValueError("online_mode must be one of: none, full, regressor")

        n_inner_ = int(self.cfg.online.n_inner if n_inner is None else n_inner)
        if n_inner_ < 1:
            raise ValueError("n_inner must be >= 1")

        dataset, loader = self._resolve_split_loader(split)

        if self.cfg.online.require_batch_size_one and getattr(loader, "batch_size", None) != 1:
            raise ValueError(
                f"Online replay expects batch_size=1 for strict sequential updates, "
                f"but got {loader.batch_size}. Set the relevant loader batch size to 1."
            )

        if load_best:
            self.model.load_state_dict(torch.load(self.best_ckpt_path, map_location=self.device))

        # Online replay follows repo test-time style: eval mode, but with gradient steps after label reveal
        self.model.eval()
        self._set_online_trainable_mode(mode)

        pred_batches = []
        true_batches = []
        rec_losses = []
        pred_losses = []
        other_losses = []

        t0 = time.time()

        pbar = tqdm(
            loader,
            total=len(loader),
            desc=f"Online-{split}[{mode}]",
            dynamic_ncols=True,
        )

        for batch in pbar:
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch
            del batch_x_mark, batch_y_mark

            # 1) Score the prediction BEFORE any online update
            with torch.no_grad():
                out = self._forward_losses(batch_x, batch_y, is_training_forward=False)

                pred_loss = float(out["pred_loss"].detach().cpu().item())
                rec_loss = float(out["rec_loss"].detach().cpu().item())
                other_loss = float(out["other_loss"].detach().cpu().item())

                pred_losses.append(pred_loss)
                rec_losses.append(rec_loss)
                other_losses.append(other_loss)

                B = out["pred_flat"].shape[0]
                eval_channels = out["pred_flat"].shape[1] // self.cfg.windows.pred_len

                pred_seq = out["pred_flat"].view(B, self.cfg.windows.pred_len, eval_channels).detach().cpu().numpy()
                true_seq = out["true_flat"].view(B, self.cfg.windows.pred_len, eval_channels).detach().cpu().numpy()

                pred_batches.append(pred_seq)
                true_batches.append(true_seq)

            # 2) Reveal label and adapt online
            if mode != "none":
                self._online_update_after_label(batch_x, batch_y, n_inner=n_inner_)

            # running metrics preview
            if pred_batches:
                preds_so_far = np.concatenate(pred_batches, axis=0)
                trues_so_far = np.concatenate(true_batches, axis=0)
                m = regression_metrics(preds_so_far, trues_so_far)
                pbar.set_postfix(
                    steps=f"{preds_so_far.shape[0]}",
                    mse=f"{m['mse']:.5f}",
                    mae=f"{m['mae']:.5f}",
                    pred=f"{np.mean(pred_losses):.5f}",
                )

        # Restore all params trainable for future calls
        for p in self.model.parameters():
            p.requires_grad_(True)

        if len(pred_batches) == 0:
            raise RuntimeError("No batches produced for online replay. Check dataset/window sizes.")

        preds = np.concatenate(pred_batches, axis=0)
        trues = np.concatenate(true_batches, axis=0)

        scaled_metrics = regression_metrics(preds, trues)

        unscaled_metrics = None
        inv = None
        if self.cfg.eval.compute_unscaled_metrics:
            inv = self._inverse_scale_for_metrics(preds, trues, dataset)
            if inv is not None:
                preds_u, trues_u = inv
                unscaled_metrics = regression_metrics(preds_u, trues_u)

        elapsed = time.time() - t0

        result = {
            "run_id": self.run_id,
            "checkpoint_loaded": (self.best_ckpt_path if load_best else None),
            "split": split,
            "evaluation_protocol": "online_replay_predict_then_update",
            "online_mode": mode,
            "n_inner": n_inner_,
            "num_windows": int(preds.shape[0]),
            "pred_shape": list(preds.shape),
            "true_shape": list(trues.shape),
            "avg_pred_loss": float(np.mean(pred_losses)) if pred_losses else None,
            "avg_recon_loss": float(np.mean(rec_losses)) if rec_losses else None,
            "avg_other_loss": float(np.mean(other_losses)) if other_losses else None,
            "scaled_metrics": scaled_metrics,
            "unscaled_metrics": unscaled_metrics,
            "elapsed_seconds": float(elapsed),
            "model_state_after_run": f"adapted_after_{split}_stream",
        }

        # Save arrays / summaries
        suffix = f"{split}_{mode}"
        if self.cfg.runtime.save_test_arrays:
            np.save(os.path.join(self.out_dir, f"{suffix}_preds.npy"), preds)
            np.save(os.path.join(self.out_dir, f"{suffix}_trues.npy"), trues)

            if inv is not None:
                preds_u, trues_u = inv
                np.save(os.path.join(self.out_dir, f"{suffix}_preds_unscaled.npy"), preds_u)
                np.save(os.path.join(self.out_dir, f"{suffix}_trues_unscaled.npy"), trues_u)

        adapted_ckpt_path = None
        if self.cfg.online.save_adapted_checkpoint:
            adapted_ckpt_path = os.path.join(self.ckpt_dir, f"checkpoint_{suffix}_adapted.pth")
            torch.save(self.model.state_dict(), adapted_ckpt_path)
            result["adapted_checkpoint"] = adapted_ckpt_path

        save_json(os.path.join(self.out_dir, f"{suffix}_summary.json"), result)

        print("\n=== Online Replay Results ===")
        print(f"Split      : {split}")
        print(f"Mode       : {mode}")
        print(f"n_inner    : {n_inner_}")
        print(f"Scaled     : MSE={scaled_metrics['mse']:.6f}, MAE={scaled_metrics['mae']:.6f}, RMSE={scaled_metrics['rmse']:.6f}")
        if unscaled_metrics is not None:
            print(f"Unscaled   : MSE={unscaled_metrics['mse']:.6f}, MAE={unscaled_metrics['mae']:.6f}, RMSE={unscaled_metrics['rmse']:.6f}")
        if adapted_ckpt_path is not None:
            print(f"Adapted ckpt saved: {adapted_ckpt_path}")
        print(f"Outputs saved to  : {self.out_dir}")

        return result

    def test(self, load_best: bool = True) -> Dict[str, Any]:
        """
        Default test behavior is ONLINE replay (paper-style).
        """
        return self.run_offline(split="test", load_best=load_best)

    @torch.no_grad()
    def test_static(self, load_best: bool = True) -> Dict[str, Any]:
        """
        Optional static evaluation (no online updates).
        Useful for ablation / debugging.
        """
        if load_best:
            self.model.load_state_dict(torch.load(self.best_ckpt_path, map_location=self.device))

        self.model.eval()

        pred_batches = []
        true_batches = []
        rec_losses = []
        pred_losses = []
        other_losses = []

        t0 = time.time()

        pbar = tqdm(
            self.bundle.test_loader,
            total=len(self.bundle.test_loader),
            desc="Static-Test",
            dynamic_ncols=True,
        )

        for batch in pbar:
            batch_x, batch_y, batch_x_mark, batch_y_mark = batch
            del batch_x_mark, batch_y_mark

            out = self._forward_losses(batch_x, batch_y, is_training_forward=False)

            pred_losses.append(float(out["pred_loss"].detach().cpu().item()))
            rec_losses.append(float(out["rec_loss"].detach().cpu().item()))
            other_losses.append(float(out["other_loss"].detach().cpu().item()))

            B = out["pred_flat"].shape[0]
            eval_channels = out["pred_flat"].shape[1] // self.cfg.windows.pred_len
            pred_seq = out["pred_flat"].view(B, self.cfg.windows.pred_len, eval_channels).detach().cpu().numpy()
            true_seq = out["true_flat"].view(B, self.cfg.windows.pred_len, eval_channels).detach().cpu().numpy()

            pred_batches.append(pred_seq)
            true_batches.append(true_seq)

            pbar.set_postfix(
                pred=f"{np.mean(pred_losses):.5f}",
                rec=f"{np.mean(rec_losses):.5f}",
            )

        if len(pred_batches) == 0:
            raise RuntimeError("No test batches produced. Check dataset size and window config.")

        preds = np.concatenate(pred_batches, axis=0)
        trues = np.concatenate(true_batches, axis=0)

        scaled_metrics = regression_metrics(preds, trues)

        unscaled_metrics = None
        if self.cfg.eval.compute_unscaled_metrics:
            inv = self._inverse_scale_for_metrics(preds, trues, self.bundle.test_dataset)
            if inv is not None:
                preds_u, trues_u = inv
                unscaled_metrics = regression_metrics(preds_u, trues_u)

        elapsed = time.time() - t0

        result = {
            "run_id": self.run_id,
            "checkpoint": self.best_ckpt_path,
            "evaluation_protocol": "static_no_online_updates",
            "num_test_windows": int(preds.shape[0]),
            "pred_shape": list(preds.shape),
            "true_shape": list(trues.shape),
            "avg_pred_loss": float(np.mean(pred_losses)) if pred_losses else None,
            "avg_recon_loss": float(np.mean(rec_losses)) if rec_losses else None,
            "avg_other_loss": float(np.mean(other_losses)) if other_losses else None,
            "scaled_metrics": scaled_metrics,
            "unscaled_metrics": unscaled_metrics,
            "elapsed_seconds": float(elapsed),
        }

        save_json(os.path.join(self.out_dir, "test_static_summary.json"), result)
        return result
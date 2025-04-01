from typing import Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch import optim

from ts_benchmark.baselines.self_impl.TFAD.TFAD_model import TFADModel
from ts_benchmark.baselines.self_impl.TFAD.TSDataset import TimeSeries
from ts_benchmark.baselines.self_impl.TFAD import transforms
from ts_benchmark.baselines.self_impl.TFAD.model.tfad_datamodule import reduce_labels
from ts_benchmark.baselines.time_series_library.utils.tools import (
    EarlyStopping,
    adjust_learning_rate,
)
from ts_benchmark.baselines.utils import (
    train_val_split,
    anomaly_detection_data_provider,
)

DEFAULT_TFAD_BASED_HYPER_PARAMS = {
    "anomaly_ratio": [0.1, 0.5, 1.0, 2, 3, 5.0, 10.0, 15, 20, 25],
    "batch_size": 256,
    "n_window": 192,
    "embedding_rep_dim": 120,
    "tcn_kernel_size": 7,
    "tcn_out_channels": 20,
    "tcn_layers": 8,
    "tcn_maxpool_out_channels": 8,
    "normalize_embedding": True,
    "lr": 0.0001,
    "num_epochs": 10,
    "k": 7,
    "patience": 3,
    "task_name": "anomaly_detection",
    "lradj": "type1",
    "hp_lamb": 1,
    "suspect_window_length": 1,
    "weight_fft_branch": 1,
    "rate_true_anomalies_used": 1,
    "ratio_injected_spikes": 0.01,
    "max_windows_unfold_batch": 256,
}


class TFADConfig:
    def __init__(self, **kwargs):
        for key, value in DEFAULT_TFAD_BASED_HYPER_PARAMS.items():
            setattr(self, key, value)

        for key, value in kwargs.items():
            setattr(self, key, value)


class TFAD:

    def __init__(self, **kwargs):
        super(TFAD, self).__init__()
        self.config = TFADConfig(**kwargs)
        self.scaler = StandardScaler()
        self.criterion = nn.BCELoss()
        self.model_name = "TFAD"

    @staticmethod
    def required_hyper_params() -> dict:
        """
        Return the hyperparameters required for the LOF model.

        :return: An empty dictionary indicating that the LOF model does not require additional hyperparameters.
        """
        return {}

    def detect_hyper_param_tune(self, train_data: pd.DataFrame):
        self.config.feats = train_data.shape[1]
        self.config.window_length = self.config.n_window
        ts_transform = transforms.LabelNoise(
            p_flip_1_to_0=1.0 - self.config.rate_true_anomalies_used
        )

        anom_transform = transforms.LocalOutlier(
            area_radius=2000,
            num_spikes=self.config.ratio_injected_spikes,
            spike_multiplier_range=(1.0, 4.0),
            direction_options=["increase"],
        )
        self.ts_transform = ts_transform + anom_transform

    def detect_validate(self, valid_data_loader, criterion, epoch):
        total_loss = []
        self.model.eval()

        for input, target in valid_data_loader:
            input = input.permute(0, 2, 1).float().to(self.device)
            data = input[:, :self.config.feats]
            label = input[:, self.config.feats]
            output = self.model(data)
            probs_anomaly = torch.sigmoid(output)
            target = reduce_labels(label, self.config.suspect_window_length).unsqueeze(1)
            loss = self.criterion(probs_anomaly, target)
            total_loss.append(loss.cpu().item())

        total_loss = np.mean(total_loss)
        self.model.train()
        return total_loss

    def detect_fit(self, train_data: pd.DataFrame, test_data: pd.DataFrame):
        """
        训练模型。

        :param train_data: 用于训练的时间序列数据。
        """
        setattr(self.config, "task_name", "anomaly_detection")
        self.detect_hyper_param_tune(train_data)
        Time_series_temp = TimeSeries(train_data.values, test_data.values)
        Time_series_temp = self.ts_transform.transform(Time_series_temp)

        train_data = pd.DataFrame(Time_series_temp.values)
        train_label = pd.DataFrame(Time_series_temp.labels)
        self.model = TFADModel(self.config)

        config = self.config
        train_data_value, valid_data = train_val_split(train_data, 0.8, None)
        train_data_label, valid_data_label = train_val_split(train_label, 0.8, None)
        self.scaler.fit(train_data_value.values)

        train_data_value = pd.DataFrame(
            np.concatenate((self.scaler.transform(train_data_value.values), train_data_label), axis=1),
        )

        valid_data = pd.DataFrame(
            np.concatenate((self.scaler.transform(valid_data.values), valid_data_label), axis=1),
        )

        self.valid_data_loader = anomaly_detection_data_provider(
            valid_data,
            batch_size=config.batch_size,
            win_size=config.window_length,
            step=1,
            mode="val",
        )

        self.train_data_loader = anomaly_detection_data_provider(
            train_data_value,
            batch_size=config.batch_size,
            win_size=config.window_length,
            step=1,
            mode="train",
        )

        # Define the loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=config.lr)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.early_stopping = EarlyStopping(patience=config.patience)
        self.model.to(self.device)
        total_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(f"Total trainable parameters: {total_params}")

        for epoch in range(config.num_epochs):
            self.model.train()
            for i, (input, target) in enumerate(self.train_data_loader):
                optimizer.zero_grad()
                input = input.permute(0, 2, 1).float().to(self.device)
                data = input[:, :self.config.feats]
                label = input[:, self.config.feats]
                output = self.model(data)
                probs_anomaly = torch.sigmoid(output)
                target = reduce_labels(label, self.config.suspect_window_length).unsqueeze(1)
                loss = self.criterion(probs_anomaly, target)
                loss.backward()
                optimizer.step()
            valid_loss = self.detect_validate(self.valid_data_loader, criterion, epoch)
            self.early_stopping(valid_loss, self.model)
            if self.early_stopping.early_stop:
                break

            adjust_learning_rate(optimizer, epoch + 1, config)

    def detect_score(self, test: pd.DataFrame) -> np.ndarray:
        test = pd.DataFrame(
            self.scaler.transform(test.values), columns=test.columns, index=test.index
        )
        self.model.load_state_dict(self.early_stopping.check_point)

        if self.model is None:
            raise ValueError("Model not trained. Call the fit() function first.")

        config = self.config

        self.thre_loader = anomaly_detection_data_provider(
            test,
            batch_size=config.batch_size,
            win_size=config.window_length,
            step=1,
            mode="thre",
        )

        self.model.to(self.device)
        self.model.eval()
        self.anomaly_criterion = nn.MSELoss(reduce=False)

        attens_energy = []

        for i, (input, batch_y) in enumerate(self.thre_loader):
            input = input.permute(0, 2, 1).float().to(self.device)
            score = self._detect_score(input).cpu().numpy()
            attens_energy.append(score)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        test_energy = np.nan_to_num(test_energy, nan=0.0)
        return test_energy, test_energy

    def detect_label(self, test: pd.DataFrame) -> np.ndarray:
        test = pd.DataFrame(
            self.scaler.transform(test.values), columns=test.columns, index=test.index
        )
        self.model.load_state_dict(self.early_stopping.check_point)

        if self.model is None:
            raise ValueError("Model not trained. Call the fit() function first.")

        config = self.config

        self.test_data_loader = anomaly_detection_data_provider(
            test,
            batch_size=config.batch_size,
            win_size=config.window_length,
            step=1,
            mode="test",
        )

        self.thre_loader = anomaly_detection_data_provider(
            test,
            batch_size=config.batch_size,
            win_size=config.window_length,
            step=1,
            mode="thre",
        )

        attens_energy = []

        self.model.to(self.device)
        self.model.eval()
        self.anomaly_criterion = nn.MSELoss(reduce=False)

        with torch.no_grad():
            for i, (input, batch_y) in enumerate(self.train_data_loader):
                input = input.permute(0, 2, 1).float().to(self.device)[:, :self.config.feats]
                score = self._detect_score(input).cpu().numpy()
                attens_energy.append(score)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) find the threshold
        attens_energy = []
        test_labels = []
        for i, (input, batch_y) in enumerate(self.test_data_loader):
            input = input.permute(0, 2, 1).float().to(self.device)
            score = self._detect_score(input).cpu().numpy()
            attens_energy.append(score)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        combined_energy = np.nan_to_num(combined_energy, nan=0.0)
        attens_energy = []
        test_labels = []
        for i, (input, batch_y) in enumerate(self.thre_loader):
            input = input.permute(0, 2, 1).float().to(self.device)
            score = self._detect_score(input).cpu().numpy()
            attens_energy.append(score)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        test_energy = np.nan_to_num(test_energy, nan=0.0)
        if not isinstance(self.config.anomaly_ratio, list):
            self.config.anomaly_ratio = [self.config.anomaly_ratio]

        preds = {}
        for ratio in self.config.anomaly_ratio:
            threshold = np.percentile(combined_energy, 100 - ratio)
            preds[ratio] = (test_energy > threshold).astype(int)

        return preds, test_energy

    def __repr__(self) -> str:
        """
        Returns a string representation of the model name.
        """
        return self.model_name

    def _detect_score(
            self,
            ts: torch.Tensor,
            threshold_prob_vote: float = 0.5,
            stride: Optional[int] = None,
    ) -> torch.Tensor:

        """Deploys the model over a tensor representing the time series

        Args:
            ts: Tensor with the time series. Shape (batch_size, ts_channels, time)

        Output
            pred: Tensor with the estimated probability of each timestep being anomalous. Shape (batch_size, time)
        """

        assert 0 <= threshold_prob_vote <= 1

        if stride is None:
            stride = self.config.suspect_window_length

        batch_size, ts_channels, T = ts.shape
        #         print("ts_channels is", ts_channels)
        #         print("ts.shape is", ts.shape)
        num_windows = int(1 + (T - self.config.window_length) / stride)

        # Define functions for folding and unfolding the time series
        unfold_layer = nn.Unfold(
            kernel_size=(ts_channels, self.config.window_length), stride=stride
        )
        fold_layer = nn.Fold(
            output_size=(1, T), kernel_size=(1, self.config.window_length), stride=stride
        )

        # Currently, only 4-D input tensors (batched image-like tensors) are supported
        # images = (batch, channels, height, width)
        # we adapt our time series creating a height channel of dimension 1, and then
        ts_windows = unfold_layer(ts.unsqueeze(1))
        #         print("ts_windows shape is", ts_windows.shape)

        assert ts_windows.shape == (
            batch_size,
            ts_channels * self.config.window_length,
            num_windows,
        )

        ts_windows = ts_windows.transpose(1, 2)
        ts_windows = ts_windows.reshape(
            batch_size, num_windows, ts_channels, self.config.window_length
        )

        with torch.no_grad():
            if self.config.max_windows_unfold_batch is None:
                logits_anomaly = self(ts_windows.flatten(start_dim=0, end_dim=1))
            else:
                # For very long time series, it is neccesary to process the windows in smaller chunks
                logits_anomaly = [
                    self.model(ts_windows_chunk)
                    for ts_windows_chunk in torch.split(
                        ts_windows.flatten(start_dim=0, end_dim=1),
                        self.config.max_windows_unfold_batch,
                        dim=0,
                    )
                ]
                logits_anomaly = torch.cat(logits_anomaly, dim=0)

        # Check model output shape: one label per (multivariate) window
        assert logits_anomaly.shape == (batch_size * num_windows, 1)

        # Repeat prediction for all timesteps in the suspect window, and reshape back before folding
        logits_anomaly = logits_anomaly.reshape(batch_size, num_windows, 1)
        logits_anomaly = logits_anomaly.repeat(1, 1, self.config.window_length)
        logits_anomaly[..., : -self.config.suspect_window_length] = np.nan
        logits_anomaly = logits_anomaly.transpose(1, 2)

        assert logits_anomaly.shape == (batch_size, self.config.window_length, num_windows)

        # Function to squeeze dimensions 1 and 2 after folding
        squeeze_fold = lambda x: x.squeeze(2).squeeze(1)

        ### Count the number of predictions per timestep ###
        # Indicates entries in logits_anomaly with a valid prediction
        id_suspect = torch.zeros_like(logits_anomaly)
        id_suspect[:, -self.config.suspect_window_length:] = 1.0
        num_pred = squeeze_fold(fold_layer(id_suspect))

        # Average of predicted probability of being anomalous for each timestep
        anomaly_probs = torch.sigmoid(logits_anomaly)
        # anomaly_probs_avg = squeeze_fold( fold_layer( anomaly_probs ) ) / num_pred
        anomaly_probs_nanto0 = torch.where(
            id_suspect == 1, anomaly_probs, torch.zeros_like(anomaly_probs)
        )
        anomaly_probs_avg = fold_layer(anomaly_probs_nanto0).squeeze(2).squeeze(1) / num_pred

        assert anomaly_probs_avg.shape == (batch_size, T)

        return anomaly_probs_avg

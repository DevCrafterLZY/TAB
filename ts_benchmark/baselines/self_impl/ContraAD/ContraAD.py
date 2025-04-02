import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from accelerate import Accelerator
from sklearn.preprocessing import StandardScaler
from torch import optim

from ts_benchmark.baselines.self_impl.ContraAD.model.PointAttention import PointHingeLoss, PatchAttention, cal_metric
from ts_benchmark.baselines.self_impl.ContraAD.utils.utils import normalize
from ts_benchmark.baselines.time_series_library.utils.tools import (
    EarlyStopping,
    adjust_learning_rate,
)
from ts_benchmark.baselines.utils import (
    train_val_split,
    anomaly_detection_data_provider,
)

accelerator = Accelerator()

DEFAULT_CONTRA_AD_BASED_HYPER_PARAMS = {
    "n_window": 50,
    "lr": 0.0001,
    "num_epochs": 5,
    "batch_size": 64,
    "patience": 3,
    "lradj": "type1",

    "anomaly_ratio": [0.1, 0.5, 1.0, 2, 3, 5.0, 10.0, 15, 20, 25],
}


class ContraADConfig:
    def __init__(self, **kwargs):
        for key, value in DEFAULT_CONTRA_AD_BASED_HYPER_PARAMS.items():
            setattr(self, key, value)

        for key, value in kwargs.items():
            setattr(self, key, value)


class ContraAD:

    def __init__(self, **kwargs):
        super(ContraAD, self).__init__()
        self.config = ContraADConfig(**kwargs)
        self.scaler = StandardScaler()
        self.model_name = "ContraAD"
        self.accelerator = Accelerator()

    @staticmethod
    def required_hyper_params() -> dict:
        """
        Return the hyperparameters required for the LOF model.

        :return: An empty dictionary indicating that the LOF model does not require additional hyperparameters.
        """
        return {}

    def detect_hyper_param_tune(self, train_data: pd.DataFrame):
        self.config.input_c = train_data.shape[1]
        self.config.output_c = train_data.shape[1]
        self.config.feats = train_data.shape[1]
        self.config.seq_len = self.config.n_window

    def detect_validate(self, valid_data_loader):
        total_loss = []
        self.model.eval()

        for input, _ in valid_data_loader:
            input_data = input.float().to(self.device)
            z_score = torch.sum(normalize(input_data.detach()), dim=-1)  # batch win_size
            intra = self.model(input_data)
            loss, _ = self.criterion(intra, z_score.detach())

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
        self.model = PatchAttention(self.config)
        self.criterion = nn.MSELoss()
        self.criterion = PointHingeLoss(
            mode='z_score_clamp',
            soft=True,
            soft_mode='min'
        )

        config = self.config
        train_data_value, valid_data = train_val_split(train_data, 0.8, None)
        self.scaler.fit(train_data_value.values)

        train_data_value = pd.DataFrame(
            self.scaler.transform(train_data_value.values),
            columns=train_data_value.columns,
            index=train_data_value.index,
        )

        valid_data = pd.DataFrame(
            self.scaler.transform(valid_data.values),
            columns=valid_data.columns,
            index=valid_data.index,
        )

        self.valid_data_loader = anomaly_detection_data_provider(
            valid_data,
            batch_size=config.batch_size,
            win_size=config.seq_len,
            step=1,
            mode="val",
        )

        self.train_data_loader = anomaly_detection_data_provider(
            train_data_value,
            batch_size=config.batch_size,
            win_size=config.seq_len,
            step=1,
            mode="train",
        )

        # Define the loss function and optimizer
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
                input_data = input.float().to(self.device)
                z_score = torch.sum(normalize(input_data.detach()), dim=-1)  # batch win_size
                intra = self.model(input_data)
                loss, _ = self.criterion(intra, z_score.detach())
                loss.backward()
                optimizer.step()
            valid_loss = self.detect_validate(self.valid_data_loader)
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
            win_size=config.seq_len,
            step=1,
            mode="thre",
        )

        self.model.to(self.device)
        self.model.eval()

        attns_energy_collect = []
        for i, (input, batch_y) in enumerate(self.thre_loader):
            input_data = input.float().to(self.device)
            intra = self.model(input_data)
            out = cal_metric(
                x=intra,
                z_score=None,
                mode='z_score_clamp',
                soft=True,
                soft_mode='min',
                model_mode='test'
            )
            metric = F.softmax(out, dim=1)
            attens_energy = accelerator.gather_for_metrics((metric))
            attns_energy_collect.extend([item.detach().cpu().numpy() for item in attens_energy])

        attens_energy = np.concatenate(attns_energy_collect, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)

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
            win_size=config.seq_len,
            step=1,
            mode="test",
        )

        self.thre_loader = anomaly_detection_data_provider(
            test,
            batch_size=config.batch_size,
            win_size=config.seq_len,
            step=1,
            mode="thre",
        )

        self.model.to(self.device)
        self.model.eval()

        attns_energy_collect = []
        for i, (input, batch_y) in enumerate(self.train_data_loader):
            input_data = input.float().to(self.device)
            intra = self.model(input_data)
            out = cal_metric(
                x=intra,
                z_score=None,
                mode='z_score_clamp',
                soft=True,
                soft_mode='min',
                model_mode='test'
            )
            metric = F.softmax(out, dim=1)
            attens_energy = accelerator.gather_for_metrics((metric))
            attns_energy_collect.extend([item.detach().cpu().numpy() for item in attens_energy])

        attens_energy = np.concatenate(attns_energy_collect, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        attns_energy_collect = []
        for i, (input, batch_y) in enumerate(self.thre_loader):
            input_data = input.float().to(self.device)
            intra = self.model(input_data)
            out = cal_metric(
                x=intra,
                z_score=None,
                mode='z_score_clamp',
                soft=True,
                soft_mode='min',
                model_mode='test'
            )
            metric = F.softmax(out, dim=1)
            attens_energy = accelerator.gather_for_metrics((metric))
            attns_energy_collect.extend([item.detach().cpu().numpy() for item in attens_energy])

        attens_energy = np.concatenate(attns_energy_collect, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)

        combined_energy = np.concatenate([train_energy, test_energy], axis=0)

        if not isinstance(self.config.anomaly_ratio, list):
            self.config.anomaly_ratio = [self.config.anomaly_ratio]

        preds = {}

        if not isinstance(self.config.anomaly_ratio, list):
            self.config.anomaly_ratio = [self.config.anomaly_ratio]

        for ratio in self.config.anomaly_ratio:
            threshold = np.percentile(combined_energy, 100 - ratio)
            preds[ratio] = (test_energy > threshold).astype(int)

        return preds, test_energy

    def __repr__(self) -> str:
        """
        Returns a string representation of the model name.
        """
        return self.model_name

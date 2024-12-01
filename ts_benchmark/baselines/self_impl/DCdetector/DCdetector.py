import copy
import time
from typing import Type, Dict

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import StandardScaler
from torch import optim
from ts_benchmark.baselines.self_impl.DCdetector.DCdetector_model import DCdetector_model
from ts_benchmark.baselines.utils import anomaly_detection_data_provider

from ts_benchmark.baselines.utils import train_val_split

DEFAULT_TRANSFORMER_BASED_HYPER_PARAMS = {
    "win_size": 100,
    "patch_size": [5],
    "lr": 0.0001,
    "n_heads": 1,
    "e_layers": 3,
    "d_model": 256,
    "rec_timeseries": True,
    "num_epochs": 10,
    "batch_size": 128,
    "patience": 5,
    "k": 3,
    "anomaly_ratio":  [0.1, 0.5, 1.0, 2, 3, 5.0, 10.0, 15, 20, 25],
}


def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


def adjust_learning_rate(optimizer, epoch, lr_):
    lr_adjust = {epoch: lr_ * (0.5 ** ((epoch - 1) // 1))}
    if epoch in lr_adjust.keys():
        lr = lr_adjust[epoch]
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr


class EarlyStopping:
    def __init__(self, patience=7, verbose=False, dataset_name="", delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.best_score2 = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.val_loss2_min = np.Inf
        self.delta = delta
        self.dataset = dataset_name

    def __call__(self, val_loss, val_loss2, model):
        score = -val_loss
        score2 = -val_loss2
        if self.best_score is None:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model)
        elif (
            score < self.best_score + self.delta
            or score2 < self.best_score2 + self.delta
        ):
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.best_score2 = score2
            self.save_checkpoint(val_loss, val_loss2, model)
            self.counter = 0

    def save_checkpoint(self, val_loss, val_loss2, model):
        self.val_loss_min = val_loss
        self.val_loss2_min = val_loss2
        self.check_point = copy.deepcopy(model.state_dict())


class TransformerConfig:
    def __init__(self, **kwargs):
        for key, value in DEFAULT_TRANSFORMER_BASED_HYPER_PARAMS.items():
            setattr(self, key, value)

        for key, value in kwargs.items():
            setattr(self, key, value)


class DCdetector:
    def __init__(self, **kwargs):
        super(DCdetector, self).__init__()
        self.config = TransformerConfig(**kwargs)
        self.scaler = StandardScaler()
        self.win_size = self.config.win_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @staticmethod
    def required_hyper_params() -> dict:
        """
        Return the hyperparameters required by model.

        :return: An empty dictionary indicating that model does not require additional hyperparameters.
        """
        return {}

    def __repr__(self) -> str:
        """
        Returns a string representation of the model name.
        """
        return self.model_name

    def vali(self, vali_loader):
        self.model.eval()
        loss_1 = []
        loss_2 = []
        for i, (input_data, _) in enumerate(vali_loader):
            input = input_data.float().to(self.device)
            series, prior = self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                series_loss += torch.mean(
                    my_kl_loss(
                        series[u],
                        (
                            prior[u]
                            / torch.unsqueeze(
                                torch.sum(prior[u], dim=-1), dim=-1
                            ).repeat(1, 1, 1, self.config.win_size)
                        ).detach(),
                    )
                ) + torch.mean(
                    my_kl_loss(
                        (
                            prior[u]
                            / torch.unsqueeze(
                                torch.sum(prior[u], dim=-1), dim=-1
                            ).repeat(1, 1, 1, self.config.win_size)
                        ).detach(),
                        series[u],
                    )
                )
                prior_loss += torch.mean(
                    my_kl_loss(
                        (
                            prior[u]
                            / torch.unsqueeze(
                                torch.sum(prior[u], dim=-1), dim=-1
                            ).repeat(1, 1, 1, self.config.win_size)
                        ),
                        series[u].detach(),
                    )
                ) + torch.mean(
                    my_kl_loss(
                        series[u].detach(),
                        (
                            prior[u]
                            / torch.unsqueeze(
                                torch.sum(prior[u], dim=-1), dim=-1
                            ).repeat(1, 1, 1, self.config.win_size)
                        ),
                    )
                )

            series_loss = series_loss / len(prior)
            prior_loss = prior_loss / len(prior)

            loss_1.append((prior_loss - series_loss).item())

        return np.average(loss_1), np.average(loss_2)

    def detect_fit(self, train_data: pd.DataFrame, test_data: pd.DataFrame):
        """
        训练模型。

        :param train_data: 用于训练的时间序列数据。
        """

        self.config.input_c = train_data.shape[1]
        self.config.output_c = train_data.shape[1]

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

        self.train_loader = anomaly_detection_data_provider(
            train_data_value,
            batch_size=self.config.batch_size,
            win_size=self.config.win_size,
            step=1,
            mode="train",
        )
        self.valid_loader = anomaly_detection_data_provider(
            valid_data,
            batch_size=self.config.batch_size,
            win_size=self.config.win_size,
            step=1,
            mode="val",
        )

        self.model = DCdetector_model(
            win_size=self.config.win_size,
            enc_in=self.config.input_c,
            c_out=self.config.output_c,
            n_heads=self.config.n_heads,
            d_model=self.config.d_model,
            e_layers=self.config.e_layers,
            patch_size=self.config.patch_size,
            channel=self.config.input_c,
        )
        self.model.to(self.device)
        total_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )

        print(f"Total trainable parameters: {total_params}")

        self.early_stopping = EarlyStopping(patience=self.config.patience, verbose=True)

        time_now = time.time()

        train_steps = len(self.train_loader)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.lr)

        for epoch in range(self.config.num_epochs):
            iter_count = 0

            epoch_time = time.time()
            self.model.train()
            for i, (input_data, labels) in enumerate(self.train_loader):
                self.optimizer.zero_grad()
                iter_count += 1
                input = input_data.float().to(self.device)
                series, prior = self.model(input)

                series_loss = 0.0
                prior_loss = 0.0

                for u in range(len(prior)):
                    series_loss += torch.mean(
                        my_kl_loss(
                            series[u],
                            (
                                prior[u]
                                / torch.unsqueeze(
                                    torch.sum(prior[u], dim=-1), dim=-1
                                ).repeat(1, 1, 1, self.config.win_size)
                            ).detach(),
                        )
                    ) + torch.mean(
                        my_kl_loss(
                            (
                                prior[u]
                                / torch.unsqueeze(
                                    torch.sum(prior[u], dim=-1), dim=-1
                                ).repeat(1, 1, 1, self.config.win_size)
                            ).detach(),
                            series[u],
                        )
                    )
                    prior_loss += torch.mean(
                        my_kl_loss(
                            (
                                prior[u]
                                / torch.unsqueeze(
                                    torch.sum(prior[u], dim=-1), dim=-1
                                ).repeat(1, 1, 1, self.config.win_size)
                            ),
                            series[u].detach(),
                        )
                    ) + torch.mean(
                        my_kl_loss(
                            series[u].detach(),
                            (
                                prior[u]
                                / torch.unsqueeze(
                                    torch.sum(prior[u], dim=-1), dim=-1
                                ).repeat(1, 1, 1, self.config.win_size)
                            ),
                        )
                    )

                series_loss = series_loss / len(prior)
                prior_loss = prior_loss / len(prior)

                loss = prior_loss - series_loss

                if (i + 1) % 100 == 0:
                    speed = (time.time() - time_now) / iter_count
                    left_time = speed * (
                        (self.config.num_epochs - epoch) * train_steps - i
                    )
                    print(
                        "\tspeed: {:.4f}s/iter; left time: {:.4f}s".format(
                            speed, left_time
                        )
                    )
                    iter_count = 0
                    time_now = time.time()

                loss.backward()
                self.optimizer.step()

            vali_loss1, vali_loss2 = self.vali(self.valid_loader)

            print(
                "Epoch: {0}, Cost time: {1:.3f}s ".format(
                    epoch + 1, time.time() - epoch_time
                )
            )

            self.early_stopping(vali_loss1, vali_loss2, self.model)
            if self.early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(self.optimizer, epoch + 1, self.config.lr)

    def detect_score(self, train: pd.DataFrame) -> np.ndarray:
        self.model.load_state_dict(self.early_stopping.check_point)

        thre_data = pd.DataFrame(
            self.scaler.transform(train.values),
            columns=train.columns,
            index=train.index,
        )

        self.thre_loader = anomaly_detection_data_provider(
            thre_data,
            batch_size=self.config.batch_size,
            win_size=self.config.win_size,
            step=1,
            mode="thre",
        )

        self.model.eval()
        temperature = 50

        test_labels = []
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.thre_loader):
            input = input_data.float().to(self.device)
            series, prior = self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = (
                        my_kl_loss(
                            series[u],
                            (
                                prior[u]
                                / torch.unsqueeze(
                                    torch.sum(prior[u], dim=-1), dim=-1
                                ).repeat(1, 1, 1, self.config.win_size)
                            ).detach(),
                        )
                        * temperature
                    )
                    prior_loss = (
                        my_kl_loss(
                            (
                                prior[u]
                                / torch.unsqueeze(
                                    torch.sum(prior[u], dim=-1), dim=-1
                                ).repeat(1, 1, 1, self.config.win_size)
                            ),
                            series[u].detach(),
                        )
                        * temperature
                    )
                else:
                    series_loss += (
                        my_kl_loss(
                            series[u],
                            (
                                prior[u]
                                / torch.unsqueeze(
                                    torch.sum(prior[u], dim=-1), dim=-1
                                ).repeat(1, 1, 1, self.config.win_size)
                            ).detach(),
                        )
                        * temperature
                    )
                    prior_loss += (
                        my_kl_loss(
                            (
                                prior[u]
                                / torch.unsqueeze(
                                    torch.sum(prior[u], dim=-1), dim=-1
                                ).repeat(1, 1, 1, self.config.win_size)
                            ),
                            series[u].detach(),
                        )
                        * temperature
                    )
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric.detach().cpu().numpy()
            attens_energy.append(cri)
            test_labels.append(labels)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)

        return test_energy, test_energy

    def detect_label(self, train: pd.DataFrame) -> np.ndarray:
        self.model.load_state_dict(self.early_stopping.check_point)

        thre_data = pd.DataFrame(
            self.scaler.transform(train.values),
            columns=train.columns,
            index=train.index,
        )

        self.thre_loader = anomaly_detection_data_provider(
            thre_data,
            batch_size=self.config.batch_size,
            win_size=self.config.win_size,
            step=1,
            mode="thre",
        )

        self.model.eval()
        temperature = 50

        # (1) stastic on the train set
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.train_loader):
            input = input_data.float().to(self.device)
            series, prior = self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = (
                        my_kl_loss(
                            series[u],
                            (
                                prior[u]
                                / torch.unsqueeze(
                                    torch.sum(prior[u], dim=-1), dim=-1
                                ).repeat(1, 1, 1, self.config.win_size)
                            ).detach(),
                        )
                        * temperature
                    )
                    prior_loss = (
                        my_kl_loss(
                            (
                                prior[u]
                                / torch.unsqueeze(
                                    torch.sum(prior[u], dim=-1), dim=-1
                                ).repeat(1, 1, 1, self.config.win_size)
                            ),
                            series[u].detach(),
                        )
                        * temperature
                    )
                else:
                    series_loss += (
                        my_kl_loss(
                            series[u],
                            (
                                prior[u]
                                / torch.unsqueeze(
                                    torch.sum(prior[u], dim=-1), dim=-1
                                ).repeat(1, 1, 1, self.config.win_size)
                            ).detach(),
                        )
                        * temperature
                    )
                    prior_loss += (
                        my_kl_loss(
                            (
                                prior[u]
                                / torch.unsqueeze(
                                    torch.sum(prior[u], dim=-1), dim=-1
                                ).repeat(1, 1, 1, self.config.win_size)
                            ),
                            series[u].detach(),
                        )
                        * temperature
                    )

            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) find the threshold
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.thre_loader):
            input = input_data.float().to(self.device)
            series, prior = self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = (
                        my_kl_loss(
                            series[u],
                            (
                                prior[u]
                                / torch.unsqueeze(
                                    torch.sum(prior[u], dim=-1), dim=-1
                                ).repeat(1, 1, 1, self.config.win_size)
                            ).detach(),
                        )
                        * temperature
                    )
                    prior_loss = (
                        my_kl_loss(
                            (
                                prior[u]
                                / torch.unsqueeze(
                                    torch.sum(prior[u], dim=-1), dim=-1
                                ).repeat(1, 1, 1, self.config.win_size)
                            ),
                            series[u].detach(),
                        )
                        * temperature
                    )
                else:
                    series_loss += (
                        my_kl_loss(
                            series[u],
                            (
                                prior[u]
                                / torch.unsqueeze(
                                    torch.sum(prior[u], dim=-1), dim=-1
                                ).repeat(1, 1, 1, self.config.win_size)
                            ).detach(),
                        )
                        * temperature
                    )
                    prior_loss += (
                        my_kl_loss(
                            (
                                prior[u]
                                / torch.unsqueeze(
                                    torch.sum(prior[u], dim=-1), dim=-1
                                ).repeat(1, 1, 1, self.config.win_size)
                            ),
                            series[u].detach(),
                        )
                        * temperature
                    )

            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)

        # (3) evaluation on the test set
        test_labels = []
        attens_energy = []
        for i, (input_data, labels) in enumerate(self.thre_loader):
            input = input_data.float().to(self.device)
            series, prior = self.model(input)
            series_loss = 0.0
            prior_loss = 0.0
            for u in range(len(prior)):
                if u == 0:
                    series_loss = (
                        my_kl_loss(
                            series[u],
                            (
                                prior[u]
                                / torch.unsqueeze(
                                    torch.sum(prior[u], dim=-1), dim=-1
                                ).repeat(1, 1, 1, self.config.win_size)
                            ).detach(),
                        )
                        * temperature
                    )
                    prior_loss = (
                        my_kl_loss(
                            (
                                prior[u]
                                / torch.unsqueeze(
                                    torch.sum(prior[u], dim=-1), dim=-1
                                ).repeat(1, 1, 1, self.config.win_size)
                            ),
                            series[u].detach(),
                        )
                        * temperature
                    )
                else:
                    series_loss += (
                        my_kl_loss(
                            series[u],
                            (
                                prior[u]
                                / torch.unsqueeze(
                                    torch.sum(prior[u], dim=-1), dim=-1
                                ).repeat(1, 1, 1, self.config.win_size)
                            ).detach(),
                        )
                        * temperature
                    )
                    prior_loss += (
                        my_kl_loss(
                            (
                                prior[u]
                                / torch.unsqueeze(
                                    torch.sum(prior[u], dim=-1), dim=-1
                                ).repeat(1, 1, 1, self.config.win_size)
                            ),
                            series[u].detach(),
                        )
                        * temperature
                    )
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric.detach().cpu().numpy()
            attens_energy.append(cri)
            test_labels.append(labels)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_labels = np.concatenate(test_labels, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        test_labels = np.array(test_labels)

        if not isinstance(self.config.anomaly_ratio, list):
            self.config.anomaly_ratio = [self.config.anomaly_ratio]

        preds = {}
        for ratio in self.config.anomaly_ratio:
            threshold = np.percentile(combined_energy, 100 - ratio)
            preds[ratio] = (test_energy > threshold).astype(int)
        return preds, test_energy


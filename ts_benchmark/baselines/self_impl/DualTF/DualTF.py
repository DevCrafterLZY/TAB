import copy
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import MinMaxScaler

from ts_benchmark.baselines.self_impl.DualTF.model.FrequencyTransformer import (
    FrequencyTransformer,
)
from ts_benchmark.baselines.self_impl.DualTF.model.TimeTransformer import (
    AnomalyTransformer,
)
from ts_benchmark.baselines.self_impl.DualTF.utils.utils import (
    generate_rolling_samples,
    EarlyStopping,
    adjust_learning_rate,
    get_grand_fre_dataloader,
)
from ts_benchmark.baselines.utils import (
    anomaly_detection_data_provider,
    train_val_split,
)

DEFAULT_DUAL_TF_BASED_HYPER_PARAMS = {
    "batch_size": 3,
    "anomaly_ratio": [0.1, 0.5, 1.0, 2, 3, 5.0, 10.0, 15, 20, 25],
    "seq_len": 100,
    "nest_len": 50,  # half seq_len
    "fre_anormly_ratio": 3,
    "lr": 1e-4,
    "num_epochs": 1,
    "k": 5,
    "step": 1,
}


def normalization(x):
    min_value = min(x)
    max_value = max(x)

    return np.array(
        list(map(lambda x: 1 * (x - min_value) / (max_value - min_value), x))
    )


class DualTfConfig:
    def __init__(self, **kwargs):
        for key, value in DEFAULT_DUAL_TF_BASED_HYPER_PARAMS.items():
            setattr(self, key, value)

        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def pred_len(self):
        return self.seq_len

    @property
    def learning_rate(self):
        return self.lr


def my_kl_loss(p, q):
    res = p * (torch.log(p + 0.0001) - torch.log(q + 0.0001))
    return torch.mean(torch.sum(res, dim=-1), dim=1)


class DualTF:
    def __init__(self, **kwargs):
        super(DualTF, self).__init__()
        self.config = DualTfConfig(**kwargs)
        self.time_scaler = MinMaxScaler()
        self.fre_scaler = MinMaxScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.MSELoss()
        self.seq_len = self.config.seq_len

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

    def detect_hyper_param_tune(self, train_data: pd.DataFrame):
        try:
            freq = pd.infer_freq(train_data.index)
        except Exception as ignore:
            freq = 'S'
        if freq == None:
            raise ValueError("Irregular time intervals")
        elif freq[0].lower() not in ["m", "w", "b", "d", "h", "t", "s"]:
            self.config.freq = "s"
        else:
            self.config.freq = freq[0].lower()

        column_num = train_data.shape[1]
        self.config.enc_in = column_num
        self.config.dec_in = column_num
        self.config.c_out = column_num
        self.config.label_len = 48

    def _time_detect_validate(self, vali_loader):
        self.time_model.eval()

        loss_1 = []
        loss_2 = []
        for input_data, _ in vali_loader:
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.time_model(input)
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
                            ).repeat(1, 1, 1, self.config.seq_len)
                        ).detach(),
                    )
                ) + torch.mean(
                    my_kl_loss(
                        (
                            prior[u]
                            / torch.unsqueeze(
                                torch.sum(prior[u], dim=-1), dim=-1
                            ).repeat(1, 1, 1, self.config.seq_len)
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
                            ).repeat(1, 1, 1, self.config.seq_len)
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
                            ).repeat(1, 1, 1, self.config.seq_len)
                        ),
                    )
                )
            series_loss = series_loss / len(prior)
            prior_loss = prior_loss / len(prior)

            rec_loss = self.criterion(output, input)
            loss_1.append((rec_loss - self.config.k * series_loss).item())
            loss_2.append((rec_loss + self.config.k * prior_loss).item())

        return np.average(loss_1), np.average(loss_2)

    def _fre_detect_validate(self, vali_loader):
        self.fre_model.eval()

        loss_1 = []
        loss_2 = []
        for i, input_data in enumerate(vali_loader):
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.fre_model(input)
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
                            ).repeat(
                                1,
                                1,
                                1,
                                (self.config.seq_len - self.config.nest_len + 1)
                                * (self.config.nest_len // 2),
                            )
                        ).detach(),
                    )
                ) + torch.mean(
                    my_kl_loss(
                        (
                            prior[u]
                            / torch.unsqueeze(
                                torch.sum(prior[u], dim=-1), dim=-1
                            ).repeat(
                                1,
                                1,
                                1,
                                (self.seq_len - self.config.nest_len + 1)
                                * (self.config.nest_len // 2),
                            )
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
                            ).repeat(
                                1,
                                1,
                                1,
                                (self.config.seq_len - self.config.nest_len + 1)
                                * (self.config.nest_len // 2),
                            )
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
                            ).repeat(
                                1,
                                1,
                                1,
                                (self.config.seq_len - self.config.nest_len + 1)
                                * (self.config.nest_len // 2),
                            )
                        ),
                    )
                )
            series_loss = series_loss / len(prior)
            prior_loss = prior_loss / len(prior)

            rec_loss = self.criterion(output, input)
            loss_1.append((rec_loss - self.config.k * series_loss).item())
            loss_2.append((rec_loss + self.config.k * prior_loss).item())
        return np.average(loss_1), np.average(loss_2)

    def _time_detect_fit(self, train_data: pd.DataFrame):
        self.time_model = AnomalyTransformer(
            win_size=self.config.seq_len,
            enc_in=self.config.c_in,
            c_out=self.config.c_out,
            e_layers=3,
        )
        self.optimizer = torch.optim.Adam(
            self.time_model.parameters(), lr=self.config.lr
        )
        if torch.cuda.is_available():
            self.time_model.cuda()

        self.time_early_stopping = EarlyStopping(
            patience=3,
            verbose=True,
        )

        train_loader, valid_loader = self.get_time_dataloader(train_data)

        train_steps = len(train_loader)
        time_now = time.time()
        for epoch in range(self.config.num_epochs):
            iter_count = 0
            loss1_list = []
            epoch_time = time.time()
            self.time_model.train()

            for i, (input, _) in enumerate(train_loader):
                self.optimizer.zero_grad()
                iter_count += 1
                input = input.float().to(self.device)

                output, series, prior, _ = self.time_model(input)

                # calculate Association discrepancy
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
                                ).repeat(1, 1, 1, self.config.seq_len)
                            ).detach(),
                        )
                    ) + torch.mean(
                        my_kl_loss(
                            (
                                prior[u]
                                / torch.unsqueeze(
                                    torch.sum(prior[u], dim=-1), dim=-1
                                ).repeat(1, 1, 1, self.config.seq_len)
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
                                ).repeat(1, 1, 1, self.config.seq_len)
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
                                ).repeat(1, 1, 1, self.config.seq_len)
                            ),
                        )
                    )
                series_loss = series_loss / len(prior)
                prior_loss = prior_loss / len(prior)

                rec_loss = self.criterion(output, input)

                loss1_list.append((rec_loss - self.config.k * series_loss).item())
                loss1 = rec_loss - self.config.k * series_loss
                loss2 = rec_loss + self.config.k * prior_loss
                if (i + 1) % 500 == 0:
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
                    print("loss 1: ", loss1, "loss 2: ", loss2)

                # Minimax strategy
                loss1.backward(retain_graph=True)
                loss2.backward()
                self.optimizer.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(loss1_list)

            vali_loss1, vali_loss2 = self._time_detect_validate(valid_loader)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(
                    epoch + 1, train_steps, train_loss, vali_loss1
                )
            )
            self.time_early_stopping(vali_loss1, vali_loss2, self.time_model)
            if self.time_early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(self.optimizer, epoch + 1, self.config.lr)

        print("Total Time: {}".format(time.time() - time_now))

    def _fre_detect_fit(self, train_data: pd.DataFrame):
        self.fre_model = FrequencyTransformer(
            win_size=(self.seq_len - self.config.nest_len + 1)
            * (self.config.nest_len // 2),
            enc_in=self.config.c_in,
            c_out=self.config.c_out,
            e_layers=3,
        )
        time_now = time.time()
        self.optimizer = torch.optim.Adam(
            self.fre_model.parameters(), lr=self.config.lr
        )
        if torch.cuda.is_available():
            self.fre_model.cuda()

        self.fre_train_loader, val_loader = self.get_fre_dataloader(train_data)
        train_steps = len(self.fre_train_loader)
        self.fre_early_stopping = EarlyStopping(patience=3, verbose=True)
        for epoch in range(self.config.num_epochs):
            iter_count = 0
            loss1_list = []

            epoch_time = time.time()
            self.fre_model.train()

            for i, input_data in enumerate(self.fre_train_loader):
                self.optimizer.zero_grad()
                iter_count += 1
                input = input_data.float().to(self.device)

                output, series, prior, _ = self.fre_model(input)
                loss = torch.mean(self.criterion(input, output), dim=-1)

                # calculate Association discrepancy
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
                                ).repeat(
                                    1,
                                    1,
                                    1,
                                    (self.config.seq_len - self.config.nest_len + 1)
                                    * (self.config.nest_len // 2),
                                )
                            ).detach(),
                        )
                    ) + torch.mean(
                        my_kl_loss(
                            (
                                prior[u]
                                / torch.unsqueeze(
                                    torch.sum(prior[u], dim=-1), dim=-1
                                ).repeat(
                                    1,
                                    1,
                                    1,
                                    (self.config.seq_len - self.config.nest_len + 1)
                                    * (self.config.nest_len // 2),
                                )
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
                                ).repeat(
                                    1,
                                    1,
                                    1,
                                    (self.config.seq_len - self.config.nest_len + 1)
                                    * (self.config.nest_len // 2),
                                )
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
                                ).repeat(
                                    1,
                                    1,
                                    1,
                                    (self.config.seq_len - self.config.nest_len + 1)
                                    * (self.config.nest_len // 2),
                                )
                            ),
                        )
                    )
                series_loss = series_loss / len(prior)
                prior_loss = prior_loss / len(prior)

                rec_loss = self.criterion(output, input)

                loss1_list.append((rec_loss - self.config.k * series_loss).item())
                loss1 = rec_loss - self.config.k * series_loss
                loss2 = rec_loss + self.config.k * prior_loss

                if (i + 1) % 500 == 0:
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
                    print("loss 1: ", loss1, "loss 2: ", loss2)
                # Minimax strategy
                loss1.backward(retain_graph=True)
                loss2.backward()
                self.optimizer.step()

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(loss1_list)
            vali_loss1, vali_loss2 = self._fre_detect_validate(val_loader)

            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f} ".format(
                    epoch + 1, train_steps, train_loss, vali_loss1
                )
            )
            self.fre_early_stopping(vali_loss1, vali_loss2, self.fre_model)
            if self.fre_early_stopping.early_stop:
                print("Early stopping")
                break
            adjust_learning_rate(self.optimizer, epoch + 1, self.config.lr)
        print("Total Time: {}".format(time.time() - time_now))

    def detect_fit(self, train_data: pd.DataFrame, test_data: pd.DataFrame):
        """
        Train the model.

        :param train_data: Time series data used for training.
        """
        self.detect_hyper_param_tune(train_data)
        setattr(self.config, "task_name", "anomaly_detection")
        self.config.c_in = train_data.shape[1]
        self.config.c_out = self.config.c_in
        self.config.nest_len = self.config.pred_len // 2

        self._fre_detect_fit(train_data)
        self._time_detect_fit(train_data)

    def detect_time_score(self, test: pd.DataFrame) -> np.ndarray:

        self.time_model.load_state_dict(self.time_early_stopping.check_point)
        self.time_model.to(self.device)
        self.time_model.eval()
        temperature = 50
        attens_energy = []
        test_seq = self.time_scaler.transform(test.values)
        test_loader = self.get_time_test_dataloader(test)

        for i, (input_data, target) in enumerate(test_loader):
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.time_model(input)

            loss = torch.mean(self.criterion(input, output), dim=-1)

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
                                ).repeat(1, 1, 1, self.seq_len)
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
                                ).repeat(1, 1, 1, self.seq_len)
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
                                ).repeat(1, 1, 1, self.seq_len)
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
                                ).repeat(1, 1, 1, self.seq_len)
                            ),
                            series[u].detach(),
                        )
                        * temperature
                    )
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy_old = np.concatenate(attens_energy, axis=0).reshape(-1)
        attens_energy_new = np.concatenate(attens_energy, axis=0)

        test_energy = np.array(attens_energy_old)
        test_energy_new = np.array(attens_energy_new)

        evaluation_arrays = []
        # For plotting evaluation results
        evaluation_array = np.zeros((7, len(test_seq)))
        predicted_normal_array = np.zeros((len(test_seq)))
        predicted_anomaly_array = np.zeros((len(test_seq)))
        rec_error_array = np.zeros((len(test_seq)))

        num_context = 0
        for ts in range(len(test_seq)):
            if ts < self.config.seq_len - 1:
                num_context = ts + 1
            elif ts >= self.config.seq_len - 1 and ts < len(test_seq) - self.config.seq_len + 1:
                num_context = self.config.seq_len
            elif ts >= len(test_seq) - self.config.seq_len + 1:
                num_context = len(test_seq) - ts
            evaluation_array[2][ts] = num_context

        pred_anomal_idx = []
        # Per each window
        print(f'Energy shape: {test_energy_new.shape}')
        print(f'Energy median: {np.median(test_energy_new)}')
        threshold = np.median(test_energy_new)
        for t in range(len(test_energy_new)):
            # For reconstruction error sum
            rec_error_array[t:t + self.config.seq_len] += test_energy_new[t]

            pred_normals = np.where(test_energy_new[t] <= threshold)[0]
            pred_anomalies = np.where(test_energy_new[t] > threshold)[0]

            # For Noraml
            for j in range(len(pred_normals)):
                predicted_normal_array[pred_normals[j] + t] += 1
            # For Abnormal
            for k in range(len(pred_anomalies)):
                predicted_anomaly_array[pred_anomalies[k] + t] += 1

        evaluation_array[0] = predicted_normal_array
        evaluation_array[1] = predicted_anomaly_array

        # Reconstruction Errors
        evaluation_array[6] = rec_error_array / evaluation_array[2]

        return evaluation_array[6]

    def detect_fre_score(self, test: pd.DataFrame) -> np.ndarray:
        # self.fre_model.load_state_dict(self.fre_early_stopping.check_point)
        self.fre_model.to(self.device)
        self.fre_model.eval()
        temperature = 50
        test_seq = self.fre_scaler.transform(test.values)
        test_loader = self.get_fre_test_dataloader(test)
        criterion = nn.MSELoss(reduction="none")

        attens_energy = []
        for i, input_data in enumerate(self.fre_train_loader):
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.fre_model(input)
            loss = torch.mean(criterion(input, output), dim=-1)
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
                                ).repeat(
                                    1,
                                    1,
                                    1,
                                    (self.config.seq_len - self.config.nest_len + 1)
                                    * (self.config.nest_len // 2),
                                )
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
                                ).repeat(
                                    1,
                                    1,
                                    1,
                                    (self.config.seq_len - self.config.nest_len + 1)
                                    * (self.config.nest_len // 2),
                                )
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
                                ).repeat(
                                    1,
                                    1,
                                    1,
                                    (self.config.seq_len - self.config.nest_len + 1)
                                    * (self.config.nest_len // 2),
                                )
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
                                ).repeat(
                                    1,
                                    1,
                                    1,
                                    (self.config.seq_len - self.config.nest_len + 1)
                                    * (self.config.nest_len // 2),
                                )
                            ),
                            series[u].detach(),
                        )
                        * temperature
                    )

            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) find the threshold
        attens_energy = []
        for i, input_data in enumerate(test_loader):
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.fre_model(input)

            loss = torch.mean(criterion(input, output), dim=-1)

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
                                ).repeat(
                                    1,
                                    1,
                                    1,
                                    (self.config.seq_len - self.config.nest_len + 1)
                                    * (self.config.nest_len // 2),
                                )
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
                                ).repeat(
                                    1,
                                    1,
                                    1,
                                    (self.config.seq_len - self.config.nest_len + 1)
                                    * (self.config.nest_len // 2),
                                )
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
                                ).repeat(
                                    1,
                                    1,
                                    1,
                                    (self.config.seq_len - self.config.nest_len + 1)
                                    * (self.config.nest_len // 2),
                                )
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
                                ).repeat(
                                    1,
                                    1,
                                    1,
                                    (self.config.seq_len - self.config.nest_len + 1)
                                    * (self.config.nest_len // 2),
                                )
                            ),
                            series[u].detach(),
                        )
                        * temperature
                    )
            # Metric
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)
            cri = metric * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
        thresh = np.percentile(combined_energy, 100 - self.config.fre_anormly_ratio)
        print("Threshold :", thresh)

        # (3) evaluation on the test set

        ############################ Alignment Module ############################
        sub_evaluation_arrays = []
        inference_time = time.time()
        ############################ Alignment Module ############################
        attens_energy = []
        for i, input_data in enumerate(test_loader):
            input = input_data.float().to(self.device)
            output, series, prior, _ = self.fre_model(input)

            loss = torch.mean(criterion(input, output), dim=-1)

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
                                ).repeat(
                                    1,
                                    1,
                                    1,
                                    (self.config.seq_len - self.config.nest_len + 1)
                                    * (self.config.nest_len // 2),
                                )
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
                                ).repeat(
                                    1,
                                    1,
                                    1,
                                    (self.config.seq_len - self.config.nest_len + 1)
                                    * (self.config.nest_len // 2),
                                )
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
                                ).repeat(
                                    1,
                                    1,
                                    1,
                                    (self.config.seq_len - self.config.nest_len + 1)
                                    * (self.config.nest_len // 2),
                                )
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
                                ).repeat(
                                    1,
                                    1,
                                    1,
                                    (self.config.seq_len - self.config.nest_len + 1)
                                    * (self.config.nest_len // 2),
                                )
                            ),
                            series[u].detach(),
                        )
                        * temperature
                    )
            metric = torch.softmax((-series_loss - prior_loss), dim=-1)

            cri = metric * loss
            cri = cri.detach().cpu().numpy()
            attens_energy.append(cri)



        attens_energy = np.concatenate(attens_energy, axis=0)
        test_energy = np.array(attens_energy)


        print(f"Test energy shape: {test_energy.shape}")
        print("### Inference time: {}".format(time.time() - inference_time))

        nested_test_energy = test_energy.reshape(
            test_energy.shape[0],
            self.config.seq_len - self.config.nest_len + 1,
            self.config.nest_len // 2,
        )
        ###################################### Alignment ##############################################

        for outer in range(len(nested_test_energy)):
            # as_frequency: (76, 25/2)
            # as_frequency = normalization(np.power(np.exp(np.linalg.norm(nested_test_energy[outer], axis=1)), 2))
            as_frequency = np.power(
                np.exp(np.linalg.norm(nested_test_energy[outer], axis=1)), 2
            )
            # as_frequency = np.exp(np.linalg.norm(nested_test_energy[outer], axis=1))
            # Ablation
            # as_frequency = np.linalg.norm(nested_test_energy[outer], axis=1)
            sub_evaluation_array = np.zeros((4, self.config.seq_len))
            rec_error_array = np.zeros((self.config.seq_len))

            num_context = 0
            for ts in range(self.config.seq_len):
                if ts < self.config.nest_len - 1:
                    num_context = ts + 1
                elif (
                    ts >= self.config.nest_len - 1
                    and ts < self.config.seq_len - self.config.nest_len + 1
                ):
                    num_context = self.config.nest_len
                elif ts >= self.config.seq_len - self.config.nest_len + 1:
                    num_context = self.config.seq_len - ts
                sub_evaluation_array[0][ts] = num_context  # SubSeq

            pred_anomal_idx = []
            # Per each window
            # nested shape: (76, 25, 1)
            for t in range(len(nested_test_energy[outer])):
                rec_error_array[t : t + self.config.nest_len] += as_frequency[t]

            sub_evaluation_array[1] = (
                rec_error_array / sub_evaluation_array[0]
            )  # exponential average (reconstruction error)

            # Predicted Anomaly Percentage
            for s in range(self.config.seq_len):
                # Predicted Anomaly (Binary)
                if sub_evaluation_array[1][s] > thresh:
                    sub_evaluation_array[2][s] = 1  # predicted label

            sub_evaluation_arrays.append(sub_evaluation_array)
        ###################################### Alignment ##############################################

        sub_evaluation_arrays = np.array(sub_evaluation_arrays)
        grand_evaluation_array = np.zeros((5, len(test_seq)))
        grand_rec_error_array = np.zeros((len(test_seq)))

        # Grand window array (301, 7, 100)
        for outer_win in range(len(sub_evaluation_arrays)):
            grand_evaluation_array[0][
                outer_win : outer_win + self.config.seq_len
            ] += sub_evaluation_arrays[outer_win][
                0
            ]  # sub-seq

            # For reconstruction error sum
            grand_rec_error_array[
                outer_win : outer_win + self.config.seq_len
            ] += sub_evaluation_arrays[outer_win][1]

        grand_context = 0
        # (400)
        for timestamp in range(len(test_seq)):
            if timestamp < self.config.seq_len - 1:
                grand_context = timestamp + 1
            elif (
                timestamp >= self.config.seq_len - 1
                and timestamp < len(test_seq) - self.config.seq_len + 1
            ):
                grand_context = self.config.seq_len
            elif timestamp >= len(test_seq) - self.config.seq_len + 1:
                grand_context = len(test_seq) - timestamp
            grand_evaluation_array[1][timestamp] = grand_context  # grand-seq

        grand_evaluation_array[2] = grand_rec_error_array / grand_evaluation_array[1]
        return grand_evaluation_array[2]

    def detect_score(self, test: pd.DataFrame):
        time_score = self.detect_time_score(test)
        fre_score = self.detect_fre_score(test)
        time_score = normalization(time_score)
        fre_score = normalization(fre_score)
        score = time_score + fre_score
        return score, score

    def detect_label(self, test: pd.DataFrame) -> np.ndarray:
        score_test, _ = self.detect_score(test)
        score_train, _ = self.detect_score(self.train_data)

        combined_energy = np.concatenate([score_train, score_test], axis=0)
        if not isinstance(self.config.anomaly_ratio, list):
            self.config.anomaly_ratio = [self.config.anomaly_ratio]

        preds = {}
        for ratio in self.config.anomaly_ratio:
            threshold = np.percentile(combined_energy, 100 - ratio)
            preds[ratio] = (score_test > threshold).astype(int)
        return preds, score_test

    def get_fre_dataloader(self, train_data: pd.DataFrame):
        train_data, valid_data = train_val_split(train_data, 0.8, None)
        self.train_data = train_data
        self.fre_scaler.fit(train_data.values)

        train_data_value = self.fre_scaler.transform(train_data.values)
        valid_data_value = self.fre_scaler.transform(valid_data.values)
        train_transform_data = generate_rolling_samples(
            train_data_value, self.config.seq_len
        )
        valid_transform_data = generate_rolling_samples(
            valid_data_value, self.config.seq_len
        )
        train_loader = get_grand_fre_dataloader(
            train_transform_data,
            self.config.nest_len,
            self.config.step,
            self.config.batch_size,
        )
        valid_loader = get_grand_fre_dataloader(
            valid_transform_data,
            self.config.nest_len,
            self.config.step,
            self.config.batch_size,
        )
        return train_loader, valid_loader

    def get_fre_test_dataloader(self, test_data: pd.DataFrame):
        test_data_value = self.fre_scaler.transform(test_data.values)
        test_transform_data = generate_rolling_samples(test_data_value, self.seq_len)
        test_loader = get_grand_fre_dataloader(
            test_transform_data,
            self.config.nest_len,
            self.config.step,
            self.config.batch_size,
        )
        return test_loader

    def get_time_dataloader(self, train_data: pd.DataFrame):
        train_data_value, valid_data = train_val_split(train_data, 0.8, None)
        self.time_scaler.fit(train_data_value.values)

        train_data_value = pd.DataFrame(
            self.time_scaler.transform(train_data_value.values),
            columns=train_data_value.columns,
            index=train_data_value.index,
        )

        valid_data = pd.DataFrame(
            self.time_scaler.transform(valid_data.values),
            columns=valid_data.columns,
            index=valid_data.index,
        )

        valid_data_loader = anomaly_detection_data_provider(
            valid_data,
            batch_size=self.config.batch_size,
            win_size=self.config.seq_len,
            step=1,
            mode="val",
        )

        train_data_loader = anomaly_detection_data_provider(
            train_data_value,
            batch_size=self.config.batch_size,
            win_size=self.config.seq_len,
            step=1,
            mode="train",
        )

        return train_data_loader, valid_data_loader

    def get_time_test_dataloader(self, test_data: pd.DataFrame):

        test_data_value = pd.DataFrame(
            self.time_scaler.transform(test_data.values),
        )

        test_data_loader = anomaly_detection_data_provider(
            test_data_value,
            batch_size=self.config.batch_size,
            win_size=self.config.seq_len,
            step=1,
            mode="test",
        )
        return test_data_loader

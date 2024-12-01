import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.optim import lr_scheduler

from ts_benchmark.baselines.self_impl.CATCH.models.CATCH_model import (
    CATCHModel,
)
from ts_benchmark.baselines.utils import anomaly_detection_data_provider
from ts_benchmark.baselines.utils import train_val_split
from .utils.fre_rec_loss import frequency_loss, frequency_criterion
from .utils.tools import EarlyStopping, adjust_learning_rate

DEFAULT_TRANSFORMER_BASED_HYPER_PARAMS = {
    "lr": 0.0001,
    "Mlr": 0.00001,
    "e_layers": 3,
    "n_heads": 2,
    "cf_dim": 64,
    "d_ff": 256,
    "d_model": 128,
    "head_dim": 64,
    "individual": 0,
    "dropout": 0.2,
    "head_dropout": 0.1,
    "auxi_loss": "MAE",
    "auxi_type": "complex",
    "auxi_mode": "fft",
    "auxi_lambda": 0.005,
    "score_lambda": 0.05,
    "regular_lambda": 0.5,
    "temperature": 0.07,
    "patch_stride": 8,
    "patch_size": 16,
    "inference_patch_stride": 1,
    "inference_patch_size": 32,
    "dc_lambda": 0.005,
    "module_first": True,
    "mask": False,
    "pretrained_model": None,
    "num_epochs": 3,
    "batch_size": 128,
    "patience": 3,
    "anomaly_ratio": [0.1, 0.5, 1.0, 2, 3, 5.0, 10.0, 15, 20, 25],
    "seq_len": 192,
    "pct_start": 0.3,
    "revin": 1,
    "affine": 0,
    "subtract_last": 0,
    "lradj": "type1",
}


class TransformerConfig:
    def __init__(self, **kwargs):
        for key, value in DEFAULT_TRANSFORMER_BASED_HYPER_PARAMS.items():
            setattr(self, key, value)

        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def pred_len(self):
        return self.seq_len

    @property
    def learning_rate(self):
        return self.lr


class CATCH:
    def __init__(self, **kwargs):
        super(CATCH, self).__init__()
        self.config = TransformerConfig(**kwargs)
        self.scaler = StandardScaler()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.criterion = nn.MSELoss()
        self.auxi_loss = frequency_loss(self.config)
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

    def detect_validate(self, valid_data_loader, criterion):
        config = self.config
        total_loss = []
        self.model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        with torch.no_grad():
            for input, _ in valid_data_loader:
                input = input.to(device)

                output, _, _ = self.model(input)

                output = output[:, :, :]

                output = output.detach().cpu()
                true = input.detach().cpu()

                loss = criterion(output, true).detach().cpu().numpy()
                total_loss.append(loss)

        total_loss = np.mean(total_loss)
        self.model.train()
        return total_loss

    def detect_fit(self, train_data: pd.DataFrame, test_data: pd.DataFrame):
        """
        Train the model.

        :param train_data: Time series data used for training.
        """

        self.detect_hyper_param_tune(train_data)
        setattr(self.config, "task_name", "anomaly_detection")
        self.config.c_in = train_data.shape[1]
        self.model = CATCHModel(self.config)
        self.model.to(self.device)

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

        total_params = sum(
            p.numel() for p in self.model.parameters() if p.requires_grad
        )
        print(f"Total trainable parameters: {total_params}")

        self.early_stopping = EarlyStopping(patience=self.config.patience, verbose=True)

        train_steps = len(self.train_data_loader)
        main_params = [param for name, param in self.model.named_parameters() if 'mask_generator' not in name]

        self.optimizer = torch.optim.Adam(main_params,
                                          lr=self.config.lr)
        self.optimizerM = torch.optim.Adam(self.model.mask_generator.parameters(), lr=self.config.Mlr)

        scheduler = lr_scheduler.OneCycleLR(
            optimizer=self.optimizer,
            steps_per_epoch=train_steps,
            pct_start=self.config.pct_start,
            epochs=self.config.num_epochs,
            max_lr=self.config.lr,
        )

        schedulerM = lr_scheduler.OneCycleLR(
            optimizer=self.optimizerM,
            steps_per_epoch=train_steps,
            pct_start=self.config.pct_start,
            epochs=self.config.num_epochs,
            max_lr=self.config.Mlr,
        )

        time_now = time.time()

        for epoch in range(self.config.num_epochs):
            iter_count = 0
            train_loss = []

            epoch_time = time.time()
            self.model.train()

            step = min(int(len(self.train_data_loader) / 10), 100)
            for i, (input, target) in enumerate(self.train_data_loader):
                iter_count += 1
                self.optimizer.zero_grad()

                input = input.float().to(self.device)

                output, output_complex, dcloss = self.model(input)

                output = output[:, :, :]

                rec_loss = self.criterion(output, input)

                norm_input = self.model.revin_layer(input, 'transform')
                auxi_loss = self.auxi_loss(output_complex, norm_input)

                loss = rec_loss + config.dc_lambda * dcloss + config.auxi_lambda * auxi_loss

                train_loss.append(loss.item())

                if (i + 1) % step == 0:
                    self.optimizerM.step()
                    self.optimizerM.zero_grad()

                if (i + 1) % 100 == 0:
                    print(
                        "\titers: {0}, epoch: {1} | training time loss: {2:.7f} | training fre loss: {3:.7f} | training dc loss: {4:.7f}".format(
                            i + 1, epoch + 1, rec_loss.item(), auxi_loss.item(), dcloss.item()
                        )
                    )
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

            print("Epoch: {} cost time: {}".format(epoch + 1, time.time() - epoch_time))
            train_loss = np.average(train_loss)
            valid_loss = self.detect_validate(self.valid_data_loader, self.criterion)
            print(
                "Epoch: {0}, Steps: {1} | Train Loss: {2:.7f} Vali Loss: {3:.7f}".format(
                    epoch + 1, train_steps, train_loss, valid_loss
                )
            )

            self.early_stopping(valid_loss, self.model)
            if self.early_stopping.early_stop:
                print("Early stopping")
                break

            adjust_learning_rate(self.optimizer, scheduler, epoch + 1, self.config)
            adjust_learning_rate(self.optimizerM, schedulerM, epoch + 1, self.config, printout=False)

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
        self.temp_anomaly_criterion = nn.MSELoss(reduce=False)
        self.freq_anomaly_criterion = frequency_criterion(config)
        attens_energy = []
        test_labels = []
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(self.thre_loader):
                batch_x = batch_x.float().to(self.device)
                # reconstruction
                outputs, _, _ = self.model(batch_x)
                # criterion
                temp_score = torch.mean(self.temp_anomaly_criterion(batch_x, outputs), dim=-1)
                freq_score = torch.mean(self.freq_anomaly_criterion(batch_x, outputs), dim=-1)
                score = (temp_score + config.score_lambda * freq_score).detach().cpu().numpy()
                attens_energy.append(score)
                test_labels.append(batch_y)

                print(
                    "\t testing time loss: {0} | \n testing fre loss: {1}".format(
                        temp_score.detach().cpu().numpy()[0,:5], freq_score.detach().cpu().numpy()[0,:5]
                    )
                )

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
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

        attens_energy = []

        self.model.to(self.device)
        self.model.eval()
        self.temp_anomaly_criterion = nn.MSELoss(reduce=False)
        self.freq_anomaly_criterion = frequency_criterion(config)
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(self.train_data_loader):
                batch_x = batch_x.float().to(self.device)
                # reconstruction
                outputs, _, _ = self.model(batch_x)
                # criterion
                temp_score = torch.mean(self.temp_anomaly_criterion(batch_x, outputs), dim=-1)
                freq_score = torch.mean(self.freq_anomaly_criterion(batch_x, outputs), dim=-1)

                score = (temp_score + config.score_lambda * freq_score).detach().cpu().numpy()
                attens_energy.append(score)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) find the threshold
        attens_energy = []
        test_labels = []
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(self.test_data_loader):
                batch_x = batch_x.float().to(self.device)
                # reconstruction
                outputs, _, _ = self.model(batch_x)
                # criterion
                temp_score = torch.mean(self.temp_anomaly_criterion(batch_x, outputs), dim=-1)
                freq_score = torch.mean(self.freq_anomaly_criterion(batch_x, outputs), dim=-1)
                score = (temp_score + config.score_lambda * freq_score).detach().cpu().numpy()
                attens_energy.append(score)
                test_labels.append(batch_y)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)

        attens_energy = []
        test_labels = []
        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(self.thre_loader):
                batch_x = batch_x.float().to(self.device)
                # reconstruction
                outputs, _, _ = self.model(batch_x)
                # criterion
                temp_score = torch.mean(self.temp_anomaly_criterion(batch_x, outputs), dim=-1)
                freq_score = torch.mean(self.freq_anomaly_criterion(batch_x, outputs), dim=-1)
                score = (temp_score + config.score_lambda * freq_score).detach().cpu().numpy()
                attens_energy.append(score)
                test_labels.append(batch_y)

                print(
                    "\t testing time loss: {0} | \n\t testing fre loss: {1}".format(
                        temp_score.detach().cpu().numpy()[0,:5], freq_score.detach().cpu().numpy()[0,:5]
                    )
                )

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)

        if not isinstance(self.config.anomaly_ratio, list):
            self.config.anomaly_ratio = [self.config.anomaly_ratio]

        preds = {}
        for ratio in self.config.anomaly_ratio:
            threshold = np.percentile(combined_energy, 100 - ratio)
            preds[ratio] = (test_energy > threshold).astype(int)

        return preds, test_energy
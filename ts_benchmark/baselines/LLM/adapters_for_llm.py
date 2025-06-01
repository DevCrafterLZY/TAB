import os
from typing import Type, Dict, Optional, Tuple

import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch import optim

from ts_benchmark.baselines.time_series_library.utils.tools import (
    EarlyStopping,
    adjust_learning_rate,
)
from ts_benchmark.baselines.utils_few import (
    forecasting_data_provider,
    train_val_split,
    anomaly_detection_data_provider,
    get_time_mark,
)
from ts_benchmark.models.model_base import ModelBase, BatchMaker
from ts_benchmark.utils.data_processing import split_before

DEFAULT_LLM_BASED_HYPER_PARAMS = {
    "num_samples": 100,
    "quantiles_num": 20,
    "ckpt_path":"",
    "dataset":"etth1",
    "patience": 3,
    "num_epochs": 20,
    "lradj": "type1",
    "freq": "H",
    "batch_size": 64,
    'label_len':96,
    "num_workers": 0,
    "freq": "h",
    "sampling_rate": 0.05,
    "sampling_strategy": "uniform",
    "sampling_basis": "sample",
    "is_train": 0,
    "get_train": 0,
    "lr": 0.0001,

    # "is_gpt": 1, # autotimes
    # "patch_size": 16,
    # "kernel_size": 25,
    # "pretrain": 1,
    # "seq_len": 96,
    # "horizon": 96,
    # "stride": 8,
    "gpt_layers": 3,
    "d_model": 768,
    "freeze":1,
    "use_multi_gpu": 0, # gpt4ts
    "local_rank": 0,
    "mix_embeds": 1,
    "mlp": 0,
    "mlp_hidden_layers": 0,
    "mlp_hidden_dim": 256,
    "mlp_activation": 'tanh',
    "dropout": 0.1,
    "token_len": 96,
    "use_amp": 0,

    "ln": 0, # s2lpllm
    "task_name": "long_term_forecast",
    "patch_size": 16,
    "stride": 8,
    "pretrained": 1,
    "pool_size": 1000,
    "prompt_length": 8,
    "trend_length": 96,
    "seasonal_length": 24,
    "prompt_init": 'text_prototype',

    "d_ff": 32, #timellm
    "llm_dim": 768,
    "patch_len": 16,
    "llm_model": "GPT2",
    "prompt_domain": 1,
    "n_heads": 8,
    "enc_in": 7,
    "llm_layers": 12,
    "content": "",

    "pretrain": 1,
    "max_token_num": 17, # unitime
    "mask_rate": 0.5,
    "max_backcast_len": 96,
    "max_forecast_len": 0,
    "ts_embed_dropout": 0.3,
    "lm_ft_type": "fpt",
    "instruct_path": "dataset/instruct.json",
    "lm_layer_num": 6,
    "dec_trans_layer_num": 2, 
    "ts_embed_dropout": 0.3,
    "dec_head_dropout": 0.1,

    "top_k" : 5, #LLMMixer
    "num_kernels": 6,
    # "d_model": 16,
    # "n_heads": 4,
    "e_layers": 2,
    "d_layers": 1,
    "moving_avg": 25,
    "factor": 1,
    "distil": 1,
    "channel_independence": 1,
    "decomp_method": "moving_avg",
    "down_sampling_layers": 3,
    "down_sampling_window": 2,
    "down_sampling_method": "avg",
    "use_future_temporal_feature": 0,
    "llm_path": "ts_benchmark/baselines/LLM/checkpoints/roberta-base",
    "tokenizer_path": "ts_benchmark/baselines/LLM/checkpoints/roberta-base",
    "embed": "timeF",

    "r": 8,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "word_embedding_path": "ts_benchmark/baselines/LLM/checkpoints/wte_pca_500.pt",

    "anomaly_ratio": [0.1, 0.5, 1.0, 2, 3, 5.0, 10.0, 15, 20, 25],
}

class LLMConfig:
    def __init__(self, **kwargs):
        for key, value in DEFAULT_LLM_BASED_HYPER_PARAMS.items():
            setattr(self, key, value)

        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def pred_len(self):
        return self.horizon


class LLMAdapter(ModelBase):
    def __init__(self, model_name, model_class, **kwargs):
        super(LLMAdapter, self).__init__()
        self.config = LLMConfig(**kwargs)
        self._model_name = model_name
        self.model_class = model_class
        self.scaler = StandardScaler()
        self.seq_len = self.config.seq_len


    @staticmethod
    def required_hyper_params() -> dict:
        """
        Return the hyperparameters required by model.

        :return: An empty dictionary indicating that model does not require additional hyperparameters.
        """
        return {}

    @property
    def model_name(self):
        """
        Returns the name of the model.
        """

        return self._model_name

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

    def padding_data_for_forecast(self, test):
        time_column_data = test.index
        data_colums = test.columns
        start = time_column_data[-1]

        date = pd.date_range(
            start=start, periods=self.config.horizon + 1, freq=self.config.freq.upper()
        )
        df = pd.DataFrame(columns=data_colums)

        df.iloc[: self.config.horizon + 1, :] = 0

        df["date"] = date
        df = df.set_index("date")
        new_df = df.iloc[1:]
        test = pd.concat([test, new_df])
        return test

    def _padding_time_stamp_mark(
        self, time_stamps_list: np.ndarray, padding_len: int
    ) -> np.ndarray:
        """
        Padding time stamp mark for prediction.

        :param time_stamps_list: A batch of time stamps.
        :param padding_len: The len of time stamp need to be padded.
        :return: The padded time stamp mark.
        """
        padding_time_stamp = []
        for time_stamps in time_stamps_list:
            start = time_stamps[-1]
            expand_time_stamp = pd.date_range(
                start=start,
                periods=padding_len + 1,
                freq=self.config.freq.upper(),
            )
            padding_time_stamp.append(expand_time_stamp.to_numpy()[-padding_len:])
        padding_time_stamp = np.stack(padding_time_stamp)
        whole_time_stamp = np.concatenate(
            (time_stamps_list, padding_time_stamp), axis=1
        )
        padding_mark = get_time_mark(whole_time_stamp, 1, self.config.freq)
        return padding_mark

    def validate(self, valid_data_loader, criterion):
        return None


    def forecast_fit(
        self, train_valid_data: pd.DataFrame, train_ratio_in_tv: float
    ) -> "ModelBase":
        return None


    def forecast(self, horizon: int, train: pd.DataFrame) -> np.ndarray:
        return None

    def batch_forecast(
        self, horizon: int, batch_maker: BatchMaker, **kwargs
    ) -> np.ndarray:
        return None

    def _perform_rolling_predictions(
        self,
        horizon: int,
        input_np: np.ndarray,
        all_mark: np.ndarray,
        device: torch.device,
    ) -> list:
        return None

    def _get_rolling_data(
        self,
        input_np: np.ndarray,
        output: Optional[np.ndarray],
        all_mark: np.ndarray,
        rolling_time: int,
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        return None


    def detect_validate(self, valid_data_loader, criterion):
        config = self.config
        total_loss = []
        self.model.eval()
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        for input, _ in valid_data_loader:
            input = input.to(device)

            output = self.model(input)

            output = output[:, -config.horizon :, :]

            if self.model_name == "UniTimeModel":
                output = output[:, :self.config.seq_len, :]

            output = output.detach().cpu()
            true = input.detach().cpu()

            loss = criterion(output, true).detach().cpu().numpy()
            total_loss.append(loss)

        total_loss = np.mean(total_loss)
        self.model.train()
        return total_loss

    def detect_fit(self, train_data: pd.DataFrame, test_data: pd.DataFrame):
        """
        训练模型。

        :param train_data: 用于训练的时间序列数据。
        """

        self.detect_hyper_param_tune(train_data)

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
            sampling_rate=config.sampling_rate
        )

        setattr(self.config, "task_name", "anomaly_detection")
        # x = train_data.columns.shape[0]
        setattr(self.config, "enc_in", train_data.columns.shape[0])
        self.model = self.model_class(self.config)

        total_params = sum(
            p.numel() for p in self.model.parameters()
        )
        print(f"Total parameters: {total_params}")
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Define the loss function and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=config.lr)

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
                input = input.float().to(self.device)

                output = self.model(input)

                # output = output[:, -config.horizon :, :]
                if self.model_name == "UniTimeModel":
                    output = output[:, :self.config.seq_len, :]
                loss = criterion(output, input)

                loss.backward()
                optimizer.step()
            valid_loss = self.detect_validate(self.valid_data_loader, criterion)
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
        self.anomaly_criterion = nn.MSELoss(reduce=False)

        attens_energy = []
        test_labels = []
        for i, (batch_x, batch_y) in enumerate(self.thre_loader):
            batch_x = batch_x.float().to(self.device)
            # reconstruction
            outputs = self.model(batch_x)
            if self.model_name == "UniTimeModel":
                outputs = outputs[:, :self.config.seq_len, :]
            # criterion
            score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
            score = score.detach().cpu().numpy()
            attens_energy.append(score)
            test_labels.append(batch_y)

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
        self.anomaly_criterion = nn.MSELoss(reduce=False)

        with torch.no_grad():
            for i, (batch_x, batch_y) in enumerate(self.train_data_loader):
                batch_x = batch_x.float().to(self.device)
                # reconstruction
                outputs = self.model(batch_x)
                if self.model_name == "UniTimeModel":
                    outputs = outputs[:, :self.config.seq_len, :]
                # criterion
                score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
                score = score.detach().cpu().numpy()
                attens_energy.append(score)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        train_energy = np.array(attens_energy)

        # (2) find the threshold
        attens_energy = []
        test_labels = []
        for i, (batch_x, batch_y) in enumerate(self.test_data_loader):
            batch_x = batch_x.float().to(self.device)
            # reconstruction
            outputs = self.model(batch_x)
            if self.model_name == "UniTimeModel":
                outputs = outputs[:, :self.config.seq_len, :]

            # criterion
            score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
            score = score.detach().cpu().numpy()
            attens_energy.append(score)
            test_labels.append(batch_y)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)
        combined_energy = np.concatenate([train_energy, test_energy], axis=0)

        attens_energy = []
        test_labels = []
        for i, (batch_x, batch_y) in enumerate(self.thre_loader):
            batch_x = batch_x.float().to(self.device)
            # reconstruction
            outputs = self.model(batch_x)
            if self.model_name == "UniTimeModel":
                outputs = outputs[:, :self.config.seq_len, :]
            # criterion
            score = torch.mean(self.anomaly_criterion(batch_x, outputs), dim=-1)
            score = score.detach().cpu().numpy()
            attens_energy.append(score)
            test_labels.append(batch_y)

        attens_energy = np.concatenate(attens_energy, axis=0).reshape(-1)
        test_energy = np.array(attens_energy)

        if not isinstance(self.config.anomaly_ratio, list):
            self.config.anomaly_ratio = [self.config.anomaly_ratio]

        preds = {}
        for ratio in self.config.anomaly_ratio:
            threshold = np.percentile(combined_energy, 100 - ratio)
            preds[ratio] = (test_energy > threshold).astype(int)

        return preds, test_energy

def generate_model_factory(
    model_name: str, model_class: type, required_args: dict
) -> Dict:
    """
    Generate model factory information for creating Transformer Adapters model adapters.

    :param model_name: Model name.
    :param model_class: Model class.
    :param required_args: The required parameters for model initialization.
    :return: A dictionary containing model factories and required parameters.
    """

    def model_factory(**kwargs) -> LLMAdapter:
        """
        Model factory, used to create TransformerAdapter model adapter objects.

        :param kwargs: Model initialization parameters.
        :return:  Model adapter object.
        """
        return LLMAdapter(model_name, model_class, **kwargs)

    return {
        "model_factory": model_factory,
        "required_hyper_params": required_args,
    }


def llm_adapter(model_info: Type[object]) -> object:
    if not isinstance(model_info, type):
        raise ValueError("the model_info does not exist")

    return generate_model_factory(
        model_name=model_info.__name__,
        model_class=model_info,
        required_args={
            "seq_len": "input_chunk_length",
            "horizon": "output_chunk_length",
            "norm": "norm",
        },
    )

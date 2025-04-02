import logging

import numpy as np
import pandas as pd

from ts_benchmark.baselines.self_impl.torsk.model.anomaly import sliding_score
from ts_benchmark.baselines.self_impl.torsk.model.data.numpy_dataset import NumpyImageDataset as ImageDataset
from ts_benchmark.baselines.self_impl.torsk.model.models.numpy_esn import NumpyESN as ESN
from ts_benchmark.baselines.self_impl.torsk.model.train_predict import train_predict_esn
from ts_benchmark.baselines.self_impl.torsk.model.data.window import ReverseWindowing

DEFAULT_TORSK_BASED_HYPER_PARAMS = {
    "windows_size": 10,
    "input_map_specs": [
        {"type": "pixels", "size": [10, 10], "input_scale": 3},
        {"type": "random_weights", "size": [100], "input_scale": 1},
        {"type": "conv", "size": [5, 5], "input_scale": 1, "mode": "same", "kernel_type": "gauss"}
    ],
    "reservoir_representation": "dense",
    "spectral_radius": 2.0,
    "density": 1e-1,

    "train_ratio": 0.5,
    "pred_ratio": 0.3,
    "transient_ratio": 0.2,
    "train_method": "pinv_svd",
    "imed_loss": True,
    "imed_sigma": 1.0,

    "dtype": "float64",
    "backend": "numpy",
    "timing_depth": 4,

    "tikhonov_beta": None,

    "scoring_small_window_size": 10,
    "scoring_large_window_size": 100,

    "debug": False,
    "anomaly_ratio": [0.1, 0.5, 1.0, 2, 3, 5.0, 10.0, 15, 20, 25],
}

logger = logging.getLogger(__name__)


class TORSKConfig:
    def __init__(self, **kwargs):
        for key, value in DEFAULT_TORSK_BASED_HYPER_PARAMS.items():
            setattr(self, key, value)

        for key, value in kwargs.items():
            setattr(self, key, value)

    @property
    def dict(self):
        return self.__dict__


class Torsk:

    def __init__(self, **kwargs):
        super(Torsk, self).__init__()
        self.config = TORSKConfig(**kwargs)
        self.model_name = "Torsk"

    @staticmethod
    def required_hyper_params() -> dict:
        """
        Return the hyperparameters required for the LOF model.

        :return: An empty dictionary indicating that the LOF model does not require additional hyperparameters.
        """
        return {}

    def detect_hyper_param_tune(self, train_data: pd.DataFrame):
        self.config.feats = train_data.shape[1]
        self.config.input_shape = (self.config.windows_size, self.config.feats)

    def detect_fit(self, train_data: pd.DataFrame, test_data: pd.DataFrame):
        """
        训练模型。

        :param train_data: 用于训练的时间序列数据。
        """
        self.train_data = train_data
        self.detect_hyper_param_tune(train_data)

    def detect_score(self, test: pd.DataFrame) -> np.ndarray:
        train_data = self.train_data.values.astype(np.float64)
        test_data = test.values.astype(np.float64)
        data = np.concatenate([train_data, test_data])

        padding_needed = data.shape[0] % self.config.input_shape[0] != 0
        if padding_needed:
            slices = data.shape[0] // self.config.input_shape[0]
            padding_size = (slices + 1) * self.config.input_shape[0] - data.shape[0]
            logger.info(f"Series not divisible by context window size, adding {padding_size} padding points")
            data = np.concatenate([data, np.full((padding_size, data.shape[1]), fill_value=0)], axis=0)
        data = data.reshape(
            (data.shape[0] // self.config.input_shape[0], self.config.input_shape[0], self.config.input_shape[1]))

        self.config.train_length = min(int(self.config.train_ratio * data.shape[0]), 50)
        self.config.pred_length = min(int(self.config.pred_ratio * data.shape[0]), 30)
        self.config.transient_length = min(int(self.config.transient_ratio * data.shape[0]), 20)

        steps = data.shape[0] - self.config.train_length - self.config.pred_length
        dataset = ImageDataset(images=data, scale_images=True, params=self.config)

        model = ESN(self.config)

        logger.info("Training + predicting ...")
        predictions, targets = train_predict_esn(model, dataset, steps=steps, step_length=1, step_start=0)

        logger.info("Calculating anomaly scores ...")
        logger.info(f"Prediction shape={predictions.shape}")
        logger.debug(f"Predictions targets shape={targets.shape}")

        errors = []
        for preds, labels in zip(predictions, targets):
            error = np.abs(labels - preds).mean(axis=-1).mean(axis=0)
            errors.append(error)
        logger.debug(f"{len(predictions)}x error shape: {error.shape}")
        scores, _, _, _ = sliding_score(np.array(errors),
                                        small_window=self.config.scoring_small_window_size,
                                        large_window=self.config.scoring_large_window_size)
        scores = np.concatenate([
            # the first batch of training samples has no predictions --> no scores
            np.full(shape=(self.config.train_length, self.config.windows_size),
                    fill_value=np.nan),
            scores
        ], axis=0)
        test_energy = 1 - scores.ravel()
        if padding_needed:
            # remove padding points
            logger.info("Removing padding from scores ...")
            test_energy = test_energy[:-padding_size]

        pred_size = self.config.pred_length
        context_window_size = self.config.windows_size
        size = pred_size * context_window_size + 1
        test_energy = ReverseWindowing(window_size=size).fit_transform(test_energy)[-len(test_data):]
        return test_energy, test_energy

    def detect_label(self, test: pd.DataFrame) -> np.ndarray:
        test_energy, _ = self.detect_score(test)

        preds = {}
        if not isinstance(self.config.anomaly_ratio, list):
            self.config.anomaly_ratio = [self.config.anomaly_ratio]
        for ratio in self.config.anomaly_ratio:
            threshold = np.percentile(test_energy, 100 - ratio)
            preds[ratio] = (test_energy > threshold).astype(int)

        return preds, test_energy


    def __repr__(self) -> str:
        """
        Returns a string representation of the model name.
        """
        return self.model_name

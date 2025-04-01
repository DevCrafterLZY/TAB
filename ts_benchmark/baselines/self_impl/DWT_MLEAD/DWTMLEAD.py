import numpy as np
import pandas as pd

from ts_benchmark.baselines.self_impl.DWT_MLEAD.model.model import DWT_MLEAD_model

DEFAULT_DWT_MLEAD_BASED_HYPER_PARAMS = {
    "start_level": 3,
    "anomaly_ratio": [0.1, 0.5, 1.0, 2, 3, 5.0, 10.0, 15, 20, 25],
    "quantile_epsilon": 0.01,
}


class DWT_MLEADConfig:
    def __init__(self, **kwargs):
        for key, value in DEFAULT_DWT_MLEAD_BASED_HYPER_PARAMS.items():
            setattr(self, key, value)

        for key, value in kwargs.items():
            setattr(self, key, value)


class DWT_MLEAD:

    def __init__(self, **kwargs):
        super(DWT_MLEAD, self).__init__()
        self.config = DWT_MLEADConfig(**kwargs)
        self.model_name = "DWT_MLEAD"

    @staticmethod
    def required_hyper_params() -> dict:
        """
        Return the hyperparameters required for the LOF model.

        :return: An empty dictionary indicating that the LOF model does not require additional hyperparameters.
        """
        return {}

    def detect_fit(self, train_data: pd.DataFrame, test_data: pd.DataFrame):
        """
        Train LOF models.

        :param X: Training data.
        :param y: Label data (optional).
        """
        self.train_data = train_data

    def detect_score(self, test: pd.DataFrame) -> np.ndarray:
        """
        Use the LOF model to calculate anomaly scores.

        :param X: The data of the score to be calculated.
        :return: Anomaly score array.
        """
        test_data = test.values.astype(np.float64).squeeze()
        test_energy = DWT_MLEAD_model(test_data, self.config.start_level, self.config.quantile_epsilon).detect()

        return test_energy, test_energy

    def detect_label(self, test: pd.DataFrame) -> np.ndarray:
        """
        Use LOF model for anomaly detection and generate labels.

        :param X: The data to be tested.
        :return: Anomaly label array.
        """
        test_data = test.values.astype(np.float64).squeeze()
        test_energy = DWT_MLEAD_model(test_data, self.config.start_level, self.config.quantile_epsilon).detect()

        preds = {}
        for ratio in self.config.anomaly_ratio:
            threshold = np.percentile(test_energy, 100 - ratio)
            preds[ratio] = (test_energy > threshold).astype(int)

        return preds, test_energy

    def __repr__(self) -> str:
        """
        Returns a string representation of the model name.
        """
        return self.model_name

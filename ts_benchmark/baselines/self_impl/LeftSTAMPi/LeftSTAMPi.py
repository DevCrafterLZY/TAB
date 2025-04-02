import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from stumpy import stumpi

DEFAULT_LEFT_STAMPI_BASED_HYPER_PARAMS = {
    "window_size": 50,
    "n_init_train": 100,
    "anomaly_ratio": [0.1, 0.5, 1.0, 2, 3, 5.0, 10.0, 15, 20, 25],
}


class LeftSTAMPiConfig:
    def __init__(self, **kwargs):
        for key, value in DEFAULT_LEFT_STAMPI_BASED_HYPER_PARAMS.items():
            setattr(self, key, value)

        for key, value in kwargs.items():
            setattr(self, key, value)


class LeftSTAMPi:

    def __init__(self, **kwargs):
        super(LeftSTAMPi, self).__init__()
        self.config = LeftSTAMPiConfig(**kwargs)
        self.model_name = "LeftSTAMPi"

    @staticmethod
    def required_hyper_params() -> dict:
        """
        Return the hyperparameters required for the LeftSTAMPi model.

        :return: An empty dictionary indicating that the LeftSTAMPi model does not require additional hyperparameters.
        """
        return {}

    def detect_fit(self, train_data: pd.DataFrame, test_data: pd.DataFrame):
        """
        Train LeftSTAMPi models.

        :param X: Training data.
        :param y: Label data (optional).
        """
        self.train_data = train_data

    def detect_score(self, test: pd.DataFrame) -> (np.ndarray, np.ndarray):
        """
        Use the LeftSTAMPi model to calculate anomaly scores.

        :param X: The data of the score to be calculated.
        :return: Anomaly score array.
        """
        train_data = self.train_data.values.T.astype(np.float64).squeeze()
        self.window_size = min(self.config.n_init_train, self.config.window_size)
        self.warm_up_len = self.config.n_init_train + self.config.window_size

        warm_up_data = train_data[-self.warm_up_len:]
        test_data = test.values.T.astype(np.float64).squeeze()
        stream = stumpi(warm_up_data, m=self.window_size, egress=False)
        for point in test_data:
            stream.update(point)

        test_energy = stream.left_P_[-len(test_data):]
        scaler = MinMaxScaler()
        test_energy = scaler.fit_transform(test_energy.reshape(-1, 1)).squeeze()
        return test_energy, test_energy

    def detect_label(self, test: pd.DataFrame) -> np.ndarray:
        """
        Use LOF model for anomaly detection and generate labels.

        :param X: The data to be tested.
        :return: Anomaly label array.
        """
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

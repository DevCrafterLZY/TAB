import numpy as np
import pandas as pd
import stumpy
from sklearn.preprocessing import MinMaxScaler

DEFAULT_MP_BASED_HYPER_PARAMS = {
    "window_size": 8,
    "anomaly_ratio": [0.1, 0.5, 1.0, 2, 3, 5.0, 10.0, 15, 20, 25],
}


class MPConfig:
    def __init__(self, **kwargs):
        for key, value in DEFAULT_MP_BASED_HYPER_PARAMS.items():
            setattr(self, key, value)

        for key, value in kwargs.items():
            setattr(self, key, value)


class MatrixProfile:

    def __init__(self, **kwargs):
        super(MatrixProfile, self).__init__()
        self.config = MPConfig(**kwargs)
        self.model_name = "MatrixProfile"

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
        data = test.values.T.astype(np.float64)
        matrix_profile = stumpy.mstump(data, m=self.config.window_size)[0]
        scaler = MinMaxScaler()
        scaler = scaler.fit(matrix_profile.T)
        matrix_profile = scaler.transform(matrix_profile.T)
        test_energy = np.sum(matrix_profile, axis=1)
        return test_energy, test_energy

    def detect_label(self, test: pd.DataFrame) -> np.ndarray:
        """
        Use LOF model for anomaly detection and generate labels.

        :param X: The data to be tested.
        :return: Anomaly label array.
        """
        train_data = self.train_data.values.T.astype(np.float64)
        test_data = test.values.T.astype(np.float64)
        data = np.concatenate((train_data, test_data), axis=1)
        matrix_profile = stumpy.mstump(data, m=self.config.window_size)[0]
        scaler = MinMaxScaler()
        scaler = scaler.fit(matrix_profile.T)
        matrix_profile = scaler.transform(matrix_profile.T)
        combined_energy = np.sum(matrix_profile, axis=1)

        test_matrix_profile = stumpy.mstump(test_data, m=self.config.window_size)[0]
        test_matrix_profile = scaler.transform(test_matrix_profile.T)

        test_energy = np.sum(test_matrix_profile, axis=1)

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

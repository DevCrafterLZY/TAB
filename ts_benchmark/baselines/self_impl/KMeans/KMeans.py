import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from ts_benchmark.baselines.self_impl.KMeans.model.model import KMeansAD

DEFAULT_KMEANS_BASED_HYPER_PARAMS = {
    "k": 20,
    "window_size": 50,
    "stride": 1,
    "anomaly_ratio": [0.1, 0.5, 1.0, 2, 3, 5.0, 10.0, 15, 20, 25],
}


class KMeansConfig:
    def __init__(self, **kwargs):
        for key, value in DEFAULT_KMEANS_BASED_HYPER_PARAMS.items():
            setattr(self, key, value)

        for key, value in kwargs.items():
            setattr(self, key, value)


class KMeans:

    def __init__(self, **kwargs):
        super(KMeans, self).__init__()
        self.config = KMeansConfig(**kwargs)
        self.scaler = StandardScaler()
        self.model_name = "KMeans"

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
        self.scaler.fit(train_data.values)
        self.model = KMeansAD(self.config)
        train_data = self.scaler.transform(train_data.values)
        self.preprocessed_train_data = self.model.preprocess_data(train_data)
        self.model.fit(self.preprocessed_train_data)

    def detect_score(self, test: pd.DataFrame) -> (np.ndarray, np.ndarray):
        """
        Use the LeftSTAMPi model to calculate anomaly scores.

        :param X: The data of the score to be calculated.
        :return: Anomaly score array.
        """
        test_data = self.scaler.transform(test.values)
        preprocessed_data = self.model.preprocess_data(test_data)
        test_energy = self.model.predict(preprocessed_data)

        return test_energy, test_energy

    def detect_label(self, test: pd.DataFrame) -> np.ndarray:
        """
        Use LOF model for anomaly detection and generate labels.

        :param X: The data to be tested.
        :return: Anomaly label array.
        """
        train_energy = self.model.predict(self.preprocessed_train_data)

        test_data = self.scaler.transform(test.values)
        preprocessed_data = self.model.preprocess_data(test_data)
        test_energy = self.model.predict(preprocessed_data)

        combined_energy = np.concatenate([train_energy, test_energy], axis=0)
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

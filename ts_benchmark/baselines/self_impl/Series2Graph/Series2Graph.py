import numpy as np
import pandas as pd

from ts_benchmark.baselines.self_impl.Series2Graph.model.model import Series2Graph_model

DEFAULT_SERIES2GRAPH_BASED_HYPER_PARAMS = {
    "window_size": 50,
    "query_window_size": 75,
    "rate": 30,

    "anomaly_ratio": [0.1, 0.5, 1.0, 2, 3, 5.0, 10.0, 15, 20, 25],
}


class SERIES2GRAPHConfig:
    def __init__(self, **kwargs):
        for key, value in DEFAULT_SERIES2GRAPH_BASED_HYPER_PARAMS.items():
            setattr(self, key, value)

        for key, value in kwargs.items():
            setattr(self, key, value)


class Series2Graph:

    def __init__(self, **kwargs):
        super(Series2Graph, self).__init__()
        self.config = SERIES2GRAPHConfig(**kwargs)
        self.model_name = "EIF"

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
        self.config.l = self.config.window_size
        self.config.ql = self.config.query_window_size
        self.config.latent = self.config.window_size // 3

    def detect_score(self, test: pd.DataFrame) -> np.ndarray:
        """
        Use the LOF model to calculate anomaly scores.

        :param X: The data of the score to be calculated.
        :return: Anomaly score array.
        """
        test = test.values.astype(np.float64).squeeze()
        s2g = Series2Graph_model(pattern_length=self.config.l, latent=self.config.latent, rate=self.config.rate)
        s2g.fit(test)
        s2g.score(query_length=self.config.ql)
        test_energy = s2g.decision_scores_.squeeze()
        return test_energy, test_energy

    def detect_label(self, test: pd.DataFrame) -> np.ndarray:
        """
        Use LOF model for anomaly detection and generate labels.

        :param X: The data to be tested.
        :return: Anomaly label array.
        """
        test = test.values.astype(np.float64).squeeze()
        s2g = Series2Graph_model(pattern_length=self.config.l, latent=self.config.latent, rate=self.config.rate)
        s2g.fit(test)
        s2g.score(query_length=self.config.ql)
        test_energy = s2g.decision_scores_.squeeze()

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

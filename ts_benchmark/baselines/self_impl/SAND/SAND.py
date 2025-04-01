import numpy as np
import pandas as pd

from ts_benchmark.baselines.self_impl.SAND.model import SAND_model

DEFAULT_SAND_BASED_HYPER_PARAMS = {
    "k": 6,
    "pattern_length": 50,
    "subsequence_length": 100,
    "init_length": 2000,
    "batch_size": 500,
    "alpha": 0.5,
    "anomaly_ratio": [0.1, 0.5, 1.0, 2, 3, 5.0, 10.0, 15, 20, 25],
}


class SANDConfig:
    def __init__(self, **kwargs):
        for key, value in DEFAULT_SAND_BASED_HYPER_PARAMS.items():
            setattr(self, key, value)

        for key, value in kwargs.items():
            setattr(self, key, value)


class SAND:

    def __init__(self, **kwargs):
        super(SAND, self).__init__()
        self.config = SANDConfig(**kwargs)
        self.model_name = "SAND"

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
        self.init_data = train_data.tail(self.config.init_length)
        self.model = SAND_model(
            pattern_length=self.config.pattern_length,
            subsequence_length=self.config.subsequence_length,
            k=self.config.k
        )

    def detect_score(self, test: pd.DataFrame) -> np.ndarray:
        """

        Use the LOF model to calculate anomaly scores.

        :param X: The data of the score to be calculated.
        :return: Anomaly score array.
        """
        init_data = self.init_data.values.squeeze().astype(np.float64)
        test_data = test.values.squeeze().astype(np.float64)
        data = np.concatenate([init_data, test_data])
        self.model.fit(X=data, init_length=self.config.init_length,
                       batch_size=self.config.batch_size, alpha=self.config.alpha)
        test_energy = self.model.decision_scores_[-len(test_data):]

        return test_energy, test_energy

    def detect_label(self, test: pd.DataFrame) -> np.ndarray:
        """
        Use LOF model for anomaly detection and generate labels.

        :param X: The data to be tested.
        :return: Anomaly label array.
        """
        init_data = self.init_data.values.squeeze().astype(np.float64)
        test_data = test.values.squeeze().astype(np.float64)
        data = np.concatenate([init_data, test_data])
        self.model.fit(X=data, init_length=self.config.init_length,
                       batch_size=self.config.batch_size, alpha=self.config.alpha)
        test_energy = self.model.decision_scores_[-len(test_data):]

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

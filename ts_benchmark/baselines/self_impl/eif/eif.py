import numpy as np
import pandas as pd

from ts_benchmark.baselines.self_impl.eif.models.model import iForest

DEFAULT_EIF_BASED_HYPER_PARAMS = {
    "n_trees": 200,
    "max_samples": None,
    "limit": None,
    "extension_level": None,
    "anomaly_ratio": [0.1, 0.5, 1.0, 2, 3, 5.0, 10.0, 15, 20, 25],
}


class EIFConfig:
    def __init__(self, **kwargs):
        for key, value in DEFAULT_EIF_BASED_HYPER_PARAMS.items():
            setattr(self, key, value)

        for key, value in kwargs.items():
            setattr(self, key, value)


class EIF:

    def __init__(self, **kwargs):
        super(EIF, self).__init__()
        self.config = EIFConfig(**kwargs)
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

    def detect_score(self, test: pd.DataFrame) -> np.ndarray:
        """
        Use the LOF model to calculate anomaly scores.

        :param X: The data of the score to be calculated.
        :return: Anomaly score array.
        """
        data = test.values.astype(np.float64)

        if self.config.max_samples:
            sample_size = int(self.config.max_samples * data.shape[0])
        else:
            sample_size = min(256, data.shape[0])
        limit = self.config.limit or int(np.ceil(np.log2(sample_size)))
        extension_level = self.config.extension_level or data.shape[1] - 1
        iforest = iForest(
            data,
            ntrees=self.config.n_trees,
            sample_size=sample_size,
            limit=limit,
            ExtensionLevel=extension_level,
        )
        test_energy = iforest.compute_paths(X_in=data)

        return test_energy, test_energy

    def detect_label(self, test: pd.DataFrame) -> np.ndarray:
        """
        Use LOF model for anomaly detection and generate labels.

        :param X: The data to be tested.
        :return: Anomaly label array.
        """
        data = test.values.astype(np.float64)

        if self.config.max_samples:
            sample_size = int(self.config.max_samples * data.shape[0])
        else:
            sample_size = min(256, data.shape[0])
        limit = self.config.limit or int(np.ceil(np.log2(sample_size)))
        extension_level = self.config.extension_level or data.shape[1] - 1
        iforest = iForest(
            data,
            ntrees=self.config.n_trees,
            sample_size=sample_size,
            limit=limit,
            ExtensionLevel=extension_level,
        )
        test_energy = iforest.compute_paths(X_in=data)

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

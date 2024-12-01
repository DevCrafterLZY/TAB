import os
import sys

current_file_path = os.path.abspath(__file__)

base_path = current_file_path[:current_file_path.rfind('ts_benchmark')]
sys.path.insert(0, base_path)
sys.path.insert(0, base_path + "ts_benchmark/baselines/tods/third_party")

import logging
import numpy as np
import pandas as pd

from ts_benchmark.baselines.tods.third_party.tods.sk_interface.detection_algorithm.IsolationForest_skinterface import (
    IsolationForestSKI,
)
from ts_benchmark.baselines.tods.third_party.tods.sk_interface.detection_algorithm.LSTMODetector_skinterface import (
    LSTMODetectorSKI,
)
from ts_benchmark.baselines.tods.third_party.tods.sk_interface.detection_algorithm.KNN_skinterface import (
    KNNSKI,
)
from ts_benchmark.baselines.tods.third_party.tods.sk_interface.detection_algorithm.AutoEncoder_skinterface import (
    AutoEncoderSKI,
)
from ts_benchmark.baselines.tods.third_party.tods.sk_interface.detection_algorithm.LOF_skinterface import (
    LOFSKI,
)
from ts_benchmark.baselines.tods.third_party.tods.sk_interface.detection_algorithm.OCSVM_skinterface import (
    OCSVMSKI,
)
from ts_benchmark.baselines.tods.third_party.tods.sk_interface.detection_algorithm.HBOS_skinterface import (
    HBOSSKI,
)
from ts_benchmark.baselines.tods.third_party.tods.sk_interface.detection_algorithm.LODA_skinterface import (
    LODASKI,
)
from ts_benchmark.baselines.tods.third_party.tods.sk_interface.detection_algorithm.PCAODetector_skinterface import (
    PCAODetectorSKI,
)

TODS_MODELS = [
    [IsolationForestSKI, {}],
    [LSTMODetectorSKI, {}],
    [KNNSKI, {}],
    [AutoEncoderSKI, {}],
    [LOFSKI, {}],
    [OCSVMSKI, {}],
    [HBOSSKI, {}],
    [LODASKI, {}],
    [PCAODetectorSKI, {}],
]


logger = logging.getLogger(__name__)


class TodsModelAdapter:
    """
    The Tods model adapter class is used to adapt models in the Tods framework to meet the requirements of prediction strategies.
    """

    def __init__(
        self,
        model_name: str,
        model_class: object,
        model_args: dict,
    ):
        """
        Initialize the Tods model adapter object.

        :param model_name: Model name.
        :param model_class: Tods model class.
        :param model_args: Model initialization parameters.
        """
        self.model = None
        self.model_class = model_class
        self.model_args = model_args
        self.model_name = model_name

    def detect_fit(self, series: pd.DataFrame, label: pd.DataFrame) -> object:
        """
        Fit a suitable Tods model on time series data.

        :param series: Time series data.
        :param label: Label data.
        :return: The fitted model object.
        """

        self.model = self.model_class(**self.model_args)
        X = series.values
        self.model.fit(X)

        return self.model

    def detect_score(self, train: pd.DataFrame) -> np.ndarray:
        """
        Calculate anomaly scores using an adapted Tods model.

        :param train: Training data used to calculate scores.
        :return: Anomaly score array.
        """
        X = train.values
        prediction_score = self.model.predict_score(X).reshape(-1)

        return prediction_score, prediction_score

    def detect_label(self, train: pd.DataFrame) -> np.ndarray:
        """
        Use an adapted Tods model for anomaly detection and generate labels.

        :param train: Training data used for anomaly detection.
        :return: Anomaly label array.
        """
        X = train.values
        prediction_labels = self.model.predict(X).reshape(-1)

        return prediction_labels, prediction_labels

    def __repr__(self):
        """
        Returns a string representation of the model name.
        """
        return self.model_name


def generate_model_factory(
    model_name: str,
    model_class: object,
    required_args: dict,
) -> object:
    """
    Generate model factory information for creating Tods model adapters.

    :param model_name: Model name.
    :param model_class: Tods model class.
    :param required_args: Required parameters for model initialization.
    :return: A dictionary containing the model factory and required parameters.
    """

    def model_factory(**kwargs) -> object:
        """
        Model factory, used to create Tods model adapter objects.
        :param kwargs: Model initialization parameters.
        :return: Tods model adapter object.
        """
        return TodsModelAdapter(
            model_name,
            model_class,
            kwargs,
        )

    return {"model_factory": model_factory, "required_hyper_params": required_args}


# Generate model factories for each model class and required parameters in TODS-MODELS and add them to global variables
for model_class, required_args in TODS_MODELS:
    globals()[f"{model_class.__name__.lower()}"] = generate_model_factory(
        model_class.__name__, model_class, required_args
    )

# TODO tods adapter
# def deep_tods_model_adapter(model_info: Type[object]) -> object:
#     """
#     适配深度 Tods 模型。

#     :param model_info: 要适配的深度 Tods 模型类。必须是一个类或类型对象。
#     :return: 生成的模型工厂，用于创建适配的 Tods 模型。
#     """
#     if not isinstance(model_info, type):
#         raise ValueError()

#     return generate_model_factory(
#         model_info.__name__,
#         model_info,
#         allow_fit_on_eval=False,
#     )

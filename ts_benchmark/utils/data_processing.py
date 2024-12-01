# -*- coding: utf-8 -*-

from typing import Tuple, Union

import numpy as np
import pandas as pd


def split_before(
    data: Union[pd.DataFrame, np.ndarray], index: int
) -> Union[Tuple[pd.DataFrame, pd.DataFrame], Tuple[np.ndarray, np.ndarray]]:
    """
    Split time series data into two parts at the specified index.

    :param data: Time series data to be segmented.
                 Can be a pandas DataFrame or a NumPy array.
    :param index: Split index position.
    :return: Tuple containing the first and second parts of the data.
    """
    if isinstance(data, pd.DataFrame):
        return data.iloc[:index, :], data.iloc[index:, :]
    elif isinstance(data, np.ndarray):
        return data[:index, :], data[index:, :]
    else:
        raise TypeError("Input data must be a pandas DataFrame or a NumPy array.")

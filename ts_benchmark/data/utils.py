import logging
import os
import pickle
from typing import Union, Any, Optional, Dict

import numpy as np
import pandas as pd
import scipy.sparse

logger = logging.getLogger(__name__)

FREQ_MAP = {
    "Y": "yearly",
    "A": "yearly",
    "A-DEC": "yearly",
    "A-JAN": "yearly",
    "A-FEB": "yearly",
    "A-MAR": "yearly",
    "A-APR": "yearly",
    "A-MAY": "yearly",
    "A-JUN": "yearly",
    "A-JUL": "yearly",
    "A-AUG": "yearly",
    "A-SEP": "yearly",
    "A-OCT": "yearly",
    "A-NOV": "yearly",
    "AS-DEC": "yearly",
    "AS-JAN": "yearly",
    "AS-FEB": "yearly",
    "AS-MAR": "yearly",
    "AS-APR": "yearly",
    "AS-MAY": "yearly",
    "AS-JUN": "yearly",
    "AS-JUL": "yearly",
    "AS-AUG": "yearly",
    "AS-SEP": "yearly",
    "AS-OCT": "yearly",
    "AS-NOV": "yearly",
    "BA-DEC": "yearly",
    "BA-JAN": "yearly",
    "BA-FEB": "yearly",
    "BA-MAR": "yearly",
    "BA-APR": "yearly",
    "BA-MAY": "yearly",
    "BA-JUN": "yearly",
    "BA-JUL": "yearly",
    "BA-AUG": "yearly",
    "BA-SEP": "yearly",
    "BA-OCT": "yearly",
    "BA-NOV": "yearly",
    "BAS-DEC": "yearly",
    "BAS-JAN": "yearly",
    "BAS-FEB": "yearly",
    "BAS-MAR": "yearly",
    "BAS-APR": "yearly",
    "BAS-MAY": "yearly",
    "BAS-JUN": "yearly",
    "BAS-JUL": "yearly",
    "BAS-AUG": "yearly",
    "BAS-SEP": "yearly",
    "BAS-OCT": "yearly",
    "BAS-NOV": "yearly",
    "Q": "quarterly",
    "Q-DEC": "quarterly",
    "Q-JAN": "quarterly",
    "Q-FEB": "quarterly",
    "Q-MAR": "quarterly",
    "Q-APR": "quarterly",
    "Q-MAY": "quarterly",
    "Q-JUN": "quarterly",
    "Q-JUL": "quarterly",
    "Q-AUG": "quarterly",
    "Q-SEP": "quarterly",
    "Q-OCT": "quarterly",
    "Q-NOV": "quarterly",
    "QS-DEC": "quarterly",
    "QS-JAN": "quarterly",
    "QS-FEB": "quarterly",
    "QS-MAR": "quarterly",
    "QS-APR": "quarterly",
    "QS-MAY": "quarterly",
    "QS-JUN": "quarterly",
    "QS-JUL": "quarterly",
    "QS-AUG": "quarterly",
    "QS-SEP": "quarterly",
    "QS-OCT": "quarterly",
    "QS-NOV": "quarterly",
    "BQ-DEC": "quarterly",
    "BQ-JAN": "quarterly",
    "BQ-FEB": "quarterly",
    "BQ-MAR": "quarterly",
    "BQ-APR": "quarterly",
    "BQ-MAY": "quarterly",
    "BQ-JUN": "quarterly",
    "BQ-JUL": "quarterly",
    "BQ-AUG": "quarterly",
    "BQ-SEP": "quarterly",
    "BQ-OCT": "quarterly",
    "BQ-NOV": "quarterly",
    "BQS-DEC": "quarterly",
    "BQS-JAN": "quarterly",
    "BQS-FEB": "quarterly",
    "BQS-MAR": "quarterly",
    "BQS-APR": "quarterly",
    "BQS-MAY": "quarterly",
    "BQS-JUN": "quarterly",
    "BQS-JUL": "quarterly",
    "BQS-AUG": "quarterly",
    "BQS-SEP": "quarterly",
    "BQS-OCT": "quarterly",
    "BQS-NOV": "quarterly",
    "M": "monthly",
    "BM": "monthly",
    "CBM": "monthly",
    "MS": "monthly",
    "BMS": "monthly",
    "CBMS": "monthly",
    "W": "weekly",
    "W-SUN": "weekly",
    "W-MON": "weekly",
    "W-TUE": "weekly",
    "W-WED": "weekly",
    "W-THU": "weekly",
    "W-FRI": "weekly",
    "W-SAT": "weekly",
    "D": "daily",
    "B": "daily",
    "C": "daily",
    "H": "hourly",
    "UNKNOWN": "other",
}

COVARIATES_LOAD_METHOD = {
    "adj.npz": scipy.sparse.load_npz,
}


def is_st(data: pd.DataFrame) -> bool:
    """
    Checks if data of the CSV file are in spatial-temporal format.

    :param data: The series data.
    :return: Are all values in 'cols' column are in spatial-temporal format.
    """
    return data.shape[1] == 4


def read_covariates(folder_path: str) -> Optional[Dict]:
    """
    Reads all covariates in the directory and returns a dictionary
    with filenames as keys and covariate data as values.

    :param folder_path: The covariates directory path.
    :return: A dictionary with filenames as keys and covariate data as values.
    """
    if not os.path.exists(folder_path) or not os.path.isdir(folder_path):
        return None

    covariates = {}
    for filename in os.listdir(folder_path):
        filepath = os.path.join(folder_path, filename)
        if os.path.isfile(filepath):
            try:
                # TODO the name of dictionary's key should be the type of covariates.
                covariates[filename] = get_covariate(filepath)
            except Exception as e:
                logger.warning("Error reading covariate %s: %s", filename, e)

    if not covariates:
        return None
    return covariates


def get_covariate(file_path: str) -> Any:
    """
    Reads a covariate file and returns its content.

    :param file_path: The path to the covariate file.
    :return: The content of the covariate file.
    """
    # TODO Covariate should be presented in a more fundamental and versatile form.
    # TODO deal different kind of data.
    covariate_type = os.path.basename(file_path)
    if covariate_type not in COVARIATES_LOAD_METHOD:
        raise ValueError(f"Unsupported covariate type: {covariate_type}")
    return COVARIATES_LOAD_METHOD[covariate_type](file_path)


def read_data(path: str, nrows=None) -> Union[pd.DataFrame, np.ndarray]:
    """
    Read the data file and return DataFrame.If the data is spatial-temporal format,

    return it as a numpy array; otherwise, return it as a Pandas DataFrame.

    :param path: The path to the data file.
    :return:  The content of the data file.
    """
    data = pd.read_csv(path)
    if is_st(data):
        return process_data_np(data, nrows)
    else:
        return process_data_df(data, nrows)


def process_data_df(data: pd.DataFrame, nrows=None) -> pd.DataFrame:
    """
    Read the data file and return DataFrame.

    According to the provided file path, read the data file and return the corresponding DataFrame.

    :param data: Data frame to read.
    :return:  The DataFrame of the content of the data file.
    """
    label_exists = "label" in data["cols"].values

    all_points = data.shape[0]

    columns = data.columns

    if columns[0] == "date":
        n_points = data.iloc[:, 2].value_counts().max()
    else:
        n_points = data.iloc[:, 1].value_counts().max()

    is_univariate = n_points == all_points

    n_cols = all_points // n_points
    df = pd.DataFrame()

    cols_name = data["cols"].unique()

    if columns[0] == "date" and not is_univariate:
        df["date"] = data.iloc[:n_points, 0]
        col_data = {
            cols_name[j]: data.iloc[j * n_points : (j + 1) * n_points, 1].tolist()
            for j in range(n_cols)
        }
        df = pd.concat([df, pd.DataFrame(col_data)], axis=1)
        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)

    elif columns[0] != "date" and not is_univariate:
        col_data = {
            cols_name[j]: data.iloc[j * n_points : (j + 1) * n_points, 0].tolist()
            for j in range(n_cols)
        }
        df = pd.concat([df, pd.DataFrame(col_data)], axis=1)

    elif columns[0] == "date" and is_univariate:
        df["date"] = data.iloc[:, 0]
        df[cols_name[0]] = data.iloc[:, 1]

        df["date"] = pd.to_datetime(df["date"])
        df.set_index("date", inplace=True)

    else:
        df[cols_name[0]] = data.iloc[:, 0]

    if label_exists:
        # Get the column name of the last column
        last_col_name = df.columns[-1]
        # Renaming the last column as "label"
        df.rename(columns={last_col_name: "label"}, inplace=True)

    if nrows is not None and isinstance(nrows, int) and df.shape[0] >= nrows:
        df = df.iloc[:nrows, :]

    return df


def process_data_np(df: pd.DataFrame, nrows=None) -> np.ndarray:
    """
    Convert spatial-temporal data from a DataFrame

    to a three-dimensional(time stamp,feature,sensor)  numpy array.

    :param df: Spatial-temporal data.
    :param nrows: Optional, number of rows to retain. Default is None, retaining all rows.
    :return: Three-dimensional(time stamp,feature,sensor) numpy array of the spatial temporal data.
    """
    pivot_df = df.pivot_table(index="date", columns=["id", "cols"], values="data")

    sensors = df["id"].unique()
    features = df["cols"].unique()
    pivot_df = pivot_df.reindex(
        columns=pd.MultiIndex.from_product([sensors, features]), fill_value=np.nan
    )

    data_np = pivot_df.to_numpy().reshape(len(pivot_df), len(sensors), len(features))
    data_np = np.transpose(data_np, (0, 2, 1))

    if nrows is not None:
        data_np = data_np[:nrows, :, :]

    return data_np


def load_series_info(file_path: str) -> dict:
    """
    get series info

    :param file_path: series file path
    :return: series info
    :rtype: dict
    """
    raw_data = pd.read_csv(file_path)
    if not is_st(raw_data):
        data = process_data_df(raw_data)
    else:
        data = process_data_np(raw_data)
    if_univariate = data.shape[1] == 1
    length = data.shape[0]
    time_stamp = pd.to_datetime(raw_data.iloc[:length, 0])
    freq = pd.infer_freq(time_stamp)
    freq = FREQ_MAP.get(freq, "other")
    file_name = os.path.basename(file_path)
    return {
        "file_name": file_name,
        "freq": freq,
        "if_univariate": if_univariate,
        "size": "user",
        "length": length,
        "trend": "",
        "seasonal": "",
        "stationary": "",
        "transition": "",
        "shifting": "",
        "correlation": "",
    }

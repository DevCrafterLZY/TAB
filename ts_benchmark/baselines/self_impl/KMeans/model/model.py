import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
from sklearn.base import BaseEstimator, OutlierMixin
from sklearn.cluster import KMeans


class KMeansAD(BaseEstimator, OutlierMixin):
    def __init__(self, configs):
        self.k = configs.k
        self.window_size = configs.window_size
        self.stride = configs.stride
        self.model = KMeans(n_clusters=configs.k)
        self.padding_length = 0

    def preprocess_data(self, X: np.ndarray) -> np.ndarray:
        flat_shape = (X.shape[0] - (self.window_size - 1), -1)  # in case we have a multivariate TS
        slides = sliding_window_view(X, window_shape=self.window_size, axis=0).reshape(flat_shape)[::self.stride, :]
        self.padding_length = X.shape[0] - (slides.shape[0] * self.stride + self.window_size - self.stride)
        return slides

    def _custom_reverse_windowing(self, scores: np.ndarray) -> np.ndarray:
        # compute begin and end indices of windows
        begins = np.array([i * self.stride for i in range(scores.shape[0])])
        ends = begins + self.window_size

        # prepare target array
        unwindowed_length = self.stride * (scores.shape[0] - 1) + self.window_size + self.padding_length
        mapped = np.full(unwindowed_length, fill_value=np.nan)

        # only iterate over window intersections
        indices = np.unique(np.r_[begins, ends])
        for i, j in zip(indices[:-1], indices[1:]):
            window_indices = np.flatnonzero((begins <= i) & (j - 1 < ends))
            # print(i, j, window_indices)
            mapped[i:j] = np.nanmean(scores[window_indices])

        # replace untouched indices with 0 (especially for the padding at the end)
        np.nan_to_num(mapped, copy=False)
        return mapped

    def fit(self, X: np.ndarray, y=None) -> 'KMeansAD':
        self.model.fit(X)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        clusters = self.model.predict(X)
        diffs = np.linalg.norm(X - self.model.cluster_centers_[clusters], axis=1)
        return self._custom_reverse_windowing(diffs)

import cupy as cp
from sklearn.base import BaseEstimator, TransformerMixin

class TimeWindowDecomposition(BaseEstimator, TransformerMixin):
    def __init__(self, overlap:float, num_samples_per_window:int):
        self.overlap = overlap
        self.num_samples_per_window = num_samples_per_window

    def fit(self, X, y=None):
        return self

    def transform(self, X:cp.ndarray):
        """
        Decomposes each trial into multiple time windows.
        Input:
        - x: A tensor of trials, with the shape (num_trials, num_channels, num_samples).
        - num_windows: The number of windows to decompose the signal into.
        Output:
        - A tensor of decomposed trials, with the shape (num_trials, num_windows, num_channels, num_samples_per_window).
        """

        num_trials, num_channels, num_samples = X.shape
        step = int(self.num_samples_per_window * (1 - self.overlap))
        num_windows = (num_samples - self.num_samples_per_window) // step + 1

        windows = cp.empty((num_trials, num_windows, num_channels, self.num_samples_per_window), dtype=X.dtype)
        for i in range(num_windows):
            start = i * step
            end = start + self.num_samples_per_window
            print(f'{start}ms -> {end}ms')
            windows[:, i, :, :] = X[:, :, start:end]
        return windows
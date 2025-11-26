import h5py
import cupy as cp
import numpy as np

from riemann_tangent_space_feature_extraction import RiemannTangentSpaceFeatureExtraction
from riemann_scm import RiemannSCM
from feature_fusion import FeatureFusion
from dimension_reduction import DimensionReduction
from rnn import RNNClassifier

from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline

from time_window_decomposition import TimeWindowDecomposition
from frequency_decomposition import FrequencyDecomposition
import time

from sklearn.base import BaseEstimator, TransformerMixin
class cupy_to_numpy(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()

    def transform(self, X):
        return X.get()
    
    def fit(self, X, y=None):
        return self
    
    def fit_transform(self, X, y=None):
        return self.fit(X, y).transform(X)

frequency_bands_per_person = [32, 4, 8, 2, 2, 2, 2, 16, 2, 16, 4, 16, 4]
frequency_bandwidth = 2
min_freq = 8
max_freq = 35
filter_order = 4
num_worker = 4
num_person = len(frequency_bands_per_person)
num_time_windows = 3
num_epochs = 50


with h5py.File('data/PERSONS_PREPROCESSED', 'r') as file:
    average_times = []

    for p in range(1, num_person+1):
        print(f'Person {p}')
        group_a = file[f'PERSON{p}A']
        group_b = file[f'PERSON{p}B']
        x_a = cp.array(group_a['trials'][:, :, 1000:])
        x_b = cp.array(group_b['trials'][:, :, 1000:])
        x = cp.concatenate([x_a, x_b], dtype=cp.float64)
        y_a = cp.array(group_a['labels'])
        y_b = cp.array(group_b['labels'])
        y = cp.concatenate([y_a, y_b], dtype=cp.int64)
        fs = group_a.attrs['fs']

        # Divide into train and test
        cv = StratifiedKFold(5, shuffle=True)

        times = []
        for fold_idx, (train_indices, test_indices) in enumerate(cv.split(x.get(), y.get())):
            
            x_train, x_test = x[train_indices], x[test_indices]
            y_train, y_test = y[train_indices].get(), y[test_indices].get()

            # Create the pipeline
            rnn_pipeline = make_pipeline(
                TimeWindowDecomposition(0.0, 1000),
                FrequencyDecomposition(frequency_bandwidth, fs, filter_order, min_freq, max_freq),
                cupy_to_numpy(),
                RiemannSCM(),
                RiemannTangentSpaceFeatureExtraction(max_iterations=200, num_jobs=num_worker),
                FeatureFusion((num_time_windows, -1)),
                DimensionReduction(2, True),
                RNNClassifier(num_epochs, 128, device='cuda'),
            )

            # Fit
            rnn_pipeline.fit(x_train, y_train)

            # Predict
            start = time.time()
            probs = rnn_pipeline.predict_proba(x_test)
            end = time.time()
            times.append(end-start)

        average_times.append(np.array(times).mean())
    
    print(average_times)


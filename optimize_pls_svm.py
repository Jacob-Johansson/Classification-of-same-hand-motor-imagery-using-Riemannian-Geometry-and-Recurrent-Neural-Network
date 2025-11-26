from sklearn.model_selection import RepeatedStratifiedKFold
import h5py
import torch
import cupy as cp
import numpy as np

from riemann_distance_feature_extraction import RiemannDistanceFeatureExtraction
from riemann_tangent_space_feature_extraction import RiemannTangentSpaceFeatureExtraction
from riemann_scm import RiemannSCM
from feature_fusion import FeatureFusion
from dimension_reduction import DimensionReduction
from mdrm import MDRM

from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from time_window_decomposition import TimeWindowDecomposition
from frequency_decomposition import FrequencyDecomposition
import time

num_splits = 5
num_repeats = 5
num_folds = num_splits * num_repeats

num_classes = 2

time_window_options = [250, 500, 1000, 1500, 3000]
frequency_band_options = [2, 4, 8, 16, 32]

time_overlap = 0.0
min_frequency = 8
max_frequency = 35
filter_order  = 5

num_workers = 4

# Parameters for storing the data
block = 'AB'
time_window_index = 2
frequency_band_index = -1

param_grid = {
     'svm__gamma': np.logspace(-2, 2, 5),
     'svm__C': np.logspace(-3, 3, 7),
     'pls__num_pls_components': [1, 2, 4, 8, 16]
}

cv = RepeatedStratifiedKFold(n_splits=num_splits, n_repeats=num_repeats, random_state=42)


with h5py.File(f'data/svm_optimized_block_{block}_time_{time_window_options[time_window_index]}_frequency_{frequency_band_options[frequency_band_index]}', 'w') as output:
        with h5py.File('data/PERSONS_PREPROCESSED', 'r') as file:
            output.attrs['num_person'] = 13
            output.attrs['num_folds'] = num_folds
            output.create_dataset('person_indices', data=[i for i in range(1, 14)])

            for p in range(1, 14):
               print(f'PERSON: {p}')

               group_a = file[f'PERSON{p}A']
               group_b = file[f'PERSON{p}B']
               x_a = cp.array(group_a['trials'][:, :, 1000:])
               x_b = cp.array(group_b['trials'][:, :, 1000:])
               x = cp.concatenate([x_a, x_b], dtype=cp.float64)
               y_a = cp.array(group_a['labels'])
               y_b = cp.array(group_b['labels'])
               y = cp.concatenate([y_a, y_b], dtype=cp.int64)
               fs = group_a.attrs['fs']

               print(x.shape, x.dtype, y.shape, y.dtype, fs)
               output_person_group = output.create_group(f'PERSON{p}')

               # Preprocess all signals before k-fold
               preprocessing = make_pipeline(TimeWindowDecomposition(0.0, time_window_options[time_window_index]), FrequencyDecomposition(frequency_band_options[frequency_band_index], fs, 5, min_frequency, max_frequency))
               start = time.time()
               xwb = preprocessing.fit_transform(x)
               end = time.time()
               print("Preprocessing Time", end - start)

               print("Preprocessed:", xwb.shape)

               feature_extraction_pipeline = make_pipeline(RiemannSCM(), RiemannTangentSpaceFeatureExtraction(max_iterations=200, num_jobs=num_workers), FeatureFusion((-1,)))
               start = time.time()
               features = feature_extraction_pipeline.fit_transform(xwb.get(), y.get())
               end = time.time()
               print("Feature extracted:", features.shape)
               print("Feature extraction Time:", end - start)

               from sklearn.svm import SVC
               svm_pipeline = Pipeline([
                    ('pls', DimensionReduction(2)),
                    ('svm', SVC(kernel='rbf', probability=True)),
               ])

               grid_search = GridSearchCV(svm_pipeline, param_grid, cv=cv, scoring='roc_auc', verbose=1)
               start = time.time()
               grid_search.fit(features, y.get())
               end = time.time()
               print("Grid search Time:", end - start)
               grid_search.best_params_

               # Store best parameters
               best_params_str = str(grid_search.best_params_)
               output_person_group.attrs['best_params'] = best_params_str  # Save as attribut

               # Store best cross-validation score
               best_score = grid_search.best_score_
               output_person_group.attrs['best_score'] = best_score  # Save as attribute

               # (Optional) Store train-test split or cross-validation indices
               cv_results = grid_search.cv_results_  # Contains detailed cross-validation results
               output_person_group.create_dataset('mean_test_scores', data=cv_results['mean_test_score'])
               output_person_group.create_dataset('std_test_scores', data=cv_results['std_test_score'])

               # Save all cv_results_ as datasets
               cv_group = output_person_group.create_group('cv_results')
               
               for key, value in cv_results.items():
                    if isinstance(value, np.ndarray):
                         # Save arrays directly
                         cv_group.create_dataset(key, data=value)
                    else:
                         # Save non-array data (like strings) as attributes
                         cv_group.attrs[key] = str(value)
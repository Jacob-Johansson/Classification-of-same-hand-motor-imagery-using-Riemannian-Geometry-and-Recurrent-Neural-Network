from sklearn.model_selection import RepeatedStratifiedKFold

import h5py
import cupy as cp
import numpy as np

from riemann_tangent_space_feature_extraction import RiemannTangentSpaceFeatureExtraction
from riemann_scm import RiemannSCM
from feature_fusion import FeatureFusion
from dimension_reduction import DimensionReduction

from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

from time_window_decomposition import TimeWindowDecomposition
from frequency_decomposition import FrequencyDecomposition
import time

def main():
    num_splits = 5
    num_repeats = 5
    num_folds = num_splits * num_repeats

    time_window_options = [250, 500, 1000, 1500, 3000]
    frequency_band_options = [2, 4, 8, 16, 32]

    min_freq = 8
    max_freq = 35
    filter_order = 4

    num_worker = 4
    time_window_index = 2
    freq_band_index = 0

    num_person = 13

    scoring = ['roc_auc', 'accuracy']

    param_grid = {
     'svm__gamma': np.logspace(-2, 2, 5),
     'svm__C': np.logspace(-3, 3, 7),
     'pls__num_pls_components': [1, 2, 4, 8, 16],
    }

    cv = RepeatedStratifiedKFold(n_splits=num_splits, n_repeats=num_repeats, random_state=42)

    with h5py.File(f'data/svm_grid_search_block_AB_time_{time_window_options[time_window_index]}_frequency_{frequency_band_options[freq_band_index]}_all', 'w') as output:
        with h5py.File('data/PERSONS_PREPROCESSED', 'r') as file:
            output.attrs['num_person'] = num_person
            output.attrs['num_folds'] = num_folds
            output.create_dataset('person_indices', data=[i for i in range(1, num_person+1)])

            for p in range(1, num_person+1):
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

                # Time window & Frequency preprocessing
                time_decomposer = TimeWindowDecomposition(0.0, time_window_options[time_window_index])
                freq_decomposer = FrequencyDecomposition(frequency_band_options[freq_band_index], fs, filter_order, min_freq, max_freq)
                
                x = time_decomposer.fit_transform(x, y)
                x = freq_decomposer.fit_transform(x, y)

                print('Preprocessed:', x.shape, x.dtype)
                # Transfer from GPU to CPU
                x = x.get()
                y = y.get()

                # Compute sample covariance matrices
                riemann_scm = RiemannSCM()
                x = riemann_scm.fit_transform(x, y)

                # Grid search per fold (RiemannTangentSpaceFeatureExtraction only need per fold, not per parameter)
                feature_extraction = RiemannTangentSpaceFeatureExtraction(max_iterations=200, num_jobs=num_worker)
                feature_reduction_1 = FeatureFusion((-1,))
                
                train_test_splits = []
                features = []
                labels = []
                start = time.time()
                for fold, (train_indices, test_indices) in enumerate(cv.split(x, y)):
                    # Extract features
                    features_train = feature_extraction.fit_transform(x[train_indices], y[train_indices])
                    features_test  = feature_extraction.transform(x[test_indices])

                    features_train = feature_reduction_1.transform(features_train)
                    features_test  = feature_reduction_1.transform(features_test)
 
                    # Adjust indices for concatenated features
                    offset = sum(len(f) for f in features)
                    train_test_splits.append((
                        np.arange(len(train_indices)) + offset,
                        np.arange(len(test_indices)) + offset + len(features_train)
                    ))

                    # Append to feature and label lists
                    features.extend([features_train, features_test])
                    labels.extend([y[train_indices], y[test_indices]])

                features = np.concatenate(features, axis=0)
                labels   = np.concatenate(labels, axis=0)
                end = time.time()
                print("Feature extraction Time:", end - start)
                print("Features:", features.shape, labels.shape)

                pipeline = Pipeline([
                    ('pls', DimensionReduction(2)),
                    #('feature_reduction', FeatureFusion((-1,))),
                    ('svm', SVC(kernel='rbf'))
                ])
            
                grid_search = GridSearchCV(pipeline, param_grid, cv=train_test_splits, scoring=scoring, refit=False, verbose=1, n_jobs=num_worker)
                start = time.time()
                grid_search.fit(features, labels)
                end = time.time()
                print("Grid search Time:", end - start)

                results = grid_search.cv_results_

                # Display parameter combinations and scores
                for mean_roc_auc, mean_accuracy, params in zip(
                    results['mean_test_roc_auc'], 
                    results['mean_test_accuracy'], 
                    results['params']
                ):
                    print(f"Parameters: {params}, ROC AUC: {mean_roc_auc:.4f}, Accuracy: {mean_accuracy:.4f}")

                # Store best scores and params
                best_index_roc_auc  = np.argmax(results['mean_test_roc_auc'])
                best_index_accuracy = np.argmax(results['mean_test_accuracy'])
                best_params_roc_auc = str(results['params'][best_index_roc_auc])
                best_params_accuracy = str(results['params'][best_index_accuracy])
                best_score_roc_auc = results['mean_test_roc_auc'][best_index_roc_auc]
                best_score_accuracy = results['mean_test_accuracy'][best_index_accuracy]
                output_person_group.attrs['best_params_roc_auc'] = best_params_roc_auc
                output_person_group.attrs['best_params_accuracy'] = best_params_accuracy
                output_person_group.attrs['best_score_roc_auc'] = best_score_roc_auc
                output_person_group.attrs['best_score_accuracy'] = best_score_accuracy


                # Store all cv results as dataset
                cv_group = output_person_group.create_group('cv_results')
                for key, value in results.items():
                    if isinstance(value, np.ndarray):
                         # Save arrays directly
                         cv_group.create_dataset(key, data=value)
                    else:
                         # Save non-array data (like strings) as attributes
                         cv_group.attrs[key] = str(value)




if __name__ == '__main__':
    main()
import h5py
import cupy as cp
import numpy as np
from sklearn.model_selection import RepeatedStratifiedKFold

from riemann_tangent_space_feature_extraction import RiemannTangentSpaceFeatureExtraction
from riemann_scm import RiemannSCM
from feature_fusion import FeatureFusion
from dimension_reduction import DimensionReduction
from rnn import RNNClassifier

from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, ParameterSampler

from time_window_decomposition import TimeWindowDecomposition
from frequency_decomposition import FrequencyDecomposition
import time

from joblib import Parallel, delayed

from sklearn.metrics import accuracy_score

num_splits = 5
num_repeats = 5
num_folds = num_splits * num_repeats

num_classes = 1

time_window_options = [250, 500, 1000, 1500, 3000]
frequency_band_options = [2, 4, 8, 16, 32]

time_overlap = 0.0
min_frequency = 8
max_frequency = 35
filter_order  = 4
num_epochs = 500

num_workers = 1
person_indices = [1, 2, 3, 4, 5, 6, 7, 8, 10, 11, 12, 13] # 9 is removed due to already gone through 2 Hz

device = 'cuda'

block = 'AB'
time_window_index = 2
frequency_band_index = 4
num_pls_components = 2

param_grid = {
    'rnn__num_lstm_layers': [32], #[1, 2, 4, 8, 16],
    'rnn__hidden_size': [64], # [2, 4, 8, 16, 32, 64, 128],
    'rnn__dropout_value': [0.2], # [0.1, 0.2],
    'pls__num_pls_components': [4] # [1, 2, 4, 8, 16],
}
num_param_grids = 10
param_sampler = ParameterSampler(param_grid, n_iter=num_param_grids)

def preprocess(person, time_length, frequency_width):
     with h5py.File('data/PERSONS_PREPROCESSED', 'r') as file:
        print(f'PERSON: {person}')

        group_a = file[f'PERSON{person}A']
        group_b = file[f'PERSON{person}B']
        x_a = cp.array(group_a['trials'][:, :, 1000:])
        x_b = cp.array(group_b['trials'][:, :, 1000:])
        x = cp.concatenate([x_a, x_b], dtype=cp.float64)
        y_a = cp.array(group_a['labels'])
        y_b = cp.array(group_b['labels'])
        y = cp.concatenate([y_a, y_b], dtype=cp.int64)
        fs = group_a.attrs['fs']

        print(x.shape, x.dtype, y.shape, y.dtype, fs)

        # Time window & Frequency preprocessing
        time_decomposer = TimeWindowDecomposition(0.0, time_length)
        freq_decomposer = FrequencyDecomposition(frequency_width, fs, filter_order, min_frequency, max_frequency)
        
        x = time_decomposer.fit_transform(x, y)
        x = freq_decomposer.fit_transform(x, y)

        print("Time & Frequency processed:", x.shape, x.dtype)
        # Transfer from GPU to CPU
        x = x.get()
        y = y.get()

        assert len(x) == len(y)

        # Compute sample covariance matrices
        riemann_scm = RiemannSCM()
        x = riemann_scm.fit_transform(x, y)

        # Grid search per fold (RiemannTangentSpaceFeatureExtraction only need per fold, not per parameter)
        feature_extraction = RiemannTangentSpaceFeatureExtraction(max_iterations=200, num_jobs=num_workers)
        feature_fusion = FeatureFusion((x.shape[1], -1))

        train_test_splits = []
        features = []
        labels = []
        cv = RepeatedStratifiedKFold(n_splits=num_splits, n_repeats=num_repeats, random_state=42)
        for fold, (train_indices, test_indices) in enumerate(cv.split(x, y)):
            # Extract features
            features_train = feature_extraction.fit_transform(x[train_indices], y[train_indices])
            features_test  = feature_extraction.transform(x[test_indices])

            features_train = feature_fusion.transform(features_train)
            features_test  = feature_fusion.transform(features_test)

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

        print("Preprocessed:", features.shape, labels.shape)

        return train_test_splits, features, labels
     
def process_fold(fold, train_test_split, features, labels, params, num_epochs, device):
    train_indices, test_indices = train_test_split
    fold_x_train, fold_x_test = features[train_indices], features[test_indices]
    fold_y_train, fold_y_test = labels[train_indices], labels[test_indices]

    pls = DimensionReduction(params['pls__num_pls_components'], True)
    rnn = RNNClassifier(
        num_epochs=num_epochs,
        device=device,
        num_lstm_layers=params['rnn__num_lstm_layers'],
        hidden_size=params['rnn__hidden_size'],
        dropout_value=params['rnn__dropout_value'],
        #learning_rate=0.001,
    )

    fold_x_train = pls.fit_transform(fold_x_train, fold_y_train)
    fold_x_test = pls.transform(fold_x_test)
    
    rnn.fit_eval(fold_x_train, fold_x_test, fold_y_train, fold_y_test)
    fold_preds = rnn.predict(fold_x_test)
    fold_scores = accuracy_score(fold_y_test, fold_preds)
    
    return fold, fold_scores

def grid_search(params, train_test_splits, features, labels, param_group:h5py.Group):
    # Parallelize the folds using joblib
    results = Parallel(n_jobs=num_workers)(
        delayed(process_fold)(
            fold,
            train_test_split,
            features,
            labels,
            params,
            num_epochs,
            device
        )
        for fold, train_test_split in enumerate(train_test_splits)
    )

    # Process the results
    fold_scores_list = []
    for fold, fold_scores in results:
        fold_scores_list.append(fold_scores)
        fold_group = param_group.create_group(f'fold{fold}')
        fold_group.create_dataset('fold_scores', data=fold_scores)
        print(f"Fold {fold + 1}/{len(train_test_splits)} completed with score: {fold_scores}")

    return np.mean(fold_scores_list)

def process_person(person, output_file):
    train_test_splits, features, labels = preprocess(person, time_window_options[time_window_index], frequency_band_options[frequency_band_index])

    with h5py.File(output_file, 'w') as output:
        output.attrs['num_folds'] = num_folds
        output.attrs['num_param_grids'] = num_param_grids

        for grid_idx, params in enumerate(param_sampler):

            param_group = output.create_group(f'param_grid{grid_idx}')
            param_group.attrs['params'] = str(params)

            grid_start_time = time.time()
            mean_grid_score = grid_search( params, train_test_splits, features, labels, param_group)
            grid_end_time = time.time()
            print(f"Person {person}, Grid {grid_idx+1}/{num_param_grids}, Params: {params}, 'Mean Grid Score: {mean_grid_score}', Time: {grid_end_time-grid_start_time}")

def process_fold_v2(fold, x_train, x_test, y_train, y_test, params, num_epochs, device):
    x_train = x_train.copy()
    x_test  = x_test.copy()

    pls = DimensionReduction(params['pls__num_pls_components'], True)
    rnn = RNNClassifier(
        num_epochs=num_epochs,
        device=device,
        num_lstm_layers=params['rnn__num_lstm_layers'],
        hidden_size=params['rnn__hidden_size'],
        dropout_value=params['rnn__dropout_value']
    )

    fold_x_train = pls.fit_transform(x_train, y_train)
    fold_x_test = pls.transform(x_test)

    rnn.fit_eval(fold_x_train, fold_x_test, y_train, y_test)
    fold_preds = rnn.predict(fold_x_test)
    fold_scores = accuracy_score(y_test, fold_preds)

    return fold, fold_scores


def grid_search_v2(params, freq_group, param_group, num_folds):

    results = Parallel(n_jobs=num_workers)(
        delayed(process_fold_v2)(
            fold,
            np.array(freq_group[f'fold_{fold}']['x_train']),
            np.array(freq_group[f'fold_{fold}']['x_test']),
            np.array(freq_group[f'fold_{fold}']['y_train']),
            np.array(freq_group[f'fold_{fold}']['y_test']),
            params,
            num_epochs,
            device
        )
        for fold in range(num_folds)
    )

    fold_scores_list = []
    for fold, fold_scores in results:
        fold_scores_list.append(fold_scores)
        
        fold_group = param_group.create_group(f'fold{fold}')
        fold_group.create_dataset('fold_scores', data=fold_scores)
        print(f"Fold {fold + 1}/{num_folds} completed with score: {fold_scores}")
    
    return np.mean(fold_scores_list)

def process_person_v2(person, time_length, frequency):
    print(f'Starting processing person {person}')

    with h5py.File(f'data/rnn_dataset', 'r') as input_file:
        num_folds = input_file.attrs['num_folds']

        person_group = input_file[f'Person_{person}']
        time_group = person_group[f'time_{time_length}']
        freq_group = time_group[f'frequency_{frequency}']

        # Grid search
        with h5py.File(f'data/rnn_grid_search_block_AB_time_{time_length}_frequency_{frequency}_PERSON_{person}_test', 'w') as output_file:
            output_file.attrs['num_folds'] = num_folds
            output_file.attrs['num_param_grids'] = num_param_grids

            for grid_idx, params in enumerate(param_sampler):
                param_group = output_file.create_group(f'param_grid{grid_idx}')
                param_group.attrs['params'] = str(params)

                grid_start_time = time.time()
                mean_grid_score = grid_search_v2(params, freq_group, param_group, num_folds)
                grid_end_time = time.time()
                print(f"Person {person}, Grid {grid_idx+1}/{num_param_grids}, Params: {params}, 'Mean Grid Score: {mean_grid_score}', Time: {grid_end_time-grid_start_time}")


#for person in person_indices:
#    process_person_v2(person, 1000, 2)
process_person_v2(4, 1000, 2)
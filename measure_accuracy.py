import h5py
import cupy as cp
import numpy as np

from riemann_tangent_space_feature_extraction import RiemannTangentSpaceFeatureExtraction
from riemann_scm import RiemannSCM
from feature_fusion import FeatureFusion
from dimension_reduction import DimensionReduction
from rnn import RNNClassifier, RNNDebugEpoch

from sklearn.model_selection import RepeatedStratifiedKFold
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
persons_of_interest = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
min_freq = 8
max_freq = 35
filter_order = 4
num_worker = 4
num_person = len(frequency_bands_per_person)
time_window_length = 1000
num_epochs = 300


with h5py.File('data/PERSONS_PREPROCESSED', 'r') as file:
    average_times = []

    for p_idx, person in enumerate(persons_of_interest):
        print(f'Person {person}')
        group_a = file[f'PERSON{person}A']
        group_b = file[f'PERSON{person}B']
        x_a = cp.array(group_a['trials'][:, :, 1000:])
        x_b = cp.array(group_b['trials'][:, :, 1000:])
        x = cp.concatenate([x_a, x_b], dtype=cp.float64)
        y_a = cp.array(group_a['labels'])
        y_b = cp.array(group_b['labels'])
        y = cp.concatenate([y_a, y_b], dtype=cp.int64)
        fs = group_a.attrs['fs']

        # Divide into train and test
        cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=42)

        # Get best params per person
        path_rnn = f'data/rnn_optimized_block_AB_time_{1000}_frequency_{frequency_bands_per_person[person-1]}'
        with h5py.File(path_rnn, 'r') as rnn_params_file:
            best_params = eval(rnn_params_file[f'PERSON{person}'].attrs['best_params'])
            print(f"Person {person} Best params: ", best_params)

            rnn_num_lstm_layers = best_params['rnn__num_lstm_layers']
            rnn_num_epochs = best_params['rnn__num_epochs']
            rnn_hidden_size = best_params['rnn__hidden_size']
            rnn_dropout_value = best_params['rnn__dropout_value']
            pls_num_pls_components = best_params['pls__num_pls_components']

            rnn_num_epochs = 600


        
        # Preprocess anything possible outside folds
        preprocessing_pipeline = make_pipeline(
            TimeWindowDecomposition(0.0, time_window_length),
            FrequencyDecomposition(frequency_bands_per_person[person-1], fs, filter_order, min_freq, max_freq),
            cupy_to_numpy(),
            RiemannSCM(),
        )

        x = preprocessing_pipeline.fit_transform(x, y)
        print("x", x.shape)
        num_time_windows = x.shape[1]

        times = []
        debugs_folds = []
        with h5py.File(f'data/PERSON{person}_Accuracy_RNN', 'w') as output:
            output.attrs['num_folds'] = cv.get_n_splits()

            for fold_idx, (train_indices, test_indices) in enumerate(cv.split(x, y.get())):
                print("Fold:", fold_idx)

                x_train, x_test = x[train_indices], x[test_indices]
                y_train, y_test = y[train_indices].get(), y[test_indices].get()

                # Create the pipeline
                rnn_pipeline = make_pipeline(
                    RiemannTangentSpaceFeatureExtraction(max_iterations=200, num_jobs=num_worker),
                    FeatureFusion((num_time_windows, -1)),
                    DimensionReduction(pls_num_pls_components, True),
                )

                rnn_num_epochs = 600
                rnn_classifier = RNNClassifier(rnn_num_epochs, num_lstm_layers=rnn_num_lstm_layers, hidden_size=rnn_hidden_size, dropout_value=rnn_dropout_value, device='cuda')

                # Fit & Test
                x_train = rnn_pipeline.fit_transform(x_train, y_train)
                x_test = rnn_pipeline.transform(x_test)

                debug = RNNDebugEpoch(rnn_num_epochs)
                rnn_classifier.debug_fit_test(x_train, x_test, debug, y_train, y_test)

                fold_group = output.create_group(f'fold{fold_idx}')
                fold_group.create_dataset('train_losses', data=debug.train_losses)
                fold_group.create_dataset('train_accuracy', data=debug.train_scores)
                fold_group.create_dataset('test_accuracy', data=debug.test_scores)
                fold_group.create_dataset('test_auc', data=debug.test_aucs)
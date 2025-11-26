import numpy as np
import cupy as cp
import matplotlib.pyplot as plt
from pyriemann.utils.distance import distance_riemann
from pyriemann.utils.mean import mean_riemann
import h5py


from riemann_distance_feature_extraction import RiemannDistanceFeatureExtraction
from riemann_tangent_space_feature_extraction import RiemannTangentSpaceFeatureExtraction
from riemann_scm import RiemannSCM
from feature_fusion import FeatureFusion
from dimension_reduction import DimensionReduction
from mdrm import MDRM
from time_window_decomposition import TimeWindowDecomposition
from frequency_decomposition import FrequencyDecomposition

from sklearn.pipeline import make_pipeline

from sklearn.model_selection import RepeatedStratifiedKFold

num_splits = 5
num_repeats = 1


with h5py.File('data/PERSONS_PREPROCESSED', 'r') as file:    
    for p in range(1, 14):

        print(f'PERSON {p}')

        fig = plt.figure()

        group_a = file[f'PERSON{p}A']
        group_b = file[f'PERSON{p}B']
        x_a = cp.array(group_a['trials'][:, :, 1000:])
        x_b = cp.array(group_b['trials'][:, :, 1000:])
        x = cp.concatenate([x_a, x_b])
        y_a = cp.array(group_a['labels'])
        y_b = cp.array(group_b['labels'])
        y = cp.concatenate([y_a, y_b])
        fs = group_a.attrs['fs']

        x = x_a
        y = y_a

        print(x.shape, x.dtype, y.shape, y.dtype, fs)

        xt = TimeWindowDecomposition(0.0, 1000).transform(x)
        print("XT before:", xt.shape)
        yt = cp.repeat(y[:, cp.newaxis], xt.shape[1], axis=1)
        xt = xt.reshape(xt.shape[0] * xt.shape[1], xt.shape[2], xt.shape[3])
        yt = yt.reshape(yt.shape[0] * yt.shape[1])
        print("XT", xt.shape, yt.shape)
        #x = xt
        #y = yt.get()
        #x = TimeWindowDecomposition(0.0, 3000).transform(x)
        y = y.get()
        # Preprocess all signals before k-fold
        

        #print(xwb.shape)

        rskf = RepeatedStratifiedKFold(n_splits=num_splits, n_repeats=num_repeats, random_state=42)
        fig, axes = plt.subplots(ncols=num_splits)
        for fold, (train_indices, test_indices) in enumerate(rskf.split(x, y)):

             # Prepare training and testing data
            x_train, x_test = x[train_indices], x[test_indices]
            y_train, y_test = y[train_indices], y[test_indices]

            x_train = TimeWindowDecomposition(0.0, 1000).transform(x_train)
            y_train = np.repeat(y_train[:, np.newaxis], x_train.shape[1], axis=1).reshape(-1)
            x_train = x_train.reshape(x_train.shape[0] * x_train.shape[1], x_train.shape[2], x_train.shape[3])

            train_indices = np.arange(len(x_train))
            np.random.shuffle(train_indices)
            x_train = x_train[train_indices, ...]
            y_train = y_train[train_indices, ...]

            x_test = TimeWindowDecomposition(0.0, 1000).transform(x_test)
            y_test = np.repeat(y_test[:, np.newaxis], x_test.shape[1], axis=1).reshape(-1)
            x_test = x_test.reshape(x_test.shape[0] * x_test.shape[1], x_test.shape[2], x_test.shape[3])
            
            print("X_train", x_train.shape, y_train.shape)
            print("X_test", x_test.shape, y_test.shape)

            preprocessing = make_pipeline(FrequencyDecomposition(32, fs, 5, 8, 35))
            x_train = preprocessing.fit_transform(x_train).get()
            x_test = preprocessing.transform(x_test).get()

            riemann_distance_feature_extraction = RiemannDistanceFeatureExtraction(max_iterations=200, num_jobs=4)
            rg = make_pipeline(RiemannSCM(), riemann_distance_feature_extraction)
            mdrm = MDRM()

            rg_train = rg.fit_transform(x_train, y_train)
            rg_test = rg.transform(x_test)
            mdrm.fit(rg_train)

            probs = mdrm.predict_probs(rg_test)
            preds = np.round(probs).argmax(axis=1)
            num_correct = 0
            for i in range(len(preds)):
                if riemann_distance_feature_extraction.classes_[preds[i]] == y_train[i]:
                    num_correct += 1
            
            print("Accuracy", num_correct / len(y))
            print("Probs shape:", probs.shape)

            # Plot distances
            for cls in range(len(riemann_distance_feature_extraction.classes_)):
                x_class = rg_test[y_test == cls]
                x_class = np.sum(x_class, axis=tuple(range(1, len(x_class.shape) - 1)))
                print(x_class.shape)
                num_trials, *batch, num_classes = x_class.shape

                axes[fold].scatter(x_class[:, 0], x_class[:, 1], label=f'Class {cls}', alpha=0.7)

            # Add labels and legend
            axes[fold].axline((0, 0), slope=1, color='gray', linestyle='--', label='Equal Distance')
            axes[fold].set_xlabel('Distance to Class 0')
            axes[fold].set_ylabel('Distance to Class 1')
            axes[fold].set_title(f'Distances to Class Means, person: {p}')

        plt.legend()
        plt.grid(True)
        plt.show()
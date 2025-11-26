from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np

from pyriemann.utils.mean import mean_riemann
from pyriemann.tangentspace import tangent_space

from joblib import Parallel, delayed

class RiemannTangentSpaceFeatureExtraction(BaseEstimator, TransformerMixin):
    def __init__(self, threshold = 1e-8, max_iterations = 50, num_jobs = -1):
        super().__init__()

        self.threshold = threshold
        self.max_iterations = max_iterations
        self.num_jobs = num_jobs

    def fit(self, X, y=None):
        """"
        Fits X to get the mean according to: https://www.overleaf.com/project/67975fd402977ca39095383e.

        Args:
        - X: Numpy array of the data, with the shape (num_trials, ..., num_channels, num_channels)
        - y: Numpy array of the classes, with the shape (num_trials,)
        """

        num_trials, *batch, num_channels, num_channels = X.shape
        self.means = np.empty((*batch, num_channels, num_channels))

        def compute_mean_for_batch(batch_idx):
            X_batch = X[(slice(None),) + batch_idx + (slice(None), slice(None))]
            means_batch = mean_riemann(X_batch, tol=self.threshold, maxiter=self.max_iterations)
            return batch_idx, means_batch
        
        results = Parallel(n_jobs=self.num_jobs)(delayed(compute_mean_for_batch)(batch_idx) for batch_idx in np.ndindex(*batch))
        for batch_idx, means_batch in  results:
            self.means[batch_idx] = means_batch
        
        return self
    
    def transform(self, X):

        num_trials, *batch, num_channels, num_channels = X.shape

        def compute_tangent_space_for_batch(batch_idx):
            X_batch = X[(slice(None),) + batch_idx + (slice(None), slice(None))]
            features_batch = tangent_space(X_batch, self.means[batch_idx])
            return batch_idx, features_batch
        
        results = Parallel(n_jobs=self.num_jobs)(delayed(compute_tangent_space_for_batch)(batch_idx) for batch_idx in np.ndindex(*batch))

        features = np.empty((num_trials, *batch, int(num_channels * (num_channels + 1) / 2)))
        for batch_idx, features_batch in  results:
            features[(slice(None),) + batch_idx] = features_batch

        return features
    
    def fit_transform(self, X, y = None):
        self.fit(X, y)
        return self.transform(X)
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.cross_decomposition import PLSRegression
from sklearn.decomposition import KernelPCA
from ikpls.numpy_ikpls import PLS
from sklearn.preprocessing import StandardScaler

class DimensionReduction(BaseEstimator, TransformerMixin):
    def __init__(self, num_pls_components, scale=True):
        self.num_pls_components = num_pls_components
        self.scale = scale
    
    def fit(self, X, y=None):

        num_trials, *batch, num_features = X.shape

        self.pls = []
        for batch_idx in np.ndindex(*batch):

            pls = PLS(algorithm=1)
            pls.fit(X[(slice(None),) + batch_idx], y, self.num_pls_components)

            self.pls.append(pls)
        return self
    
    def transform(self, X):

        num_trials, *batch, num_features = X.shape
        X_reduced = np.empty((num_trials, *batch, self.num_pls_components))

        for i, batch_idx in enumerate(np.ndindex(*batch)):
            pls = self.pls[i]
            X_batch = X[(slice(None),) + batch_idx]

            # Normalize
            X_batch -= pls.X_mean
            X_batch = X_batch / pls.X_std
            # Apply rotation
            x_scores = np.dot(X_batch, pls.R)
            X_reduced[(slice(None),) + batch_idx] = x_scores

        return X_reduced
    
    def fit_transform(self, X, y = None):
        return self.fit(X, y).transform(X)
    
class DimensionReductionPCA(BaseEstimator, TransformerMixin):
    def __init__(self, num_pls_components, gamma=None, scale=True):
        self.num_pls_components = num_pls_components
        self.gamma = gamma
        self.scale = scale
    
    def fit(self, X, y=None):

        num_trials, *batch, num_features = X.shape

        self.pca = []
        self.scaler = []
        for batch_idx in np.ndindex(*batch):
            X_batch = X[(slice(None),) + batch_idx]

            pca = KernelPCA(n_components=self.num_pls_components, kernel='rbf', gamma=self.gamma)
            scaler = StandardScaler()
            X_batch = scaler.fit_transform(X_batch, y)

            pca.fit(X_batch, y)

            self.pca.append(pca)
            self.scaler.append(scaler)
        return self
    
    def transform(self, X):

        num_trials, *batch, num_features = X.shape
        X_reduced = np.empty((num_trials, *batch, self.num_pls_components))

        for i, batch_idx in enumerate(np.ndindex(*batch)):
            pca = self.pca[i]
            scaler = self.scaler[i]

            X_batch = X[(slice(None),) + batch_idx]
            X_batch = scaler.transform(X_batch)
            X_scores = pca.transform(X_batch)

            X_reduced[(slice(None),) + batch_idx] = X_scores

        return X_reduced
    
    def fit_transform(self, X, y = None):
        return self.fit(X, y).transform(X)
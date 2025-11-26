import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin

class FeatureFusion(BaseEstimator, TransformerMixin):
    def __init__(self, shape):
        """
        Initialize the reshaper with specified axes for concatenation/reshaping.

        Args:
        - axes (tuple): Axes along which to concatenate the features.
        """

        self.shape = shape
    
    def fit(self, X, y=None):
        """
        No fitting required for this transformer.

        Args:
        - X: Input data (not used in fit for this transformer).
        - y: Target values (not used in fit for this transformer).

        Returns:
        - self
        """
        return self
    
    def transform(self, X):
        return X.reshape((X.shape[0],) + self.shape)
    
    def fit_transform(self, X, y = None):
        return self.fit(X, y).transform(X)
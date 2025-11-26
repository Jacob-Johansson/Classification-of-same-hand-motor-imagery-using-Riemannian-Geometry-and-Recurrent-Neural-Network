from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from pyriemann.utils.covariance import covariance_scm

class RiemannSCM(BaseEstimator, TransformerMixin):
  def __init__(self, lambda_regulation=10e-6):
    super().__init__()

    self.lambda_regulation = lambda_regulation

  def fit(self, X, y=None):
    return self

  def transform(self, X):
    *batch, num_channels, num_samples = X.shape
    scsms = np.empty((*batch, num_channels, num_channels))
    for idx in np.ndindex(*batch):
      scsms[idx] = covariance_scm(X[idx], assume_centered=False) + self.lambda_regulation * np.eye(num_channels)

    return scsms
  
  def fit_transform(self, X, y=None):
    self.fit(X, y)
    return self.transform(X)
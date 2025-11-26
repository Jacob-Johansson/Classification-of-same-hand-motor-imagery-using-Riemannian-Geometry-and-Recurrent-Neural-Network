from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
from pyriemann.utils.distance import distance_riemann
from pyriemann.utils.mean import mean_riemann
from joblib import Parallel, delayed

class RiemannDistanceFeatureExtraction(BaseEstimator, TransformerMixin):
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

    # Compute the means for each class
    self.classes_ = np.unique(y)
    self.means = np.empty((len(self.classes_), *batch, num_channels, num_channels))

    def compute_mean_for_batch(class_index, batch_idx):
      X_class_batch = X[(y == self.classes_[class_index],) + batch_idx + (slice(None), slice(None))]
      means_class_batch = mean_riemann(X_class_batch, tol=self.threshold, maxiter=self.max_iterations)
      return class_index, batch_idx, means_class_batch
  
    results = Parallel(n_jobs=self.num_jobs)(delayed(compute_mean_for_batch)(class_index, batch_idx)
                                             for class_index in range(len(self.classes_))
                                             for batch_idx in np.ndindex(*batch))
    
    for class_index, batch_idx, means_class_batch in results:
      self.means[(class_index,) + batch_idx] = means_class_batch

    return self

  def transform(self, X):
    """"

    Args:
    - X: Numpy array of the data, with the shape (num_trials, ..., num_channels, num_channels)

    """
    num_trials, *batch, num_channels, num_channels = X.shape

    def compute_distances_for_batch(class_index, batch_idx):
      distances_batch = [distance_riemann(X[(trial,) + batch_idx], self.means[(class_index,) + batch_idx]) for trial in range(num_trials)]
      return class_index, batch_idx, distances_batch
    
    results = Parallel(n_jobs=self.num_jobs)(delayed(compute_distances_for_batch)(class_index, batch_idx)
                                             for class_index in range(len(self.classes_))
                                             for batch_idx in np.ndindex(*batch))
    
    distances = np.empty((num_trials, *batch, len(self.classes_)))

    for class_index, batch_idx, distances_batch in results:
      distances[(slice(None),) + batch_idx + (class_index,)] = distances_batch

    return distances
  
  def fit_transform(self, X, y=None):
    self.fit(X, y)
    return self.transform(X)
  


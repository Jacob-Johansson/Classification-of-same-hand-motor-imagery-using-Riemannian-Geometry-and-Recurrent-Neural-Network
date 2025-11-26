from sklearn.base import BaseEstimator, TransformerMixin
import numpy as np
import torch
import components

class RMMDRM():
    def __init__(self, num_classes):
        self.means = dict()
        self.classes = dict()

    def fit(self, x:torch.Tensor, y:torch.Tensor):
        """
        Fits the MDRM classifier to the data.
        Input:
        - x: A tensor of the covariance matrices, with the shape (num_trials, num_windows, num_frequency_bands, num_channels, num_channels).
        - y: A tensor of the labels, with the shape (num_trials).
        """

        classes = torch.unique(y)

        num_trials, num_windows, num_frequency_bands, num_channels, num_channels = x.shape

        for index, c in enumerate(classes):
            xc = x[y == c, :, :, :, :]

            means = []
            # Compute Riemannian mean for each frequency band for all time windows
            for w in range(num_windows):
                w_means = []
                for b in range(num_frequency_bands):
                    xcwb = xc[:, w, b, :, :]
                    w_means.append(torch.from_numpy(components.component_riemannian_mean(xcwb.cpu().numpy(), max_iterations=200)))
                means.append(w_means)
            self.means[c] = means
            self.classes[c] = index
        self.num_classes = len(classes)

    def transform(self, x:torch.Tensor) -> torch.Tensor:
        """
        Transforms the data and returns riemannian distance to each class provided in the fit function.
        Input:
        - x: A tensor of the covariance matrices, with the shape (num_trials, num_windows, num_frequency_bands, num_channels, num_channels).
        Output:
        - A tensor representing the distance to each class, with the shape (num_trials, num_windows, num_frequency_bands, num_classes)
        """
        num_trials, num_windows, num_frequency_bands, num_channels, num_channels = x.shape

        class_means = torch.zeros((num_windows, num_frequency_bands, self.num_classes, num_channels, num_channels), device=x.device, dtype=x.dtype)
        for c, index in self.classes.items():
            for w in range(num_windows):
                for b in range(num_frequency_bands):
                    class_means[w, b, index] = self.means[c][w][b]

        return components.component_riemannian_distance_to_mean_batched(x, class_means)
    
    def predict_probabilities(self, x:torch.Tensor) -> torch.Tensor:
        """
        Input:
        - x: A tensor of distances to each class, with the shape (num_trials, num_windows, num_frequency_bands, num_classes)
        Output:
        - A tensor of probabilities for each class, with the shape (num_trials, num_classes)
        """
        average_distances = x.mean(dim=(1, 2))
        output = torch.softmax(-average_distances, dim=1)
        return output

class MDRM(BaseEstimator, TransformerMixin):
    def __init__(self):
        super().__init__()

    def fit(self, X, y=None):
        """
        
        Args:
        - X: Numpy array of distances to each class, with the shape (num_trials, ..., num_classes)
        """
        return self
    
    def transform(self, X):
        """
        
        Args:
        - X: Numpy array of distances to each class, with the shape (num_trials, ..., num_classes)
        """
        return np.argmin(X, axis=-1)      
    
    def fit_transform(self, X, y = None):
        """
        
        Args:
        - X: Numpy array of distances to each class, with the shape (num_trials, ..., num_classes)
        """

        self.fit(X, y)
        return self.transform(X)
    
    def predict_probs(self, X):
        """
        
        Args:
        - X: Numpy array of distances to each class, with the shape (num_trials, ..., num_classes)
        """

        absolutes = np.sum(X, axis=tuple(range(1, len(X.shape) - 1)))

        from scipy.special import softmax
        return softmax(-absolutes, axis=1)
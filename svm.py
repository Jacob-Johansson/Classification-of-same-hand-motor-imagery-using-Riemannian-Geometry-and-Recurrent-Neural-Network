from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV 
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.cross_decomposition import PLSRegression
import components
import torch
import numpy as np

# Riemannian Mean Support Vector Machine
class RMSVM():
    def __init__(self, num_pls_components, means:torch.Tensor):
        self.means = means
        self.svm = make_pipeline(SVC(kernel='rbf', probability=True)) #SVC(kernel='rbf', probability=True)
        self.num_pls_components = num_pls_components
        self.plss = []
        self.pls = PLSRegression(n_components=self.num_pls_components, scale=True)

        param_grid = {'C': [0.1, 1, 10, 100, 1000],  
              'gamma': [0.001, 0.01, 0.1, 1, 10, 100, 1000], 
              'kernel': ['rbf']}  
  
        self.grid = GridSearchCV(SVC(kernel='rbf', probability=True), param_grid, refit = True, verbose = 1) 

    def fit(self, x:torch.Tensor, y:torch.Tensor):
        """
        Fits the SVM classifier to the data.
        Input:
        - x: A tensor of the covariance matrices, with the shape (num_trials, num_windows, num_frequency_bands, num_channels, num_channels).
        - y: A tensor of the labels, with the shape (num_trials).
        """
        num_trials, num_windows, num_frequency_bands, num_channels, num_channels = x.shape

        # Vectorize each covariance matrix in each time window and frequency band for each trial
        features = components.component_vectorize_covariance_matrices_batched(x, self.means)
        features_reduced = np.zeros((num_trials, num_windows, num_frequency_bands, self.num_pls_components), dtype=np.float64)
        for w in range(num_windows):
            plss = []
            for b in range(num_frequency_bands):
                pls = PLSRegression(n_components=self.num_pls_components, scale=True)

                scores_x, scores_y = pls.fit_transform(features[:, w, b, :].cpu(), y.cpu())
                features_reduced[:, w, b, :] = scores_x
                plss.append(pls)
            self.plss.append(plss)

        features_reduced, _ = self.pls.fit_transform(features.reshape(num_trials, -1).cpu(), y.cpu())
        self.svm.fit(features_reduced, y.cpu())
        #self.svm.fit(features_reduced.reshape(num_trials, -1), y.cpu())
        self.classes = self.svm.classes_


    def transform(self, x:torch.Tensor) -> torch.Tensor:
        """
        Transforms the data into feature vectors.
        Input:
        - x: A tensor of the covariance matrices, with the shape (num_trials, num_windows, num_frequency_bands, num_channels, num_channels).
        Output:
        - A Tensor of vectors, with the shape (num_trials, num_windows * num_frequency_bands * num_channels * (num_channels + 1) / 2).
        """
        num_trials, num_windows, num_frequency_bands, num_channels, num_channels = x.shape

        # Vectorize each covariance matrix in each time window and frequency band for each trial
        features = components.component_vectorize_covariance_matrices_batched(x, self.means)
        features_reduced = np.zeros((num_trials, num_windows, num_frequency_bands, self.num_pls_components), dtype=np.float64)
        for w in range(num_windows):
            for b in range(num_frequency_bands):
                pls = self.plss[w][b]

                scores_x = pls.transform(features[:, w, b, :].cpu())
                features_reduced[:, w, b, :] = scores_x
        
        return torch.from_numpy(self.pls.transform(features.reshape(num_trials, -1).cpu()))
        return torch.from_numpy(features_reduced.reshape(num_trials, -1)).to(device=x.device, dtype=x.dtype)

    
    def predict_probabilities(self, x:torch.Tensor):
        features = self.transform(x).cpu().numpy()
        probs    = self.svm.predict_proba(features)
        return probs
import torch
import torch.nn as nn
from torch.optim import Adamax
import torch.functional as F
import numpy as np
import components
from sklearn.preprocessing import StandardScaler
from sklearn.cross_decomposition import PLSRegression
from torch.utils.data import DataLoader, Dataset
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.metrics import accuracy_score, roc_auc_score

class SequentialDataset(Dataset):
    def __init__(self, data, labels):
        self.data = data
        self.labels = labels

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]
    
class RNNDebugEpoch:
    def __init__(self, num_epochs):
        self.train_losses = []
        self.train_scores = []
        self.test_aucs = []
        self.test_scores = []
        self.num_epochs = num_epochs
    
    def append_train(self, loss, score):
        self.train_losses.append(loss)
        self.train_scores.append(score)
    def append_test(self, auc, score):
        self.test_aucs.append(auc)
        self.test_scores.append(score)

    

class RMRNN():
    def __init__(self, num_pls_components, input_size, hidden_size, output_size, device, dtype, means:torch.Tensor):
        self.rnn = RNN(input_size, hidden_size, output_size, device, dtype).to('cuda')
        torch.compile(self.rnn)
        self.plss = []
        self.means = means
        self.num_pls_components = num_pls_components
    
    def fit(self, x:torch.Tensor, y:torch.Tensor):
        """
        Fits the model to the data.
        Input:
        - x: A tensor of the covariance matrices, with the shape (num_trials, num_windows, num_frequency_bands, num_channels, num_channels).
        - y: A tensor of the labels, with the shape (num_trials).
        """
        num_trials, num_windows, num_frequency_bands, num_channels, num_channels = x.shape

        # Vectorize each covariance matrix in each time window and frequency band for each trial
        features = components.component_vectorize_covariance_matrices_batched(x, self.means).to(dtype=torch.float32)

        #import matplotlib.pyplot as plt
        #import seaborn as sns
        #plt.figure(figsize=(12, 8))
        #sns.heatmap(torch.var(features.reshape(num_trials, num_windows, -1), dim=1).cpu().numpy(), cmap="viridis", annot=False, cbar=True)
        #plt.title("Variance Across Windows and Components")
        #plt.xlabel("trials")
        #plt.ylabel("Features")
        #plt.show()

        features_reduced = np.zeros((num_trials, num_windows, self.num_pls_components), dtype=np.float64)
        for w in range(num_windows):
            features_window = features[:, w, :, :].reshape(num_trials, -1)
            #print("Labels:", y.cpu())
            #print("Feature variance Before:", torch.var(features_window, dim=0), torch.var(features_window, dim=0).shape)
            pls = PLSRegression(n_components=self.num_pls_components, scale=True)
            features_reduced[:, w, :], _ = pls.fit_transform(features_window.cpu(), y.to(dtype=torch.float32).cpu())
            #print("Feature variance After:", torch.var(torch.from_numpy(features_reduced[:, w, :]), dim=0), torch.var(torch.from_numpy(features_reduced[:, w, :]), dim=0).shape)

            # Variance explained
            #total_variance_x = features_window.cpu().numpy().var(axis=0).sum()
            #total_variance_y = y.cpu().numpy().var()
            #variance_explained_X = pls.x_scores_.var(axis=0) / total_variance_x
            #variance_explained_Y = pls.y_scores_.var(axis=0) / total_variance_y
            #
            #print(f"Variance explained in X: {variance_explained_X}")
            #print(f"Variance explained in Y: {variance_explained_Y}")

            #correlations = [np.corrcoef(features_reduced[:, w, i], y.ravel())[0, 1] for i in range(self.num_pls_components)]
            #print(f"Correlation of PLS components with response: {correlations}")

            #import matplotlib.pyplot as plt
#
            #x = np.arange(1, len(variance_explained_X) + 1)  # Component numbers
            #width = 0.35  # Bar width
#
            #fig, ax = plt.subplots(figsize=(10, 6))
            #ax.bar(x - width/2, variance_explained_X, width, label="Variance in X")
            #ax.bar(x + width/2, variance_explained_Y, width, label="Variance in Y", alpha=0.7)
#
            ## Customize the plot
            #ax.set_xlabel("PLS Components")
            #ax.set_ylabel("Variance Explained")
            #ax.set_title("Variance Explained by PLS Components")
            #ax.set_xticks(x)
            #ax.legend()
#
            ## Show the plot
            #plt.tight_layout()
            #plt.show()

            self.plss.append(pls)

    def transform(self, x:torch.Tensor) -> torch.Tensor:
        """
        Transforms the data into scaled, dimensional reduced feature vectors.
        Input:
        - x: A tensor of the covariance matrices, with the shape (num_trials, num_windows, num_frequency_bands, num_channels, num_channels).
        """
        num_trials, num_windows, num_frequency_bands, num_channels, num_channels = x.shape

        # Vectorize each covariance matrix in each time window and frequency band for each trial
        features = components.component_vectorize_covariance_matrices_batched(x, self.means)
        features_reduced = torch.zeros((num_trials, num_windows, self.num_pls_components), device=x.device, dtype=x.dtype)
        for w in range(num_windows):
            features_reduced[:, w, :] = torch.from_numpy(self.plss[w].transform(features[:, w, :, :].reshape(num_trials, -1).cpu()))

        return features_reduced

class EarlyStopping:
    def __init__(self, patience=5, delta=0):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.min_loss = float('inf')
    
    def early_stop(self, loss):
        if loss < self.min_loss:
            self.min_loss = loss
            self.counter = 0
        elif loss > (self.min_loss + self.delta):
            self.counter += 1
            if self.counter >= self.patience:
                return True
        return False

class RNNClassifier(ClassifierMixin, BaseEstimator):
    def __init__(self,
                 num_epochs=100,
                 batch_size=128,
                 learning_rate=1e-3,
                 num_lstm_layers=1,
                 hidden_size=64,
                 dropout_value=0.0,
                 device='cpu'
                 ):
        super().__init__()

        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.num_lstm_layers = num_lstm_layers
        self.hidden_size = hidden_size
        self.dropout_value = dropout_value 
        self.device = device

        self.model = None
        self.optimizer = None
        self.criterion = nn.BCEWithLogitsLoss()
        

    def fit(self, X, y=None):

        # Get the classes
        self.classes_ = np.unique(y)

        # Convert from numpy to torch
        X = torch.from_numpy(X).to(device=self.device, dtype=torch.float32)
        y = torch.from_numpy(y).to(device=self.device, dtype=torch.float32)

        # Create dataset and dataloader
        dataset = SequentialDataset(X, y)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)

        # Build model
        self.rnn = RNN2(self.device, self.num_lstm_layers, self.dropout_value, X.shape[2], self.hidden_size)
        self.optimizer = Adamax(self.rnn.parameters(), self.learning_rate, weight_decay=1e-6)

        # Train
        self.rnn.train()

        for epoch in range(self.num_epochs):
            for batch in dataloader:
                inputs, targets = batch

                outputs, _ = self.rnn(inputs)
                loss = self.criterion(outputs[:, 0], targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

        return self
    
    def fit_eval(self, X_train, X_test, y_train, y_test):
        # Get the classes
        self.classes_ = np.unique(y_train)

        # Convert from numpy to torch
        X_train = torch.from_numpy(X_train).to(device=self.device, dtype=torch.float32)
        y_train = torch.from_numpy(y_train).to(device=self.device, dtype=torch.float32)
        X_test = torch.from_numpy(X_test).to(device=self.device, dtype=torch.float32)
        y_test = torch.from_numpy(y_test).to(device=self.device, dtype=torch.float32)

        # Create dataset and dataloader
        train_dataset = SequentialDataset(X_train, y_train)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        test_dataset = SequentialDataset(X_test, y_test)
        test_dataloader = DataLoader(test_dataset, batch_size=len(X_test), shuffle=False)

        # Build model
        self.rnn = RNN2(self.device, self.num_lstm_layers, self.dropout_value, X_train.shape[2], self.hidden_size)
        self.optimizer = Adamax(self.rnn.parameters(), self.learning_rate, weight_decay=1e-6)
        self.early_stopping = EarlyStopping(patience=5, delta=0.05)

        scaler = torch.amp.GradScaler(self.device)

        for epoch in range(self.num_epochs):
            # Train
            self.rnn.train()
            for inputs, targets in train_dataloader:
                with torch.amp.autocast(self.device):
                    outputs, _ = self.rnn(inputs)
                    loss = self.criterion(outputs[:, 0], targets)

                self.optimizer.zero_grad()
                scaler.scale(loss).backward()
                scaler.step(self.optimizer)
                scaler.update()
            
            # Eval
            self.rnn.eval()
            epoch_loss = 0.0
            with torch.no_grad():
                for inputs, targets in test_dataloader:
                    outputs, _ = self.rnn(inputs)
                    loss = self.criterion(outputs[:, 0], targets)
                    epoch_loss += loss.item()

                mean_loss = epoch_loss / len(test_dataloader)
                if self.early_stopping.early_stop(mean_loss):
                    break



    
    def debug_fit_test(self, X_train, X_test, debug:RNNDebugEpoch, y_train, y_test):
        # Get the classes
        self.classes_ = np.unique(y_train)

        # Convert from numpy to torch
        X_train = torch.from_numpy(X_train).to(device=self.device, dtype=torch.float32)
        y_train = torch.from_numpy(y_train).to(device=self.device, dtype=torch.float32)
        X_test = torch.from_numpy(X_test).to(device=self.device, dtype=torch.float32)
        y_test = torch.from_numpy(y_test).to(device=self.device, dtype=torch.float32)

        # Create dataset and dataloader
        train_dataset = SequentialDataset(X_train, y_train)
        train_dataloader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)

        # Build model
        self.rnn = RNN2(self.device, self.num_lstm_layers, self.dropout_value, X_train.shape[2], self.hidden_size)
        self.optimizer = Adamax(self.rnn.parameters(), self.learning_rate, weight_decay=1e-6)


        for epoch in range(self.num_epochs):

            # Train
            self.rnn.train()

            epoch_loss = 0.0
            epoch_score = 0.0
            for batch in train_dataloader:
                inputs, targets = batch

                outputs, _ = self.rnn(inputs)
                loss = self.criterion(outputs[:, 0], targets)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                epoch_loss += loss.item()
                probs = torch.round(torch.sigmoid(outputs[:, 0])).detach().cpu().numpy()
                epoch_score += accuracy_score(probs, targets.cpu().numpy())

            epoch_loss = epoch_loss / len(train_dataloader)
            epoch_score = epoch_score / len(train_dataloader)
            debug.append_train(epoch_loss, epoch_score)

            # Test
            self.rnn.eval()
            with torch.no_grad():

                outputs, _ = self.rnn(X_test)
                probs = torch.round(torch.sigmoid(outputs[:, 0])).cpu().numpy()
                epoch_score = accuracy_score(probs, y_test.cpu().numpy())
                epoch_roc_auc = roc_auc_score(y_test.cpu().numpy(), torch.sigmoid(outputs[:, 0]).cpu().numpy())

                debug.append_test(epoch_roc_auc, epoch_score)

        return self


    def transform(self, X):

        # Convert from numpy to torch
        X = torch.from_numpy(X).to(device=self.device, dtype=torch.float32)

        # Transform
        self.rnn.eval()
        with torch.no_grad():
            outputs, _ = self.rnn(X)
            return outputs
        
    def predict(self, X):
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    def predict_proba(self, X):
        logits = self.transform(X)
        prob_1 = torch.sigmoid(logits).cpu().numpy()
        prob_0 = 1 - prob_1
        return np.column_stack((prob_0, prob_1))

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, device, dtype):
        super(RNN, self).__init__()

        self.num_layers = 1
        self.directional = 2

        self.lstm = nn.LSTM(input_size, hidden_size, num_layers=self.num_layers, batch_first=True, bidirectional=self.directional==2, device=device, dtype=dtype)
        self.dropout = nn.Dropout(p=0.2)
        
        self.fc = nn.Linear(self.directional * hidden_size, output_size, device=device, dtype=dtype)
        self.out_layer = nn.Sigmoid()
        self.hidden_size = hidden_size

    def forward(self, x:torch.Tensor):
        h0 = torch.zeros((self.directional * self.num_layers, x.shape[0], self.hidden_size), device=x.device, dtype=x.dtype)
        c0 = torch.zeros((self.directional * self.num_layers, x.shape[0], self.hidden_size), device=x.device, dtype=x.dtype)
        out, hidden = self.lstm(x, (h0, c0))
        out = self.dropout(out)

        out = self.fc(out[:, -1, :])
        out = self.out_layer(out)
        return out
        

class RNN2(nn.Module):
    def __init__(self, device, num_lstm_layers, dropout, input_size, hidden_size):
        super(RNN2, self).__init__()

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_lstm_layers = num_lstm_layers
        self.dropout_value = dropout

        self.device = device
        self.to(self.device)
        torch.compile(self)

        self.lstm = nn.LSTM(
            input_size=self.input_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_lstm_layers,
            batch_first=True,
            dropout=self.dropout_value,
            device=self.device,
            bidirectional=True
        )

        # Fully Connected Layer
        self.fc = nn.Sequential(
            nn.Linear(self.hidden_size * 2, 1, device=self.device),
            #nn.Sigmoid()
        )

    def forward(self, x):
        # x shape: (batch_size, sequence_length, input_size)

        # LSTM Layers
        x, _ = self.lstm(x)

        features = x[:, -1, :]
        x = self.fc(features)
        return x, features


import numpy as np
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torch

def visualize_tsne(features, labels, title, class_colors, class_names):
    """
    Visualize features in 2D using t-SNE.

    Args:
        features (numpy.ndarray): Feature vectors (num_samples, num_features).
        labels (numpy.ndarray): Class labels (num_samples,).
        title (str): Title for the plot.
        class_colors (list): Colors for each class.
        class_names (list): Names for each class.
    """
    # Apply t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=16, n_iter=1000)
    reduced_features = tsne.fit_transform(features)

    # Create scatter plot
    plt.figure(figsize=(8, 6))
    for i, class_name in enumerate(class_names):
        plt.scatter(
            reduced_features[labels == i, 0],
            reduced_features[labels == i, 1],
            c=class_colors[i],
            label=class_name,
            alpha=0.6,
            edgecolors='w'
        )
    plt.title(title)
    plt.legend()
    plt.xlabel("t-SNE Component 1")
    plt.ylabel("t-SNE Component 2")
    plt.grid(True)
    plt.show()
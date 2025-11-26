import scipy.signal as signal
import cupy as cp
import cupyx.scipy.signal as cpsignal
from sklearn.base import BaseEstimator, TransformerMixin

class FrequencyFilter:
    def __init__(self, wn, fs, order):
        self.sos = signal.butter(N=order, Wn=wn, btype='band', fs=fs, output='sos')
    
    def transform(self, x:cp.ndarray) -> cp.ndarray:
        return cpsignal.sosfiltfilt(self.sos, x.cpu().numpy(), -1)

class NotchFilter:
    def __init__(self, frequency:int, fs:int):
        self.b, self.a = cpsignal.iirnotch(frequency, 25, fs)
    
    def transform(self, x:cp.ndarray) -> cp.ndarray:
        return cpsignal.filtfilt(self.b, self.a, x, -1)

class HighpassFilter:
    def __init__(self, cutoff:int, fs:int, order:int):
        self.sos = cpsignal.butter(order, cutoff, btype='highpass', fs=fs, output='sos')
    
    def transform(self, x:cp.ndarray) -> cp.ndarray:
        return cpsignal.sosfiltfilt(self.sos, x, -1)

class FrequencyDecomposition(BaseEstimator, TransformerMixin):
    def __init__(self, frequency_step:int, fs:int, order:int, min_frequency:float, max_frequency:float):       
        self.frequency_step = frequency_step
        self.max_frequency = max_frequency
        self.min_frequency = min_frequency

        self.sos = []
        freq = min_frequency
        next_freq = 0
        idx = 0
        while next_freq < max_frequency:
            next_freq = min(freq + frequency_step, max_frequency)
            print(f'${freq} -> {next_freq}')
            self.sos.append(cpsignal.butter(N=order, Wn=[freq, next_freq], btype='band', fs=fs, output='sos'))
            idx+=1
            freq += 2
        
        self.num_frequency_bands = len(self.sos)

    def fit(self, x, y=None):
        return self

    def transform(self, x:cp.ndarray) -> cp.ndarray:
        """
        Component to apply bandpass filtering to the signal.
        Input:
        - x: A ndarray of data to filter, with the shape (..., num_channels, num_samples).
        Output:
        - A ndarray of filtered data, with the shape (..., num_bands, num_channels, num_samples)
        """

        *batch, num_channels, num_samples = x.shape
        output = cp.zeros((*batch, self.num_frequency_bands, num_channels, num_samples), dtype=x.dtype)
        
        for b in range(self.num_frequency_bands):
            output[..., b, :, :] = cpsignal.sosfiltfilt(self.sos[b], x, axis=-1)
        
        return output
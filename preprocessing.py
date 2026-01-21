from sklearn.preprocessing import StandardScaler
from scipy.signal import savgol_filter
import torch
import numpy as np
import pandas as pd

class PreprocessNIR():
    def __init__(self, savgol=False, window_length=5, polyorder=2, deriv=0, scale=True, scale_y=True):
        self.scale = scale
        self.scale_y = scale_y
        self.savgol = savgol

        if self.scale:
            self.scaler = StandardScaler()
        if self.scale_y:
            self.y_scaler = StandardScaler()
        if self.savgol:
            self.window_length = window_length
            self.polyorder = polyorder
            self.deriv = deriv


    def fit(self, X, y):
        if self.scale:
            self.scaler.fit(X)
        if self.scale_y:
            self.y_scaler.fit(y)
        return self

    def transform(self, X, y):
        if self.scale:
            X = self.scaler.transform(X)
        if self.scale_y:
            if y.ndim == 1:
                y = y.reshape(-1, 1)
            y = self.y_scaler.transform(y)
        if self.savgol:
            X = savgol_filter(X, window_length=self.window_length, polyorder=self.polyorder, deriv=self.deriv)

        if type(y) == np.ndarray:
            y = torch.tensor(y, dtype=torch.float32)
        if type(y) == pd.DataFrame:
            y = torch.tensor(y.values, dtype=torch.float32)
        if type(X) == np.ndarray:
            X = torch.tensor(X, dtype=torch.float32)
        if type(X) == pd.DataFrame:
            X = torch.tensor(X.values, dtype=torch.float32)
        return X, y

    def fit_transform(self, X, y):
        if self.scale:
            X = self.scaler.fit_transform(X)
        if self.scale_y:
            if y.ndim == 1:
                y = y.reshape(-1, 1)
            y = self.y_scaler.fit_transform(y)
        if self.savgol:
            X = savgol_filter(X, window_length=self.window_length, polyorder=self.polyorder, deriv=self.deriv)

        if type(y) == np.ndarray:
            y = torch.tensor(y, dtype=torch.float32)
        if type(y) == pd.DataFrame:
            y = torch.tensor(y.values, dtype=torch.float32)
        if type(X) == np.ndarray:
            X = torch.tensor(X, dtype=torch.float32)
        if type(X) == pd.DataFrame:
            X = torch.tensor(X.values, dtype=torch.float32)
        
        return X, y

    def inverse_transform(self, X, y):
        if self.scale:
            X = self.scaler.inverse_transform(X)
        if self.scale_y:
            y = self.y_scaler.inverse_transform(y)

        if type(y) == np.ndarray:
            y = torch.tensor(y, dtype=torch.float32)
        if type(X) == np.ndarray:
            X = torch.tensor(X, dtype=torch.float32)
        return X, y
from __future__ import annotations
import numpy as np
from typing import Any, Dict
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.utils.validation import check_is_fitted
from sklearn.preprocessing import StandardScaler
from skorch import NeuralNetBinaryClassifier
import torch
import torch.nn as nn

RANDOM_STATE = 42

class _TabTransformerModule(nn.Module):
    def __init__(self, n_features: int):
        super().__init__()
        self.tokenizer = nn.Sequential(
            nn.Linear(n_features, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
        )
        enc = nn.TransformerEncoderLayer(d_model=64, nhead=8, batch_first=True, dim_feedforward=128)
        self.encoder = nn.TransformerEncoder(enc, num_layers=2)
        self.cls = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1), nn.Sigmoid(),
        )

    def forward(self, X):
        tok = self.tokenizer(X)   # (B,64)
        tok = tok.unsqueeze(1)    # (B,1,64)
        enc = self.encoder(tok)   # (B,1,64)
        out = self.cls(enc)       # (B,1)
        return out

class SkorchTabTransformer(BaseEstimator, ClassifierMixin):
    def __init__(self, max_epochs: int = 50, lr: float = 1e-3,
                 batch_size: int = 64, patience: int = 8,
                 device: str = 'cpu', random_state: int = RANDOM_STATE):
        self.max_epochs = max_epochs; self.lr = lr
        self.batch_size = batch_size; self.patience = patience
        self.device = device; self.random_state = random_state
        self._net = None; self._scaler = StandardScaler()

    def fit(self, X: np.ndarray, y: np.ndarray):
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float32).reshape(-1)  # umesto reshape(-1, 1)
        Xs = self._scaler.fit_transform(X).astype(np.float32)

        import numpy as _np, torch as _torch
        _np.random.seed(self.random_state)
        _torch.manual_seed(self.random_state)

        module = _TabTransformerModule(n_features=Xs.shape[1])
        self._net = NeuralNetBinaryClassifier(
            module,
            max_epochs=self.max_epochs,
            lr=self.lr,
            batch_size=self.batch_size,
            optimizer=_torch.optim.Adam,
            train_split=None,
            iterator_train__shuffle=True,
            verbose=0,
            device=self.device,
        )
        self._net.fit(Xs, y)
        self.classes_ = np.array([0, 1], dtype=int)   # <<< DODAJ OVO
        self.is_fitted_ = True
        return self
    def predict(self, X: np.ndarray) -> np.ndarray:
        # potreban sklearn-u za validaciju; prag 0.5 (tuning radimo odvojeno)
        probs = self.predict_proba(X)[:, 1]
        return (probs >= 0.5).astype(int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        check_is_fitted(self, 'is_fitted_')
        X = np.asarray(X, dtype=np.float32)
        Xs = self._scaler.transform(X).astype(np.float32)
        proba = self._net.predict_proba(Xs)
        probs_pos = proba[:, 1:2] if proba.ndim == 2 and proba.shape[1] == 2 else proba.reshape(-1, 1)
        probs_neg = 1.0 - probs_pos
        return np.hstack([probs_neg, probs_pos])

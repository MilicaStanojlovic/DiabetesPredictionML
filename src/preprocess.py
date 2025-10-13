from __future__ import annotations
from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from .data import FEATURES, ZERO_AS_MISSING

def _winsorize_iqr(X: np.ndarray, k: float = 1.5) -> np.ndarray:
    X = X.copy()
    q1 = np.nanpercentile(X, 25, axis=0)
    q3 = np.nanpercentile(X, 75, axis=0)
    iqr = q3 - q1
    lo = q1 - k*iqr
    hi = q3 + k*iqr
    return np.clip(X, lo, hi)

from sklearn.preprocessing import FunctionTransformer

def build_preprocess(imputer: str = "median", scale: bool = True, winsor: bool = False) -> ColumnTransformer:
    if imputer not in {"median","knn"}:
        raise ValueError("imputer must be 'median' or 'knn'")
    if imputer == "median":
        imp = SimpleImputer(strategy="median")
    else:
        imp = KNNImputer(n_neighbors=5, weights="uniform")
    steps = [("imputer", imp)]
    if winsor:
        steps.append(("winsor", FunctionTransformer(_winsorize_iqr, validate=False)))
    if scale:
        steps.append(("scaler", StandardScaler()))
    pipe = Pipeline(steps)
    return ColumnTransformer([("num", pipe, FEATURES)], remainder="drop")
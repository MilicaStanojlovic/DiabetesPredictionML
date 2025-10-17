from __future__ import annotations
import pandas as pd

FEATURES = [
    "Pregnancies","Glucose","BloodPressure","SkinThickness","Insulin",
    "BMI","DiabetesPedigreeFunction","Age"
]
TARGET = "Outcome"

ZERO_AS_MISSING = ["Insulin","SkinThickness"]

def load_csv(path: str, zero_as_missing: bool = True) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = set([*FEATURES, TARGET]) - set(df.columns)
    if missing:
        raise ValueError(f"CSV missing columns: {missing}")
    
    df = df[[*FEATURES, TARGET]].copy()
    
    if zero_as_missing:
        for c in ZERO_AS_MISSING:
            df.loc[df[c] == 0, c] = pd.NA
    return df

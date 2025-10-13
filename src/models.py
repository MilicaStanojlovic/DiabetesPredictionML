from __future__ import annotations
from typing import Dict
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from .preprocess import build_preprocess

RANDOM_STATE = 42

def build_models(imputer: str = "median", scale: bool = True, winsor: bool = False) -> Dict[str, Pipeline]:
    pre = build_preprocess(imputer=imputer, scale=scale, winsor=winsor)

    models = {}

    lr = LogisticRegression(max_iter=200, class_weight="balanced", random_state=RANDOM_STATE)
    models["lr"] = Pipeline([("pre", pre), ("clf", lr)])

    rf = RandomForestClassifier(
        n_estimators=400, max_depth=None, n_jobs=-1, random_state=RANDOM_STATE, class_weight="balanced"
    )
    models["rf"] = Pipeline([("pre", pre), ("clf", rf)])

    svm = SVC(kernel="rbf", probability=True, class_weight="balanced", random_state=RANDOM_STATE)
    models["svm"] = Pipeline([("pre", pre), ("clf", svm)])

    mlp = MLPClassifier(hidden_layer_sizes=(32,16), activation="relu",
                        early_stopping=True, max_iter=200, random_state=RANDOM_STATE)
    models["mlp_32x16"] = Pipeline([("pre", pre), ("clf", mlp)])

    mlp_big = MLPClassifier(hidden_layer_sizes=(64,32), activation="relu",
                            early_stopping=True, max_iter=200, random_state=RANDOM_STATE, alpha=1e-3, learning_rate_init=0.001)
    models["mlp_64x32"] = Pipeline([("pre", pre), ("clf", mlp_big)])
    try:
        from .transformer_tab import SkorchTabTransformer
        from sklearn.compose import ColumnTransformer
        from sklearn.impute import SimpleImputer, KNNImputer
        from .data import FEATURES
        imp = SimpleImputer(strategy="median") if imputer == "median" else KNNImputer(n_neighbors=5, weights="uniform")
        # Izbegni duplu standardizaciju: ovde samo imputacija, skaliranje je u modelu
        pre_only_imp = ColumnTransformer([("num", imp, FEATURES)], remainder="drop")
        models["tabtr"] = Pipeline([("pre", pre_only_imp), ("clf", SkorchTabTransformer())])
    except Exception:
        pass
    return models

from __future__ import annotations
import argparse, os, json
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split, cross_val_predict
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score, precision_score, recall_score, f1_score,
    roc_curve, precision_recall_curve, confusion_matrix
)
from sklearn.base import BaseEstimator
import matplotlib.pyplot as plt
import joblib

from .data import load_csv, FEATURES, TARGET
from .models import build_models

RANDOM_STATE = 42

def find_threshold_for_recall(y_true: np.ndarray, y_prob: np.ndarray, min_recall: float = 0.80) -> float:
    # sweep thresholds on validation predictions to reach target recall
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)
    # thresholds array length is len(precisions)-1
    thr = 0.5
    idx = np.where(recalls[:-1] >= min_recall)[0]
    if len(idx) > 0:
        thr = max(0.0, min(1.0, thresholds[idx[-1]]))
    else:
        # if recall target unattainable, choose threshold that maximizes F1
        f1s = 2*precisions[:-1]*recalls[:-1]/(precisions[:-1]+recalls[:-1] + 1e-12)
        thr = float(thresholds[np.argmax(f1s)])
    return thr

def evaluate(y_true, y_prob, thr: float):
    y_pred = (y_prob >= thr).astype(int)
    return {
        "threshold": float(thr),
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "pr_auc": float(average_precision_score(y_true, y_prob)),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

def plot_curves(y_true, y_prob, out_dir, tag):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    fig = plt.figure()
    plt.plot(fpr, tpr, label="ROC")
    plt.plot([0,1],[0,1], linestyle="--")
    plt.xlabel("FPR"); plt.ylabel("TPR"); plt.title(f"ROC: {tag}")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f"roc_{tag}.png"))
    plt.close(fig)

    prec, rec, _ = precision_recall_curve(y_true, y_prob)
    fig = plt.figure()
    plt.plot(rec, prec)
    plt.xlabel("Recall"); plt.ylabel("Precision"); plt.title(f"PR: {tag}")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, f"pr_{tag}.png"))
    plt.close(fig)
def _json_default(o):
    if isinstance(o, (np.floating, np.float32, np.float64)): return float(o)
    if isinstance(o, (np.integer,  np.int32,  np.int64)):     return int(o)
    if isinstance(o, np.ndarray):                             return o.tolist()
    return str(o)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--imputer", choices=["median","knn"], default="median")
    ap.add_argument("--out", default="artifacts")
    ap.add_argument("--zero-missing", choices=["on","off"], default="on")
    ap.add_argument("--scale", choices=["on","off"], default="on")
    ap.add_argument("--winsor", choices=["on","off"], default="off")

    args = ap.parse_args()

    os.makedirs(args.out, exist_ok=True)
    df = load_csv(args.csv, zero_as_missing=(args.zero_missing=="on"))
    X = df.drop(columns=[TARGET])
    y = df[TARGET].astype(int).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=RANDOM_STATE
    )

    models = build_models(
    imputer=args.imputer,
    scale=(args.scale=="on"),
    winsor=(args.winsor=="on")
)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_results = {}
    best_name = None
    best_auc = -np.inf

    for name, pipe in models.items():
        # cross_val_predict on train for CV ROC-AUC
        y_prob_cv = cross_val_predict(pipe, X_train, y_train, cv=cv, method="predict_proba")[:,1]
        auc_cv = roc_auc_score(y_train, y_prob_cv)
        thr = find_threshold_for_recall(y_train, y_prob_cv, 0.80)
        cv_results[name] = evaluate(y_train, y_prob_cv, thr)
        cv_results[name]["cv_threshold"] = thr
        if auc_cv > best_auc:
            best_auc = auc_cv
            best_name = name

    # Fit best on full train, tune threshold on train-CV probs (already done)
    best_pipe = models[best_name]
    best_thr = cv_results[best_name]["cv_threshold"]
    best_pipe.fit(X_train, y_train)

    # Test evaluation
    y_prob_test = best_pipe.predict_proba(X_test)[:,1]
    test_metrics = evaluate(y_test, y_prob_test, best_thr)
    
    # --- dodatak: plot Confusion Matrix ---
    from sklearn.metrics import ConfusionMatrixDisplay

    cm = np.array(test_metrics["confusion_matrix"])
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(values_format="d")
    plt.title(f"Confusion Matrix: test_{best_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(args.out, f"cm_test_{best_name}.png"))
    plt.close()
    # --- kraj dodatka ---
    
    plot_curves(y_test, y_prob_test, args.out, f"test_{best_name}")

    # Persist
    joblib.dump(best_pipe, os.path.join(args.out, "best_model.pkl"))
    with open(os.path.join(args.out, "cv_results.json"), "w") as f:
        json.dump(cv_results, f, indent=2, default=_json_default)
    with open(os.path.join(args.out, "test_metrics.json"), "w") as f:
        json.dump(test_metrics, f, indent=2, default=_json_default)

    print(f"Best model by CV ROC-AUC: {best_name} = {best_auc:.4f}")
    print(f"Test metrics saved to {os.path.join(args.out, 'test_metrics.json')}")

if __name__ == "__main__":
    main()

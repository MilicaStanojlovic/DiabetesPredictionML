from __future__ import annotations
import argparse
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from .data import load_csv, FEATURES, TARGET

RANDOM_STATE = 42

def eda(df: pd.DataFrame, out_dir: str):
    os.makedirs(out_dir, exist_ok=True)
    # Missing summary
    miss = df.isna().mean().sort_values(ascending=False)
    miss.to_csv(os.path.join(out_dir, "missing_ratio.csv"))

    # Univariate: hist + box for all 8 features
    for col in FEATURES:
        fig = plt.figure()
        df[col].hist(bins=30)
        plt.title(f"Histogram: {col}")
        plt.xlabel(col); plt.ylabel("Count")
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, f"hist_{col}.png"))
        plt.close(fig)

        fig = plt.figure()
        plt.boxplot(df[col].dropna().values, vert=True)
        plt.title(f"Box-plot: {col}")
        plt.ylabel(col)
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, f"box_{col}.png"))
        plt.close(fig)

    # Bivariate: box by class for selected
    for col in ["Glucose","BMI","Age"]:
        fig = plt.figure()
        sns.boxplot(x=TARGET, y=col, data=df[[col, TARGET]].dropna())
        plt.title(f"{col} by Outcome")
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, f"box_{col}_by_outcome.png"))
        plt.close(fig)

    # Positive rates by quartiles for Glucose/BMI/Age
    for col in ["Glucose","BMI","Age"]:
        tmp = df[[col, TARGET]].dropna().copy()
        tmp["q"] = pd.qcut(tmp[col], 4, duplicates="drop")
        # rate = tmp.groupby("q")[TARGET].mean()
        rate = tmp.groupby("q", observed=False)[TARGET].mean()
        fig = plt.figure()
        rate.plot(kind="bar")
        plt.title(f"Positive rate by {col} quartiles")
        plt.ylabel("Positive rate")
        plt.tight_layout()
        fig.savefig(os.path.join(out_dir, f"rate_by_quartiles_{col}.png"))
        plt.close(fig)

    # Correlation heatmap (Pearson)
    corr = df[FEATURES + [TARGET]].corr(numeric_only=True)
    fig = plt.figure()
    sns.heatmap(corr, annot=True, fmt=".2f")
    plt.title("Correlation heatmap")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "corr_heatmap.png"))
    plt.close(fig)

    # Correlation heatmap (Spearman)
    corr_s = df[FEATURES + [TARGET]].corr(method="spearman", numeric_only=True)
    fig = plt.figure()
    sns.heatmap(corr_s, annot=True, fmt=".2f")
    plt.title("Correlation heatmap (Spearman)")
    plt.tight_layout()
    fig.savefig(os.path.join(out_dir, "corr_heatmap_spearman.png"))
    plt.close(fig)

    # Save simple EDA summary
    strong_pairs = []
    for a in FEATURES:
        for b in FEATURES:
            if a < b:
                r = corr.loc[a, b]
                if abs(r) > 0.7:
                    strong_pairs.append((a, b, float(r)))
    summary = {
        "rows": int(df.shape[0]),
        "missing_ratio": miss.to_dict(),
        "strong_corr_pairs_|r|>0.7": strong_pairs,
        "class_balance": df[TARGET].value_counts(normalize=True).to_dict()
    }
    pd.Series(summary).to_json(os.path.join(out_dir, "eda_summary.json"))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", required=True)
    ap.add_argument("--out", default="plots")
    args = ap.parse_args()
    df = load_csv(args.csv)
    eda(df, args.out)

if __name__ == "__main__":
    main()

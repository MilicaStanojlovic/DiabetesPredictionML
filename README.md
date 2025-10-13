# Diabetes Prediction (PIMA) — ML Project

## What you get
- **Fully reproducible** Python project for PIMA Indians Diabetes.
- **EDA** with histograms, box-plots, correlation heatmap, and bivariate views.
- **Preprocessing**: treat zeros in `Insulin` and `SkinThickness` as missing. Median or KNN imputers. StandardScaler.
- **Models**: Logistic Regression, Random Forest, SVM, MLP (small grid). 5-fold stratified CV. ROC-AUC primary.
- **Threshold tuning**: choose decision threshold to target Recall ≥ 0.80 on train-CV, then evaluate once on the held-out test.
- **Artifacts**: best model saved to `artifacts/best_model.pkl`, metrics JSON, ROC/PR curves.
- **App**: Streamlit UI for single-subject inference (`streamlit run app.py`).

## Quick start
1) Put the dataset at `data/diabetes.csv` (Kaggle PIMA). File must have columns:
   `Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age,Outcome`
2) Create a virtual env and install:
   ```bash
   pip install -r requirements.txt
   ```
3) Run EDA:
   ```bash
   python -m src.eda --csv data/diabetes.csv
   ```
4) Train and evaluate:
   ```bash
   python -m src.train --csv data/diabetes.csv --imputer median  # or knn
   ```
5) Launch Streamlit app:
   ```bash
   streamlit run app.py
   ```

## Notes
- Transformer model stub is provided in `src/transformer_stub.py` with instructions to plug a TabTransformer implementation later.
- All random seeds fixed for reproducibility.

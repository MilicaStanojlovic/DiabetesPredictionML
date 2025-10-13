# Predikcija dijabetesa korišćenjem metoda mašinskog učenja

## 1. Definicija problema
Cilj projekta je klasifikacija pacijenata na dve klase — **oboleli od dijabetesa** i **zdravi** —  
na osnovu medicinskih parametara poput nivoa glukoze, krvnog pritiska, BMI-ja i starosti.  
Projekat koristi mašinsko učenje za ranu detekciju rizika od dijabetesa.

---

## 2. Skup podataka
**Dataset:** PIMA Indians Diabetes Database (`data/diabetes.csv`)  
**Broj instanci:** 768  
**Broj atributa:** 8 numeričkih + ciljna promenljiva `Outcome`  
**Izvor:** [Kaggle – PIMA Diabetes Dataset](https://www.kaggle.com/datasets/uciml/pima-indians-diabetes-database)

---

## 3. Metodologija

### Podela podataka
- 80 % trening skup  
- 20 % test skup  
- Stratifikovano po ciljnoj promenljivoj  

### Predobrada
- Imputacija nedostajućih vrednosti: **median**  
- Standardizacija numeričkih atributa (`StandardScaler`)  
- Kodiranje ciljne promenljive u binarni format (0 = zdravi, 1 = dijabetes)

### Modeli korišćeni u evaluaciji
| Model | Opis |
|--------|------|
| LR | Logistic Regression |
| RF | Random Forest (400 stabala) |
| SVM | Support Vector Machine (RBF kernel) |
| MLP 32×16 | Neural Network (2 sloja) |
| MLP 64×32 | Neural Network (veća mreža) |
| TabTr | TabTransformer (PyTorch + Skorch implementacija) |

### Cross-validation
- 5-fold Stratified K-Fold  
- Glavna metrika: **ROC-AUC**  
- Dodatno praćene metrike: Precision, Recall, F1, Accuracy, PR-AUC  

---

## 4. Rezultati

### 📈 Cross-Validation (trening skup)

| Model | ROC-AUC | Precision | Recall | F1 | Accuracy | Threshold |
|-------|----------|-----------|--------|----|-----------|------------|
| LR | 0.8331 | 0.57 | 0.80 | 0.67 | 0.72 | 0.4167 |
| RF | 0.8215 | 0.58 | 0.80 | 0.67 | 0.73 | 0.2975 |
| **SVM ⭐** | **0.8356** | **0.61** | **0.80** | **0.69** | **0.75** | **0.3008** |
| MLP 32×16 | 0.496 | 0.33 | 0.80 | 0.46 | 0.36 | 0.1436 |
| MLP 64×32 | 0.8145 | 0.56 | 0.80 | 0.66 | 0.71 | 0.3595 |
| TabTr | 0.7260 | 0.40 | 0.80 | 0.53 | 0.51 | 0.5000 |

➡️ **Najbolji model:** SVM (RBF kernel)

---

### 🧪 Test skup (20 %)

| Metrika | Vrednost |
|----------|-----------|
| ROC-AUC | 0.800 |
| PR-AUC | 0.662 |
| Accuracy | 0.675 |
| Precision | 0.525 |
| Recall | 0.778 |
| F1 | 0.627 |
| Optimalni threshold | 0.30077 |

**Konfuziona matrica:**
[[62, 38],
[12, 42]]


---

## 5. Vizuelizacija

Slike generisane tokom evaluacije (u folderu `artifacts/`):

- `roc_test_svm.png` — ROC kriva  
- `pr_test_svm.png` — Precision-Recall kriva  

U oba grafikona, SVM pokazuje stabilnu separaciju i dobar balans između TPR i FPR.

---

## 6. Zaključak
Model **SVM (RBF kernel)** ostvario je najbolji rezultat po ROC-AUC i F1 metričkim vrednostima.  
Najefikasniji prag za odlučivanje je **0.30077**, što povećava osetljivost sistema.  

Model pruža pouzdanu osnovu za dijagnostičku pomoć, i može se dalje unaprediti:
- širenjem skupa podataka,  
- balansiranjem klasa (SMOTE),  
- optimizacijom hiperparametara.  

---

## 7. Arhitektura projekta
diabetes_pred_project/
│
├── data/
│ └── diabetes.csv
├── src/
│ ├── data.py
│ ├── preprocess.py
│ ├── models.py
│ ├── transformer_tab.py
│ └── train.py
├── artifacts/
│ ├── best_model.pkl
│ ├── test_metrics.json
│ ├── cv_results.json
│ ├── roc_test_svm.png
│ └── pr_test_svm.png
└── report.md

---

## 8. Reference
- UCI Machine Learning Repository  
- Kaggle PIMA Diabetes Dataset  
- Scikit-learn, PyTorch, Skorch biblioteke  

---

## 9. Autor
**Milica Stanojlović**  
Fakultet tehničkih nauka, Novi Sad  
2025.

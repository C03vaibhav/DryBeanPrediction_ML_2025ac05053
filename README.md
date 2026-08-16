# Dry Bean Classification — DryBeanPrediction_ML_2025ac05053

**Problem statement:** Predict the class of a dry bean sample from its measured morphological features. The goal is to compare several classic machine-learning classifiers on the same dataset and report evaluation metrics so the best performing model can be identified and used for prediction.

**Dataset description:**
- **Name:** Dry Bean Dataset (CSV expected as `Dry_Bean_Dataset.csv` in the project root).
- **Target column:** `Class` (categorical label identifying bean type).
- **Features:** Numerical morphological measurements (the repository code reads all columns except `Class` as input features). The preprocessing in [model/preprocess.py](model/preprocess.py#L1-L200) handles duplicate removal, label encoding, train/test splitting and standard scaling.

**GitHub repository:** Replace the URL below with your repository link (maintain the repo with all required files):

- Repository: https://github.com/<your-username>/DryBeanPrediction_ML_2025ac05053

**Models used:** The project contains scripts and training code for the following models (files under the `model/` folder):
- Logistic Regression — [model/logistic_regression.py](model/logistic_regression.py#L1-L200)
- Decision Tree — [model/decision_tree.py](model/decision_tree.py#L1-L200)
- K-Nearest Neighbors — [model/knn.py](model/knn.py#L1-L200)
- Gaussian Naive Bayes — [model/naive_bayes.py](model/naive_bayes.py#L1-L200)
- Random Forest (Ensemble) — [model/random_forest.py](model/random_forest.py#L1-L200)

The Streamlit app [app.py](app.py#L1-L400) exposes training and prediction functionality and uses the same preprocessing pipeline to prepare data for each model.

**Comparison table (evaluation metrics)**

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9195 | 0.9935 | 0.9201 | 0.9195 | 0.9197 | 0.9028 |
| Decision Tree | 0.8955 | 0.9357 | 0.8954 | 0.8955 | 0.8953 | 0.8737 |
| K-Nearest Neighbors | 0.9155 | 0.9811 | 0.9163 | 0.9155 | 0.9157 | 0.8978 |
| Gaussian Naive Bayes | 0.8970 | 0.9899 | 0.8997 | 0.8970 | 0.8972 | 0.8762 |
| Random Forest (Ensemble) | 0.9169 | 0.9905 | 0.9170 | 0.9169 | 0.9169 | 0.8995 |

Populate these values by running the training for each model (see instructions below). Each model script prints evaluation metrics; the Streamlit app displays accuracy, precision, recall and F1 after training.

**Observations about model performance**

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Strong overall performance with the highest F1 (0.9197) and very high AUC (0.9935). |
| Decision Tree | Lower performance than ensemble and linear models; more prone to overfitting (AUC 0.9357). |
| KNN | Competitive accuracy and F1 (0.9157), performs well with scaled features. |
| Naive Bayes | Good AUC (0.9899) but slightly lower overall F1 (0.8972) compared to top models. |
| Random Forest (Ensemble) | High and consistent metrics (F1 0.9169), strong AUC and MCC—good ensemble choice. |
| Overall Winner for my dataset? | Logistic Regression (best by F1: 0.9197) |

**How to run the app and reproduce metrics**

1. Create and activate a Python environment (recommended):

```powershell
python -m venv assignment_venv
.\\assignment_venv\Scripts\Activate.ps1   # PowerShell
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Place `Dry_Bean_Dataset.csv` in the project root (same folder as [app.py](app.py#L1)).

4. Run the Streamlit app and train models interactively:

```powershell
streamlit run app.py
```

5. Alternatively, run individual model scripts to reproduce printed metrics (these scripts use the same preprocessing pipeline):

```powershell
python model/logistic_regression.py
python model/decision_tree.py
python model/knn.py
python model/naive_bayes.py
python model/random_forest.py
```

Each script loads `Dry_Bean_Dataset.csv`, trains the model, prints evaluation metrics and returns the trained model objects when run as `__main__`.
 


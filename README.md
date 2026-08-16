# Dry Bean Classification — DryBeanPrediction_ML_2025ac05053

**Problem statement:** Predict the class of a dry bean sample from its measured morphological features. The goal is to compare several classic machine-learning classifiers on the same dataset and report evaluation metrics so the best performing model can be identified and used for prediction.

**Dataset description:**
- **Name:** Dry Bean Dataset (CSV expected as `Dry_Bean_Dataset.csv` in the project root).
- **Target column:** `Class` (categorical label identifying bean type).
- **Features:** Numerical morphological measurements (the repository code reads all columns except `Class` as input features). The preprocessing in [model/preprocess.py](model/preprocess.py#L1-L200) handles duplicate removal, label encoding, train/test splitting and standard scaling.
- **Where to get it:** If you do not already have the file, obtain the dataset (for example, from the original source or course materials) and place `Dry_Bean_Dataset.csv` in the same folder as [app.py](app.py#L1).

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
| Logistic Regression | TBD | TBD | TBD | TBD | TBD | TBD |
| Decision Tree | TBD | TBD | TBD | TBD | TBD | TBD |
| KNN | TBD | TBD | TBD | TBD | TBD | TBD |
| Naive Bayes | TBD | TBD | TBD | TBD | TBD | TBD |
| Random Forest (Ensemble) | TBD | TBD | TBD | TBD | TBD | TBD |

Populate these values by running the training for each model (see instructions below). Each model script prints evaluation metrics; the Streamlit app displays accuracy, precision, recall and F1 after training.

**Observations about model performance**

| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | TBD |
| Decision Tree | TBD |
| KNN | TBD |
| Naive Bayes | TBD |
| Random Forest (Ensemble) | TBD |
| Overall Winner for your dataset? | TBD |

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
 


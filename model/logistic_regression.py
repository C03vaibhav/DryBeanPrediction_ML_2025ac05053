"""
===========================================================
Logistic Regression Model
Dry Bean Classification
===========================================================

This module trains a Logistic Regression model using the
common preprocessing module.

When this module is imported, the model is automatically
trained and made available for prediction.

Logistic Regression requires feature scaling, so the scaled
training data is used.

===========================================================
"""

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
    classification_report
)

from model.preprocess import prepare_training_data


# ===========================================================
# Dataset Path
# ===========================================================

FILE_PATH = "Dry_Bean_Dataset.csv"


# ===========================================================
# Prepare Training Data
# ===========================================================

(
    X_train,
    X_test,
    X_train_scaled,
    X_test_scaled,
    y_train,
    y_test,
    label_encoder,
    scaler,
    feature_names
) = prepare_training_data(FILE_PATH)


# ===========================================================
# Create Logistic Regression Model
# ===========================================================

model = LogisticRegression(
    max_iter=1000,
    random_state=42
)


# ===========================================================
# Train the Model
# ===========================================================

model.fit(
    X_train_scaled,
    y_train
)


# ===========================================================
# Evaluate the Model
# ===========================================================

y_pred = model.predict(X_test_scaled)

y_prob = model.predict_proba(X_test_scaled)


accuracy = accuracy_score(
    y_test,
    y_pred
)

auc = roc_auc_score(
    y_test,
    y_prob,
    multi_class="ovr",
    average="weighted"
)

precision = precision_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

recall = recall_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

f1 = f1_score(
    y_test,
    y_pred,
    average="weighted",
    zero_division=0
)

mcc = matthews_corrcoef(
    y_test,
    y_pred
)


# ===========================================================
# Display Model Performance
# ===========================================================

print("\n==============================================")
print("Logistic Regression")
print("==============================================")

print(f"Accuracy  : {accuracy:.4f}")
print(f"AUC Score : {auc:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print(f"MCC Score : {mcc:.4f}")

print("\nClassification Report")
print("----------------------------------------------")

print(
    classification_report(
        y_test,
        y_pred,
        target_names=label_encoder.classes_,
        zero_division=0
    )
)


# ===========================================================
# Prediction Function
# ===========================================================

def predict(test_data):
    """
    Make predictions using the trained Logistic Regression
    model.

    Parameters
    ----------
    test_data : array-like
        Preprocessed and scaled test data.

    Returns
    -------
    predictions : numpy.ndarray
        Predicted encoded class labels.
    """

    return model.predict(test_data)
from sklearn.neighbors import KNeighborsClassifier
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
FILE_PATH = "Dry_Bean_Dataset.csv"
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

model = KNeighborsClassifier(
    n_neighbors=5
)

model.fit(
    X_train_scaled,
    y_train
)

y_pred = model.predict(
    X_test_scaled
)

y_prob = model.predict_proba(
    X_test_scaled
)


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

print("\n==============================================")
print("K-Nearest Neighbors")
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

def predict(test_data):
    return model.predict(test_data)
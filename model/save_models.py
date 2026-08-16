import sys
import os
import pathlib
from joblib import dump

ROOT = pathlib.Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from model.logistic_regression import train_model as train_logistic
from model.decision_tree import train_model as train_decision_tree
from model.knn import train_model as train_knn
from model.naive_bayes import train_model as train_naive_bayes
from model.random_forest import train_model as train_random_forest

DATASET = "Dry_Bean_Dataset.csv"
OUT_DIR = os.path.dirname(__file__)

models = {
    "logistic_regression": train_logistic,
    "decision_tree": train_decision_tree,
    "knn": train_knn,
    "naive_bayes": train_naive_bayes,
    "random_forest": train_random_forest,
}

for name, fn in models.items():
    print(f"Training {name}...")
    model, scaler, label_encoder, feature_names = fn(DATASET)

    base = os.path.join(OUT_DIR, name)
    artifact = {
        "model": model,
        "scaler": scaler,
        "label_encoder": label_encoder,
        "feature_names": feature_names,
    }
    dump(artifact, base + ".pkl")

    print(f"Saved artifact for {name} to {OUT_DIR}")

print("Done.")

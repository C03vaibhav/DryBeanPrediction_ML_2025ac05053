import streamlit as st
import pandas as pd
import os
import joblib
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    matthews_corrcoef,
)

from model.preprocess import (
    prepare_training_data,
    prepare_test_data
)

st.set_page_config(
    page_title="Dry Bean Classifier",
    page_icon="🌱",
    layout="wide"
)

DATASET_PATH = "Dry_Bean_Dataset.csv"

MODEL_OPTIONS = [
    "Logistic Regression",
    "Decision Tree",
    "K-Nearest Neighbors",
    "Gaussian Naive Bayes",
    "Random Forest"
]

# Map user-visible model names to artifact file prefixes saved in `model/`
ARTIFACT_MAP = {
    "Logistic Regression": "logistic_regression",
    "Decision Tree": "decision_tree",
    "K-Nearest Neighbors": "knn",
    "Gaussian Naive Bayes": "naive_bayes",
    "Random Forest": "random_forest"
}

SCALED_MODELS = [
    "Logistic Regression",
    "K-Nearest Neighbors",
    "Gaussian Naive Bayes"
]

@st.cache_data
def load_evaluation_data():
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
    ) = prepare_training_data(DATASET_PATH)

    return (
        X_test,
        X_test_scaled,
        y_test,
        label_encoder,
        feature_names,
        scaler
    )




def load_saved_artifacts(model_name):
    """Load saved artifacts (model, scaler, label_encoder, feature_names)
    from the `model/` directory based on `ARTIFACT_MAP`.
    Returns tuple or None on failure.
    """
    prefix = ARTIFACT_MAP.get(model_name)
    if not prefix:
        return None

    base = os.path.join("model", prefix)
    try:
        artifact = joblib.load(base + ".pkl")
        model = artifact.get("model")
        scaler = artifact.get("scaler")
        label_encoder = artifact.get("label_encoder")
        feature_names = artifact.get("feature_names")
        return model, scaler, label_encoder, feature_names
    except Exception:
        return None

@st.cache_data
def get_all_model_results():
    # Attempt to compute metrics from saved artifacts (preferred) so we
    # don't import and run model training at app import time.
    results = []

    X_test, X_test_scaled, y_test, _, _, _ = load_evaluation_data()

    for model_name in MODEL_OPTIONS:
        saved = load_saved_artifacts(model_name)

        if saved is not None:
            model_obj, scaler_obj, label_encoder_obj, feature_names_obj = saved

            use_scaled = model_name in SCALED_MODELS
            X_eval = X_test_scaled if use_scaled else X_test

            y_pred = model_obj.predict(X_eval)

            # AUC requires probabilities; handle absence gracefully
            try:
                y_prob = model_obj.predict_proba(X_eval)
                auc = roc_auc_score(y_test, y_prob, multi_class="ovr", average="weighted")
            except Exception:
                auc = float("nan")

            results.append(
                {
                    "Model": model_name,
                    "Accuracy": accuracy_score(y_test, y_pred),
                    "AUC": auc,
                    "Precision": precision_score(y_test, y_pred, average="weighted", zero_division=0),
                    "Recall": recall_score(y_test, y_pred, average="weighted", zero_division=0),
                    "F1 Score": f1_score(y_test, y_pred, average="weighted", zero_division=0),
                    "MCC": matthews_corrcoef(y_test, y_pred),
                }
            )

        else:
            # Saved artifacts missing — leave placeholders (app shows warning elsewhere)
            results.append(
                {
                    "Model": model_name,
                    "Accuracy": float("nan"),
                    "AUC": float("nan"),
                    "Precision": float("nan"),
                    "Recall": float("nan"),
                    "F1 Score": float("nan"),
                    "MCC": float("nan"),
                }
            )

    return pd.DataFrame(results)

st.title("Dry Bean Classification")
st.write("Select a trained machine learning model and upload Dry Bean test data to predict its class.")

# Sidebar
st.sidebar.header("Model Selection")
selected_model = st.sidebar.selectbox("Choose a model:", MODEL_OPTIONS)
st.sidebar.info(selected_model)

# Dataset Information
st.header("Dataset Information")
try:
    dataset = pd.read_csv(DATASET_PATH)
    col1, col2, col3 = st.columns(3)
    with col1: st.metric("Rows", dataset.shape[0])
    with col2: st.metric("Features", dataset.shape[1] - 1)
    with col3: st.metric("Classes", dataset["Class"].nunique())
    with st.expander("View Dataset"):
        st.dataframe(dataset.head(10), use_container_width=True)
except FileNotFoundError:
    st.error("Dry_Bean_Dataset.csv was not found.")
    st.stop()

# Selected Model: show artifact availability and instruction to upload test data for evaluation
saved_perf = load_saved_artifacts(selected_model)

if saved_perf is not None:
    perf_model, perf_scaler, perf_label_encoder, perf_feature_names = saved_perf
    st.info("Uplaod a dataset to see the evaluation metrics")
else:
    st.warning(
        f"Saved artifacts for '{selected_model}' not found in model/ — the app cannot run predictions or evaluation."
    )
    st.info("Generate artifacts by running `python -m model.save_models` or train and save a model.")

st.header("Upload Test Data")
uploaded_file = st.file_uploader("Upload a CSV file", type=["csv"])

if uploaded_file is not None:
    try:
        test_data = pd.read_csv(uploaded_file)
        st.success("Test data uploaded successfully!")

        # Show preview
        with st.expander("View Uploaded Test Data"):
            st.dataframe(test_data.head(10), use_container_width=True)

        # Decide if scaling is needed
        use_scaling = selected_model in SCALED_MODELS

        # Prefer saved .pkl artifacts (model + scaler + label encoder) if available
        saved = load_saved_artifacts(selected_model)

        if saved is not None:
            saved_model, saved_scaler, saved_label_encoder, saved_feature_names = saved
            st.info(f"Loaded saved artifacts for {selected_model} from model/ directory.")
            scaler_to_use = saved_scaler
            label_encoder_to_use = saved_label_encoder
            feature_names_to_use = saved_feature_names
            model_for_prediction = saved_model
        else:
            st.warning(
                f"Saved artifacts for '{selected_model}' not found — cannot run prediction."
            )
            st.info("Generate artifacts by running `python -m model.save_models` or train and save a model.")
            model_for_prediction = None

        # Predict with selected model (only if artifacts are available)
        if model_for_prediction is not None:
            # Preprocess uploaded test data (returns only X)
            X_new = prepare_test_data(
                test_data,
                feature_names_to_use,
                scaler=scaler_to_use,
                scale_data=use_scaling
            )
            y_pred_new = model_for_prediction.predict(X_new)

            # Decode numeric predictions back to original class names
            predicted_classes = label_encoder_to_use.inverse_transform(y_pred_new)

            st.header("Predictions on Uploaded Data")
            prediction_df = pd.DataFrame({
                "Predicted": predicted_classes
            })
            st.dataframe(prediction_df, use_container_width=True)

            # If ground-truth labels are provided in uploaded file, compute evaluation metrics
            if "Class" in test_data.columns:
                y_true_strings = test_data["Class"].values

                # Selected model metrics
                try:
                    # Accuracy/Precision/Recall/F1/MCC using string labels
                    accuracy = accuracy_score(y_true_strings, predicted_classes)
                    precision = precision_score(y_true_strings, predicted_classes, average="weighted", zero_division=0)
                    recall = recall_score(y_true_strings, predicted_classes, average="weighted", zero_division=0)
                    f1 = f1_score(y_true_strings, predicted_classes, average="weighted", zero_division=0)
                    mcc = matthews_corrcoef(y_true_strings, predicted_classes)

                    # AUC (requires probabilities and numeric encoding per-model)
                    try:
                        y_prob = model_for_prediction.predict_proba(X_new)
                        y_true_int = label_encoder_to_use.transform(y_true_strings)
                        auc = roc_auc_score(y_true_int, y_prob, multi_class="ovr", average="weighted")
                    except Exception:
                        auc = float("nan")

                    # Metric Cards
                    col1, col2, col3, col4, col5, col6 = st.columns(6)
                    with col1: st.metric("Accuracy", f"{accuracy:.4f}")
                    with col2: st.metric("AUC", f"{auc:.4f}")
                    with col3: st.metric("Precision", f"{precision:.4f}")
                    with col4: st.metric("Recall", f"{recall:.4f}")
                    with col5: st.metric("F1 Score", f"{f1:.4f}")
                    with col6: st.metric("MCC", f"{mcc:.4f}")

                    # Confusion Matrix
                    st.header("Confusion Matrix")
                    cm = confusion_matrix(y_true_strings, predicted_classes)
                    class_names = label_encoder_to_use.classes_
                    cm_df = pd.DataFrame(cm,
                        index=[f"Actual - {name}" for name in class_names],
                        columns=[f"Predicted - {name}" for name in class_names])
                    st.dataframe(cm_df, use_container_width=True)

                    # Classification Report
                    st.header("Classification Report")
                    report = classification_report(y_true_strings, predicted_classes, target_names=class_names, output_dict=True, zero_division=0)
                    report_df = pd.DataFrame(report).transpose()
                    st.dataframe(report_df, use_container_width=True)

                except Exception as metric_err:
                    st.error(f"Error computing evaluation metrics: {metric_err}")

                # Compute comparison across all saved models using the uploaded test set
                comparison_results = []
                for model_name in MODEL_OPTIONS:
                    saved = load_saved_artifacts(model_name)
                    if saved is None:
                        comparison_results.append({
                            "Model": model_name,
                            "Accuracy": float("nan"),
                            "AUC": float("nan"),
                            "Precision": float("nan"),
                            "Recall": float("nan"),
                            "F1 Score": float("nan"),
                            "MCC": float("nan"),
                        })
                        continue

                    model_obj, scaler_obj, label_encoder_obj, feature_names_obj = saved
                    use_scaled_local = model_name in SCALED_MODELS
                    # Prepare features for this model from uploaded test data
                    try:
                        X_eval_local = prepare_test_data(test_data, feature_names_obj, scaler=scaler_obj, scale_data=use_scaled_local)
                    except Exception as prep_err:
                        comparison_results.append({
                            "Model": model_name,
                            "Accuracy": float("nan"),
                            "AUC": float("nan"),
                            "Precision": float("nan"),
                            "Recall": float("nan"),
                            "F1 Score": float("nan"),
                            "MCC": float("nan"),
                        })
                        continue

                    y_pred_local = model_obj.predict(X_eval_local)
                    pred_classes_local = label_encoder_obj.inverse_transform(y_pred_local)

                    try:
                        y_prob_local = model_obj.predict_proba(X_eval_local)
                        y_true_int_local = label_encoder_obj.transform(test_data["Class"].values)
                        auc_local = roc_auc_score(y_true_int_local, y_prob_local, multi_class="ovr", average="weighted")
                    except Exception:
                        auc_local = float("nan")

                    comparison_results.append({
                        "Model": model_name,
                        "Accuracy": accuracy_score(test_data["Class"].values, pred_classes_local),
                        "AUC": auc_local,
                        "Precision": precision_score(test_data["Class"].values, pred_classes_local, average="weighted", zero_division=0),
                        "Recall": recall_score(test_data["Class"].values, pred_classes_local, average="weighted", zero_division=0),
                        "F1 Score": f1_score(test_data["Class"].values, pred_classes_local, average="weighted", zero_division=0),
                        "MCC": matthews_corrcoef(test_data["Class"].values, pred_classes_local),
                    })

                comp_df = pd.DataFrame(comparison_results)
                comp_display = comp_df.copy()
                comp_display[["Accuracy","AUC","Precision","Recall","F1 Score","MCC"]] = comp_display[["Accuracy","AUC","Precision","Recall","F1 Score","MCC"]].round(4)
                st.header("Model Comparison on Uploaded Test Set")
                st.dataframe(comp_display, use_container_width=True, hide_index=True)
                try:
                    best_idx = comp_df["F1 Score"].idxmax()
                    best_name = comp_df.loc[best_idx, "Model"]
                    best_f1 = comp_df.loc[best_idx, "F1 Score"]
                    st.success(f"Best performing model on uploaded test set (by F1): {best_name} ({best_f1:.4f})")
                except Exception:
                    pass
            else:
                st.info("Uploaded test file does not contain 'Class' column — evaluation metrics require ground-truth labels.")
        else:
            st.error("Prediction unavailable because model artifacts are missing.")

    except Exception as e:
        st.error(f"Error processing uploaded file: {e}")
else:
    st.info("Upload a CSV file to test the model on new data.")

# Footer
st.markdown("---")
st.caption("2025AC05053 | Cheruvu Tarun Rama Vaibhav | ML-Assignment-2")

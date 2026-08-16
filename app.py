import streamlit as st
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report

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

SCALED_MODELS = [
    "Logistic Regression",
    "K-Nearest Neighbors",
    "Gaussian Naive Bayes"
]

@st.cache_resource
def load_model(model_name):
    if model_name == "Logistic Regression":
        from model import logistic_regression
        return logistic_regression
    elif model_name == "Decision Tree":
        from model import decision_tree
        return decision_tree
    elif model_name == "K-Nearest Neighbors":
        from model import knn
        return knn
    elif model_name == "Gaussian Naive Bayes":
        from model import naive_bayes
        return naive_bayes
    elif model_name == "Random Forest":
        from model import random_forest
        return random_forest
    return None

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

def calculate_model_metrics(model_module):
    return (
        model_module.accuracy,
        model_module.precision,
        model_module.recall,
        model_module.f1,
        model_module.auc,
        model_module.mcc
    )

@st.cache_data
def get_all_model_results():
    results = []
    for model_name in MODEL_OPTIONS:
        model_module = load_model(model_name)
        results.append(
            {
                "Model": model_name,
                "Accuracy": model_module.accuracy,
                "AUC": model_module.auc,
                "Precision": model_module.precision,
                "Recall": model_module.recall,
                "F1 Score": model_module.f1,
                "MCC": model_module.mcc
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

# Load Selected Model
with st.spinner(f"Loading {selected_model}..."):
    model_module = load_model(selected_model)

# Load Evaluation Data
X_test, X_test_scaled, y_test, evaluation_label_encoder, feature_names, evaluation_scaler = load_evaluation_data()

# Selected Model Performance
st.header("Model Performance")
accuracy, precision, recall, f1, auc, mcc = calculate_model_metrics(model_module)
y_pred = model_module.predict(X_test_scaled if selected_model in SCALED_MODELS else X_test)

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
cm = confusion_matrix(y_test, y_pred)
class_names = evaluation_label_encoder.classes_
cm_df = pd.DataFrame(cm,
    index=[f"Actual - {name}" for name in class_names],
    columns=[f"Predicted - {name}" for name in class_names])
st.dataframe(cm_df, use_container_width=True)

# Classification Report
st.header("Classification Report")
report = classification_report(y_test, y_pred, target_names=class_names, output_dict=True, zero_division=0)
report_df = pd.DataFrame(report).transpose()
st.dataframe(report_df, use_container_width=True)

# Model Comparison
st.header("Model Comparison")
try:
    comparison_df = get_all_model_results()
    display_comparison = comparison_df.copy()
    display_comparison[["Accuracy","AUC","Precision","Recall","F1 Score","MCC"]] = display_comparison[["Accuracy","AUC","Precision","Recall","F1 Score","MCC"]].round(4)
    st.dataframe(display_comparison, use_container_width=True, hide_index=True)
    best_model_index = comparison_df["F1 Score"].idxmax()
    best_model = comparison_df.loc[best_model_index, "Model"]
    best_f1 = comparison_df.loc[best_model_index, "F1 Score"]
    st.success(f"Best performing model based on F1 Score: {best_model} ({best_f1:.4f})")
except Exception as error:
    st.warning(f"Unable to generate model comparison: {error}")

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

        # Preprocess uploaded test data (returns only X)
        X_new = prepare_test_data(
            test_data,
            feature_names,
            scaler=evaluation_scaler,
            scale_data=use_scaling
        )

        # Predict with selected model
        y_pred_new = model_module.predict(X_new)

        # Decode numeric predictions back to original class names
        predicted_classes = evaluation_label_encoder.inverse_transform(y_pred_new)

        st.header("Predictions on Uploaded Data")
        prediction_df = pd.DataFrame({
            "Predicted": predicted_classes
        })
        st.dataframe(prediction_df, use_container_width=True)

    except Exception as e:
        st.error(f"Error processing uploaded file: {e}")
else:
    st.info("Upload a CSV file to test the model on new data.")

# Footer
st.markdown("---")
st.caption("2025AC05053 | Cheruvu Tarun Rama Vaibhav | ML-Assignment-2")

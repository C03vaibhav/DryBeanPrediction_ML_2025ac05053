"""
===========================================================
Preprocessing Module
Dry Bean Classification
===========================================================

This module contains the common preprocessing steps used
by all machine learning models in the project.

The preprocessing includes:
    - Loading the dataset
    - Removing duplicate rows
    - Separating features and target
    - Encoding the target variable
    - Train-test splitting
    - Feature scaling
    - Preparing user-uploaded data for prediction

Models such as Logistic Regression, KNN and Naive Bayes
use the scaled data, while Decision Tree and Random Forest
use the unscaled data.

===========================================================
"""

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def prepare_training_data(
    file_path,
    test_size=0.2,
    random_state=42
):
    """
    Load the dataset and perform common preprocessing.

    Parameters
    ----------
    file_path : str
        Path to the dataset CSV file.

    test_size : float
        Proportion of data to be used for testing.

    random_state : int
        Random state used for reproducible splitting.

    Returns
    -------
    X_train : pandas.DataFrame
        Training features without scaling.

    X_test : pandas.DataFrame
        Testing features without scaling.

    X_train_scaled : numpy.ndarray
        Scaled training features.

    X_test_scaled : numpy.ndarray
        Scaled testing features.

    y_train : numpy.ndarray
        Encoded training target.

    y_test : numpy.ndarray
        Encoded testing target.

    label_encoder : LabelEncoder
        Encoder used to convert class names into numbers.

    scaler : StandardScaler
        Scaler fitted using the training data.

    feature_names : list
        Names of the input features.
    """

    # -------------------------------------------------------
    # Load dataset
    # -------------------------------------------------------

    df = pd.read_csv(file_path)

    # -------------------------------------------------------
    # Check for target column
    # -------------------------------------------------------

    if "Class" not in df.columns:

        raise ValueError(
            "The dataset must contain a 'Class' column."
        )

    # -------------------------------------------------------
    # Remove duplicate rows
    # -------------------------------------------------------

    df = df.drop_duplicates().reset_index(drop=True)

    # -------------------------------------------------------
    # Separate features and target
    # -------------------------------------------------------

    X = df.drop(
        "Class",
        axis=1
    )

    y = df["Class"]

    # -------------------------------------------------------
    # Check that all features are numeric
    # -------------------------------------------------------

    non_numeric_columns = X.select_dtypes(
        exclude="number"
    ).columns.tolist()

    if non_numeric_columns:

        raise ValueError(
            "The following feature columns are not numeric: "
            + ", ".join(non_numeric_columns)
        )

    # -------------------------------------------------------
    # Store feature names and their original order
    # -------------------------------------------------------

    feature_names = X.columns.tolist()

    # -------------------------------------------------------
    # Encode target labels
    # -------------------------------------------------------

    label_encoder = LabelEncoder()

    y = label_encoder.fit_transform(y)

    # -------------------------------------------------------
    # Train-test split
    # -------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    # -------------------------------------------------------
    # Feature scaling
    # -------------------------------------------------------

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

    # -------------------------------------------------------
    # Return all required preprocessing objects/data
    # -------------------------------------------------------

    return (
        X_train,
        X_test,
        X_train_scaled,
        X_test_scaled,
        y_train,
        y_test,
        label_encoder,
        scaler,
        feature_names
    )


def prepare_test_data(
    test_df,
    feature_names,
    scaler=None,
    scale_data=False
):
    """
    Prepare user-uploaded data for prediction.

    The uploaded CSV should contain exactly the same feature
    columns that were used during model training.

    Parameters
    ----------
    test_df : pandas.DataFrame
        User-uploaded test data.

    feature_names : list
        Feature names obtained from the training dataset.

    scaler : StandardScaler, optional
        Scaler fitted on the training data.

    scale_data : bool
        Whether the input data should be scaled.

    Returns
    -------
    pandas.DataFrame or numpy.ndarray
        Processed data ready for prediction.
    """

    # -------------------------------------------------------
    # Make a copy
    # -------------------------------------------------------

    X = test_df.copy()

    # -------------------------------------------------------
    # Remove target column if it exists
    # -------------------------------------------------------

    if "Class" in X.columns:

        X = X.drop(
            "Class",
            axis=1
        )

    # -------------------------------------------------------
    # Check for missing feature columns
    # -------------------------------------------------------

    missing_columns = [
        column
        for column in feature_names
        if column not in X.columns
    ]

    if missing_columns:

        raise ValueError(
            "The uploaded file is missing these columns: "
            + ", ".join(missing_columns)
        )

    # -------------------------------------------------------
    # Check for unexpected extra columns
    # -------------------------------------------------------

    extra_columns = [
        column
        for column in X.columns
        if column not in feature_names
    ]

    if extra_columns:

        raise ValueError(
            "The uploaded file contains unexpected columns: "
            + ", ".join(extra_columns)
        )

    # -------------------------------------------------------
    # Check that all feature values are numeric
    # -------------------------------------------------------

    non_numeric_columns = X.select_dtypes(
        exclude="number"
    ).columns.tolist()

    if non_numeric_columns:

        raise ValueError(
            "The following feature columns must contain "
            "numeric values: "
            + ", ".join(non_numeric_columns)
        )

    # -------------------------------------------------------
    # Arrange columns in the same order as training
    # -------------------------------------------------------

    X = X[feature_names]

    # -------------------------------------------------------
    # Apply scaling if required
    # -------------------------------------------------------

    if scale_data:

        if scaler is None:

            raise ValueError(
                "A fitted scaler is required when "
                "scale_data=True."
            )

        X = scaler.transform(X)

    # -------------------------------------------------------
    # Return processed data
    # -------------------------------------------------------

    return X
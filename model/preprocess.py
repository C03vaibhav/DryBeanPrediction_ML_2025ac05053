import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler


def prepare_training_data(
    file_path,
    test_size=0.2,
    random_state=42
):

    df = pd.read_csv(file_path)

    if "Class" not in df.columns:

        raise ValueError(
            "The dataset must contain a 'Class' column."
        )

    df = df.drop_duplicates().reset_index(drop=True)

    X = df.drop(
        "Class",
        axis=1
    )

    y = df["Class"]

    non_numeric_columns = X.select_dtypes(
        exclude="number"
    ).columns.tolist()

    if non_numeric_columns:

        raise ValueError(
            "The following feature columns are not numeric: "
            + ", ".join(non_numeric_columns)
        )
    feature_names = X.columns.tolist()

    label_encoder = LabelEncoder()

    y = label_encoder.fit_transform(y)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(
        X_train
    )

    X_test_scaled = scaler.transform(
        X_test
    )

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
    X = test_df.copy()

    if "Class" in X.columns:

        X = X.drop(
            "Class",
            axis=1
        )

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

    non_numeric_columns = X.select_dtypes(
        exclude="number"
    ).columns.tolist()

    if non_numeric_columns:

        raise ValueError(
            "The following feature columns must contain "
            "numeric values: "
            + ", ".join(non_numeric_columns)
        )

    X = X[feature_names]

    if scale_data:

        if scaler is None:

            raise ValueError(
                "A fitted scaler is required when "
                "scale_data=True."
            )

        X = scaler.transform(X)

    return X
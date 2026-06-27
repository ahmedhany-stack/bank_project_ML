import os
import pickle

import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import (
    OneHotEncoder,
    MinMaxScaler
)

from src.logger import logger


# ============================================
# CONFIG
# ============================================

TARGET = "deposit"

PREPROCESSOR_PATH = "models/preprocessor.pkl"

os.makedirs(
    "models",
    exist_ok=True
)


# ============================================
# BUILD PREPROCESSOR
# ============================================

def build_preprocessor(
    X: pd.DataFrame
) -> ColumnTransformer:

    logger.info(
        "Building preprocessing pipeline"
    )

    categorical_columns = X.select_dtypes(
        include=["object", "category"]
    ).columns.tolist()

    numerical_columns = X.select_dtypes(
        exclude=["object", "category"]
    ).columns.tolist()

    logger.info(
        f"Categorical Columns : {categorical_columns}"
    )

    logger.info(
        f"Numerical Columns : {numerical_columns}"
    )

    preprocessor = ColumnTransformer(

        transformers=[

            (

                "categorical",

                OneHotEncoder(

                    handle_unknown="ignore",

                    sparse_output=False

                ),

                categorical_columns

            ),

            (

                "numerical",

                MinMaxScaler(),

                numerical_columns

            )

        ],

        remainder="drop"

    )

    return preprocessor


# ============================================
# FIT PREPROCESSOR
# ============================================

def fit_preprocessor(
    X: pd.DataFrame
):

    logger.info(
        "Fitting preprocessing pipeline"
    )

    preprocessor = build_preprocessor(X)

    preprocessor.fit(X)

    with open(
        PREPROCESSOR_PATH,
        "wb"
    ) as f:

        pickle.dump(
            preprocessor,
            f
        )

    logger.info(
        "Preprocessor saved successfully"
    )

    return preprocessor


# ============================================
# LOAD PREPROCESSOR
# ============================================

def load_preprocessor():

    if not os.path.exists(
        PREPROCESSOR_PATH
    ):

        raise FileNotFoundError(
            "Preprocessor not found."
        )

    with open(
        PREPROCESSOR_PATH,
        "rb"
    ) as f:

        return pickle.load(f)


# ============================================
# TRANSFORM TRAIN DATA
# ============================================

def transform_train_data(
    df: pd.DataFrame
):

    logger.info(
        "Transforming training data"
    )

    X = df.drop(
        columns=[TARGET]
    )

    y = df[TARGET]

    preprocessor = fit_preprocessor(X)

    X_processed = preprocessor.transform(X)

    feature_names = preprocessor.get_feature_names_out()

    X_processed = pd.DataFrame(

        X_processed,

        columns=feature_names,

        index=X.index

    )

    logger.info(
        f"Processed Shape : {X_processed.shape}"
    )

    return X_processed, y


# ============================================
# TRANSFORM PREDICTION DATA
# ============================================

def transform_prediction_data(
    df: pd.DataFrame
):

    logger.info(
        "Transforming prediction data"
    )

    preprocessor = load_preprocessor()

    X_processed = preprocessor.transform(df)

    feature_names = preprocessor.get_feature_names_out()

    X_processed = pd.DataFrame(

        X_processed,

        columns=feature_names,

        index=df.index

    )

    return X_processed


if __name__ == "__main__":

    df = pd.read_csv("bank_data_transformed.csv")

    df["deposit"] = df["deposit"].map({
        "no": 0,
        "yes": 1
    })

    from feature_engineering import create_features

    df = create_features(df)

    X, y = transform_train_data(df)

    print(X.shape)
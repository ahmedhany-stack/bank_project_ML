import pandas as pd
import logging
import os

from sklearn.preprocessing import (
    LabelEncoder,
    OneHotEncoder,
    MinMaxScaler
)

# ===================================
# CONFIG
# ===================================

INPUT_FILE = "bank_data_transformed.csv"
OUTPUT_FILE = "bank_data_encoded.csv"

LOG_DIR = "logs"
LOG_FILE = os.path.join(
    LOG_DIR,
    "feature_engineering_pipeline.log"
)

os.makedirs(
    LOG_DIR,
    exist_ok=True
)

# ===================================
# LOGGING
# ===================================

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(name)s | %(message)s"
)

file_handler = logging.FileHandler(
    LOG_FILE,
    mode="a"
)

file_handler.setFormatter(
    formatter
)

console_handler = logging.StreamHandler()

console_handler.setFormatter(
    formatter
)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

# ===================================
# LOAD DATA
# ===================================

def load_data(
    file_path: str
) -> pd.DataFrame:

    logger.info(
        f"Loading dataset from {file_path}"
    )

    df = pd.read_csv(file_path)

    logger.info(
        f"Dataset loaded successfully | Shape={df.shape}"
    )

    return df


# ===================================
# VALIDATE DATA
# ===================================

def validate_dataframe(
    df: pd.DataFrame
) -> None:

    if df is None:
        raise ValueError(
            "DataFrame is None"
        )

    if df.empty:
        raise ValueError(
            "DataFrame is empty"
        )

    logger.info(
        "Data validation passed"
    )


# ===================================
# ENCODE CATEGORICAL FEATURES
# ===================================

def encode_categorical_features(
    df: pd.DataFrame
) -> pd.DataFrame:

    logger.info(
        "Starting categorical encoding"
    )

    label_encoder = LabelEncoder()
    onehot_encoder = OneHotEncoder(
        sparse_output=False,
        handle_unknown="ignore"
    )

    categorical_cols = df.select_dtypes(
        include=["object"]
    ).columns.tolist()

    logger.info(
        f"Found {len(categorical_cols)} categorical columns"
    )

    for col in categorical_cols:

        unique_values = df[col].nunique()

        if unique_values == 2:

            df[col] = label_encoder.fit_transform(
                df[col]
            )

            logger.info(
                f"{col} -> Label Encoded"
            )

        else:

            encoded_data = onehot_encoder.fit_transform(
                df[[col]]
            )

            encoded_df = pd.DataFrame(
                encoded_data,
                columns=[
                    f"{col}_{category}"
                    for category in onehot_encoder.categories_[0]
                ],
                index=df.index
            )

            df = pd.concat(
                [
                    df.drop(
                        columns=[col]
                    ),
                    encoded_df
                ],
                axis=1
            )

            logger.info(
                f"{col} -> One Hot Encoded"
            )

    logger.info(
        "Categorical encoding completed"
    )

    return df


# ===================================
# SCALE FEATURES
# ===================================

def scale_features(
    df: pd.DataFrame
) -> pd.DataFrame:

    logger.info(
        "Starting feature scaling"
    )

    target = df["deposit"]

    features = df.drop(
        columns=["deposit"]
    )

    scaler = MinMaxScaler()

    scaled_features = scaler.fit_transform(
        features
    )

    scaled_df = pd.DataFrame(
        scaled_features,
        columns=features.columns,
        index=df.index
    )

    final_df = pd.concat(
        [
            scaled_df,
            target
        ],
        axis=1
    )

    logger.info(
        "Feature scaling completed"
    )

    return final_df


# ===================================
# SAVE DATA
# ===================================

def save_dataset(
    df: pd.DataFrame,
    filename: str
) -> None:

    df.to_csv(
        filename,
        index=False
    )

    logger.info(
        f"Dataset saved -> {filename}"
    )


# ===================================
# MAIN PIPELINE
# ===================================

def main():

    try:

        logger.info("=" * 60)
        logger.info(
            "FEATURE ENGINEERING PIPELINE STARTED"
        )
        logger.info("=" * 60)

        df = load_data(
            INPUT_FILE
        )

        validate_dataframe(
            df
        )

        df = encode_categorical_features(
            df
        )

        final_df = scale_features(
            df
        )

        save_dataset(
            final_df,
            OUTPUT_FILE
        )

        logger.info(
            f"Final dataset shape: {final_df.shape}"
        )

        logger.info(
            "Pipeline completed successfully"
        )

    except Exception as e:

        logger.exception(
            f"Pipeline failed: {e}"
        )

        raise

    finally:

        logger.info(
            "Pipeline execution finished"
        )


if __name__ == "__main__":
    main()
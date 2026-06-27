import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import logging
from src.logger import logger
from src.config import config
from sklearn.preprocessing import PowerTransformer

# =====================================
# CONFIG
# =====================================

INPUT_FILE = "cleaned_bank_data_from_outliers.csv"
OUTPUT_FILE = "bank_data_transformed.csv"
LOG_FILE = "logs/pipeline.log"

# =====================================
# LOGGING
# =====================================

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s"
)

file_handler = logging.FileHandler(LOG_FILE)
file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)


# =====================================
# LOAD DATA
# =====================================

def load_data(file_path):

    logger.info(f"Loading data from {file_path}")

    df = pd.read_csv(file_path)

    logger.info(
        f"Dataset loaded successfully | Shape={df.shape}"
    )

    return df


# =====================================
# VALIDATE DATA
# =====================================

def validate_data(df):

    if df is None or df.empty:
        raise ValueError(
            "DataFrame is empty or None"
        )

    logger.info("Validation passed")


# =====================================
# NUMERIC COLUMNS
# =====================================

def get_numeric_columns(df):

    numerical_cols = df.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    if "deposit" in numerical_cols:
        numerical_cols.remove("deposit")

    logger.info(
        f"Numeric columns selected: {len(numerical_cols)}"
    )

    return numerical_cols


# =====================================
# VISUALIZATION
# =====================================

def plot_distribution(
    before_data,
    after_data,
    column_name
):

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(12, 4)
    )

    sns.histplot(
        before_data,
        kde=True,
        ax=axes[0]
    )

    axes[0].set_title(
        f"{column_name} - Before"
    )

    sns.histplot(
        after_data,
        kde=True,
        ax=axes[1]
    )

    axes[1].set_title(
        f"{column_name} - After"
    )

    plt.tight_layout()
    plt.show()


# =====================================
# POWER TRANSFORMATION
# =====================================

def apply_power_transformation(
    df,
    numerical_cols
):

    logger.info(
        "Starting Power Transformation"
    )

    transformer = PowerTransformer(
        method="yeo-johnson"
    )

    transformed_count = 0

    for col in numerical_cols:

        if df[col].isnull().sum() > 0:

            logger.warning(
                f"{col} skipped because of missing values"
            )

            continue

        original_data = df[col].copy()

        df[col] = transformer.fit_transform(
            df[[col]]
        )

        transformed_count += 1

        logger.info(
            f"{col} transformed successfully"
        )

        plot_distribution(
            original_data,
            df[col],
            col
        )

    logger.info(
        f"Transformation completed | Columns transformed={transformed_count}"
    )

    return df


# =====================================
# SAVE DATA
# =====================================

def save_data(df, file_path):

    df.to_csv(
        file_path,
        index=False
    )

    logger.info(
        f"Dataset saved to {file_path}"
    )


# =====================================
# MAIN PIPELINE
# =====================================

def main():

    try:

        logger.info(
            "Pipeline started"
        )

        df = load_data(INPUT_FILE)

        validate_data(df)

        numerical_cols = get_numeric_columns(df)

        df = apply_power_transformation(
            df,
            numerical_cols
        )

        save_data(
            df,
            OUTPUT_FILE
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
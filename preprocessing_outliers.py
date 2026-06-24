from discover_data import df1
import matplotlib.pyplot as plt
import pandas as pd
import logging
import os

# ===================================
# CONFIG
# ===================================

OUTPUT_FILE = "cleaned_bank_data.csv"

LOG_DIR = "logs"
LOG_FILE = os.path.join(
    LOG_DIR,
    "outlier_pipeline.log"
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
# VALIDATION
# ===================================

def validate_dataframe(df: pd.DataFrame) -> None:
    """
    Validate input dataframe.
    """

    if df is None:
        raise ValueError("DataFrame is None")

    if df.empty:
        raise ValueError("DataFrame is empty")

    logger.info(
        f"Validation passed | Shape={df.shape}"
    )


# ===================================
# GET NUMERIC COLUMNS
# ===================================

def get_numeric_columns(
    df: pd.DataFrame
) -> list:

    numeric_cols = df.select_dtypes(
        include=["int64", "float64"]
    ).columns.tolist()

    logger.info(
        f"Found {len(numeric_cols)} numeric columns"
    )

    return numeric_cols


# ===================================
# GET COLUMNS TO CAP
# ===================================

def get_columns_to_cap(
    numeric_cols: list
) -> list:

    skip_cols = [
        "age",
        "balance",
        "duration",
        "pdays",
        "previous"
    ]

    cols_to_cap = [
        col
        for col in numeric_cols
        if col not in skip_cols
    ]

    logger.info(
        f"Columns selected for capping: {cols_to_cap}"
    )

    return cols_to_cap


# ===================================
# CAP OUTLIERS
# ===================================

def cap_outliers(
    df: pd.DataFrame,
    cols_to_cap: list
) -> pd.DataFrame:

    logger.info(
        "Starting outlier capping process"
    )

    df_clean = df.copy()

    total_outliers = 0

    for col in cols_to_cap:

        q1 = df_clean[col].quantile(0.25)
        q3 = df_clean[col].quantile(0.75)

        iqr = q3 - q1

        lower = q1 - (1.5 * iqr)
        upper = q3 + (1.5 * iqr)

        outliers_count = (
            (df_clean[col] < lower)
            |
            (df_clean[col] > upper)
        ).sum()

        df_clean[col] = df_clean[col].clip(
            lower=lower,
            upper=upper
        )

        total_outliers += outliers_count

        logger.info(
            f"{col} -> capped {outliers_count} outliers"
        )

    logger.info(
        f"Outlier capping completed | Total capped values={total_outliers}"
    )

    return df_clean


# ===================================
# SAVE DATASET
# ===================================

def save_dataset(
    df: pd.DataFrame,
    filename: str = OUTPUT_FILE
) -> None:

    df.to_csv(
        filename,
        index=False
    )

    logger.info(
        f"Dataset saved successfully -> {filename}"
    )


# ===================================
# PRINT SUMMARY
# ===================================

def print_summary(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    column: str = "previous"
) -> None:

    print("\n" + "=" * 60)
    print("DATASET SHAPE")
    print("=" * 60)

    print(f"Before : {df_before.shape}")
    print(f"After  : {df_after.shape}")

    print("\n" + "=" * 60)
    print(f"{column.upper()} SUMMARY")
    print("=" * 60)

    print("Before:")

    print(
        df_before[column].describe()
    )

    print("\nAfter:")

    print(
        df_after[column].describe()
    )


# ===================================
# VISUALIZATION
# ===================================

def plot_boxplots(
    df_before: pd.DataFrame,
    df_after: pd.DataFrame,
    numeric_cols: list
) -> None:

    logger.info(
        "Generating boxplots"
    )

    for col in numeric_cols:

        fig, axes = plt.subplots(
            1,
            2,
            figsize=(10, 4)
        )

        axes[0].boxplot(
            df_before[col]
        )

        axes[0].set_title(
            f"Before - {col}"
        )

        axes[1].boxplot(
            df_after[col]
        )

        axes[1].set_title(
            f"After - {col}"
        )

        plt.tight_layout()
        plt.show()

    logger.info(
        "Visualization completed"
    )


# ===================================
# MAIN PIPELINE
# ===================================

def main():

    try:

        logger.info("=" * 60)
        logger.info(
            "OUTLIER CAPPING PIPELINE STARTED"
        )
        logger.info("=" * 60)

        validate_dataframe(df1)

        df = df1.copy()

        numeric_cols = get_numeric_columns(
            df
        )

        cols_to_cap = get_columns_to_cap(
            numeric_cols
        )

        df_clean = cap_outliers(
            df,
            cols_to_cap
        )

        save_dataset(
            df_clean
        )

        print_summary(
            df,
            df_clean
        )

        plot_boxplots(
            df,
            df_clean,
            numeric_cols
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
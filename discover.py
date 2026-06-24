import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# ==================================================
# CONFIG
# ==================================================

DATA_FILE = "bank-full.csv"

REPORTS_DIR = "reports"
PLOTS_DIR = "plots"
LOGS_DIR = "logs"

REPORT_FILE = os.path.join(
    REPORTS_DIR,
    "eda_report.txt"
)

LOG_FILE = os.path.join(
    LOGS_DIR,
    "eda_pipeline.log"
)

# ==================================================
# CREATE DIRECTORIES
# ==================================================

os.makedirs(REPORTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

os.makedirs(
    os.path.join(PLOTS_DIR, "boxplots"),
    exist_ok=True
)

os.makedirs(
    os.path.join(PLOTS_DIR, "target_numeric"),
    exist_ok=True
)

os.makedirs(
    os.path.join(PLOTS_DIR, "target_categorical"),
    exist_ok=True
)

# ==================================================
# LOGGING
# ==================================================

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

formatter = logging.Formatter(
    "%(asctime)s | %(levelname)s | %(message)s"
)

file_handler = logging.FileHandler(
    LOG_FILE,
    mode="a",
    encoding="utf-8"
)

file_handler.setFormatter(formatter)

console_handler = logging.StreamHandler()
console_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(console_handler)

# ==================================================
# LOAD DATA
# ==================================================

def load_data():

    logger.info("Loading dataset...")

    df = pd.read_csv(
        DATA_FILE,
        sep=";"
    )

    df.rename(
        columns={
            "y": "deposit",
            "default": "default_flag"
        },
        inplace=True
    )

    logger.info(
        f"Dataset loaded successfully | Shape={df.shape}"
    )

    return df

# ==================================================
# GENERATE REPORT
# ==================================================

def generate_report(df):

    logger.info(
        "Generating EDA report..."
    )

    report_lines = []

    report_lines.append("=" * 80)
    report_lines.append("DATASET OVERVIEW")
    report_lines.append("=" * 80)
    report_lines.append(f"Shape: {df.shape}")

    report_lines.append("\n")
    report_lines.append("=" * 80)
    report_lines.append("DATA TYPES")
    report_lines.append("=" * 80)
    report_lines.append(str(df.dtypes))

    report_lines.append("\n")
    report_lines.append("=" * 80)
    report_lines.append("MISSING VALUES")
    report_lines.append("=" * 80)
    report_lines.append(str(df.isnull().sum()))

    report_lines.append("\n")
    report_lines.append("=" * 80)
    report_lines.append("MISSING VALUES (%)")
    report_lines.append("=" * 80)
    report_lines.append(
        str(
            (
                df.isnull().sum()
                / len(df)
            ) * 100
        )
    )

    report_lines.append("\n")
    report_lines.append("=" * 80)
    report_lines.append("DUPLICATES")
    report_lines.append("=" * 80)
    report_lines.append(
        str(
            df.duplicated().sum()
        )
    )

    report_lines.append("\n")
    report_lines.append("=" * 80)
    report_lines.append("NUMERICAL SUMMARY")
    report_lines.append("=" * 80)
    report_lines.append(
        str(
            df.describe()
        )
    )

    report_lines.append("\n")
    report_lines.append("=" * 80)
    report_lines.append("TARGET DISTRIBUTION")
    report_lines.append("=" * 80)
    report_lines.append(
        str(
            df["deposit"].value_counts()
        )
    )

    report_lines.append("\n")
    report_lines.append(
        str(
            df["deposit"]
            .value_counts(normalize=True)
            * 100
        )
    )

    report_lines.append("\n")
    report_lines.append("=" * 80)
    report_lines.append("UNIQUE VALUES")
    report_lines.append("=" * 80)

    for col in df.columns:

        report_lines.append(
            f"{col}: {df[col].nunique()}"
        )

    categorical_cols = df.select_dtypes(
        include="object"
    ).columns

    report_lines.append("\n")
    report_lines.append("=" * 80)
    report_lines.append("CATEGORICAL FEATURES")
    report_lines.append("=" * 80)

    for col in categorical_cols:

        report_lines.append("\n")
        report_lines.append(
            f"Column: {col}"
        )

        report_lines.append(
            str(
                df[col]
                .value_counts()
            )
        )

    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "\n".join(report_lines)
        )

    logger.info(
        f"EDA report saved -> {REPORT_FILE}"
    )

# ==================================================
# CORRELATION HEATMAP
# ==================================================

def save_correlation_heatmap(df):

    logger.info(
        "Saving correlation heatmap..."
    )

    numeric_df = df.select_dtypes(
        include=np.number
    )

    corr_matrix = numeric_df.corr()

    plt.figure(
        figsize=(12, 8)
    )

    sns.heatmap(
        corr_matrix,
        annot=True,
        cmap="coolwarm"
    )

    plt.title(
        "Correlation Matrix"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "correlation_heatmap.png"
        )
    )

    plt.close()

# ==================================================
# HISTOGRAMS
# ==================================================

def save_histograms(df):

    logger.info(
        "Saving histograms..."
    )

    numeric_df = df.select_dtypes(
        include=np.number
    )

    numeric_df.hist(
        figsize=(15, 10),
        bins=20
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            PLOTS_DIR,
            "histograms.png"
        )
    )

    plt.close()

# ==================================================
# BOXPLOTS
# ==================================================

def save_boxplots(df):

    logger.info(
        "Saving boxplots..."
    )

    numeric_df = df.select_dtypes(
        include=np.number
    )

    for col in numeric_df.columns:

        plt.figure(
            figsize=(6, 4)
        )

        plt.boxplot(
            df[col]
        )

        plt.title(
            f"Boxplot - {col}"
        )

        plt.savefig(
            os.path.join(
                PLOTS_DIR,
                "boxplots",
                f"{col}.png"
            )
        )

        plt.close()

# ==================================================
# TARGET VS NUMERIC
# ==================================================

def save_target_vs_numeric(df):

    logger.info(
        "Saving target vs numeric plots..."
    )

    numeric_df = df.select_dtypes(
        include=np.number
    )

    for col in numeric_df.columns:

        plt.figure(
            figsize=(7, 4)
        )

        df.boxplot(
            column=col,
            by="deposit"
        )

        plt.title(
            f"{col} vs Deposit"
        )

        plt.suptitle("")

        plt.savefig(
            os.path.join(
                PLOTS_DIR,
                "target_numeric",
                f"{col}.png"
            )
        )

        plt.close()

# ==================================================
# TARGET VS CATEGORICAL
# ==================================================

def save_target_vs_categorical(df):

    logger.info(
        "Saving target vs categorical plots..."
    )

    categorical_cols = df.select_dtypes(
        include="object"
    ).columns

    for col in categorical_cols:

        pd.crosstab(
            df[col],
            df["deposit"]
        ).plot(
            kind="bar",
            figsize=(10, 5)
        )

        plt.title(
            f"{col} vs Deposit"
        )

        plt.tight_layout()

        plt.savefig(
            os.path.join(
                PLOTS_DIR,
                "target_categorical",
                f"{col}.png"
            )
        )

        plt.close()

# ==================================================
# MAIN
# ==================================================

def main():

    try:

        logger.info("=" * 80)
        logger.info(
            "EDA PIPELINE STARTED"
        )
        logger.info("=" * 80)

        df = load_data()

        generate_report(df)

        save_correlation_heatmap(df)

        save_histograms(df)

        save_boxplots(df)

        save_target_vs_numeric(df)

        save_target_vs_categorical(df)

        logger.info(
            "EDA PIPELINE COMPLETED SUCCESSFULLY"
        )

        return df

    except Exception as e:

        logger.exception(
            f"Pipeline failed: {e}"
        )

        raise

    finally:

        logger.info(
            "Pipeline execution finished"
        )

# ==================================================
# RUN
# ==================================================

if __name__ == "__main__":

    df1 = main()
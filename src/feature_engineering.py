import numpy as np
import pandas as pd

from src.logger import logger


# ==========================================
# CONSTANTS
# ==========================================

BALANCE_HIGH_THRESHOLD = 3000
DURATION_HIGH_THRESHOLD = 500
CAMPAIGN_HIGH_THRESHOLD = 5


# ==========================================
# SAFE LOG FUNCTION
# ==========================================

def safe_log(series: pd.Series) -> pd.Series:
    """
    Apply log1p safely.
    Any value < -1 becomes -1 then log1p(-1)=0 after clipping.
    Missing values become 0.
    """

    series = pd.to_numeric(series, errors="coerce")

    series = series.fillna(0)

    series = np.clip(series, -0.999999, None)

    return np.log1p(series)


# ==========================================
# SAFE BINARY FEATURE
# ==========================================

def safe_binary(condition):

    return condition.fillna(False).astype(int)


# ==========================================
# FEATURE ENGINEERING
# ==========================================

def create_features(
    df: pd.DataFrame
) -> pd.DataFrame:

    logger.info("=" * 60)
    logger.info("Starting Feature Engineering")
    logger.info("=" * 60)

    df = df.copy()

    # =====================================
    # Fill Missing Numeric Values
    # =====================================

    numeric_columns = [
        "age",
        "balance",
        "campaign",
        "previous",
        "duration",
        "pdays"
    ]

    for col in numeric_columns:

        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

        df[col] = df[col].fillna(
            df[col].median()
        )

    # =====================================
    # Log Features
    # =====================================

    df["balance_log"] = safe_log(
        df["balance"]
    )

    df["campaign_log"] = safe_log(
        df["campaign"]
    )

    df["previous_log"] = safe_log(
        df["previous"]
    )

    df["duration_log"] = safe_log(
        df["duration"]
    )

    # =====================================
    # Boolean Features
    # =====================================

    df["balance_high"] = safe_binary(
        df["balance"] >= BALANCE_HIGH_THRESHOLD
    )

    df["duration_high"] = safe_binary(
        df["duration"] >= DURATION_HIGH_THRESHOLD
    )

    df["campaign_high"] = safe_binary(
        df["campaign"] >= CAMPAIGN_HIGH_THRESHOLD
    )

    df["pdays_contacted"] = safe_binary(
        df["pdays"] != -1
    )

    # =====================================
    # Age Group
    # =====================================

    df["age"] = df["age"].clip(
        lower=0,
        upper=120
    )

    age_group = pd.cut(

        df["age"],

        bins=[0, 25, 40, 60, 120],

        labels=[0, 1, 2, 3],

        include_lowest=True

    )

    df["age_group"] = (

        age_group

        .cat.add_categories([-1])

        .fillna(-1)

        .astype(int)

    )

    logger.info(
        f"Feature Engineering Finished | Shape={df.shape}"
    )

    return df
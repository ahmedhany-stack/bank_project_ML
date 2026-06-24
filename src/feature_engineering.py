import numpy as np

from logger import logger

def create_features(df):

    logger.info(
        "Starting feature engineering"
    )

    df = df.copy()

    df["balance_log"] = np.log1p(
        df["balance"]
    )

    df["balance_high"] = (
        df["balance"] >
        df["balance"].quantile(0.9)
    ).astype(int)

    df["campaign_log"] = np.log1p(
        df["campaign"]
    )

    df["campaign_high"] = (
        df["campaign"] > 5
    ).astype(int)

    df["pdays_contacted"] = (
        df["pdays"] != -1
    ).astype(int)

    df["previous_log"] = np.log1p(
        df["previous"]
    )

    df["duration_log"] = np.log1p(
        df["duration"]
    )

    df["duration_high"] = (
        df["duration"] >
        df["duration"].quantile(0.9)
    ).astype(int)

    df["age_group"] = np.select(
        [
            df["age"] <= 25,
            (df["age"] > 25) &
            (df["age"] <= 40),
            (df["age"] > 40) &
            (df["age"] <= 60),
            df["age"] > 60
        ],
        [0, 1, 2, 3],
        default=0
    )

    logger.info(
        "Feature engineering completed"
    )

    return df
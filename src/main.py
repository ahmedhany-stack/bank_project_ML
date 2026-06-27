import os
import pandas as pd

from src.logger import logger
from src.feature_engineering import create_features
from src.encoding import transform_train_data
from src.data_split import split_data
from src.model import (
    calculate_class_weight,
    build_model
)
from src.train import (
    train_model,
    save_model
)
from src.evaluation import (
    find_best_threshold,
    evaluate
)


def main():

    try:

        logger.info("=" * 60)
        logger.info("PIPELINE STARTED")
        logger.info("=" * 60)

        # =====================================
        # LOAD DATA
        # =====================================

        BASE_DIR = os.path.dirname(
            os.path.abspath(__file__)
        )

        file_path = os.path.join(
            BASE_DIR,
            "bank_data_transformed.csv"
        )

        df = pd.read_csv(file_path)

        logger.info(
            f"Dataset Shape : {df.shape}"
        )

        # =====================================
        # ENCODE TARGET
        # =====================================

        df["deposit"] = (
            df["deposit"]
            .map({
                "no": 0,
                "yes": 1
            })
            .astype(int)
        )

        logger.info(
            f"\nTarget Distribution\n{df['deposit'].value_counts()}"
        )

        # =====================================
        # FEATURE ENGINEERING
        # =====================================

        df = create_features(df)

        # =====================================
        # ENCODING + SCALING
        # =====================================

        X, y = transform_train_data(df)

        # =====================================
        # TRAIN TEST SPLIT
        # =====================================

        x_train, x_test, y_train, y_test = split_data(
            X,
            y
        )

        logger.info(
            f"Train Shape : {x_train.shape}"
        )

        logger.info(
            f"Test Shape : {x_test.shape}"
        )

        # =====================================
        # BUILD MODEL
        # =====================================

        scale_pos_weight = calculate_class_weight(
            y_train
        )

        logger.info(
            f"scale_pos_weight = {scale_pos_weight}"
        )

        model = build_model(
            scale_pos_weight
        )

        # =====================================
        # TRAIN MODEL
        # =====================================

        model = train_model(
            model,
            x_train,
            y_train
        )

        # =====================================
        # PREDICTIONS
        # =====================================

        probs = model.predict_proba(
            x_test
        )[:, 1]

        threshold = find_best_threshold(
            probs,
            y_test
        )

        predictions = (
            probs >= threshold
        ).astype(int)

        # =====================================
        # EVALUATION
        # =====================================

        evaluate(
            y_test,
            predictions
        )

        # =====================================
        # SAVE MODEL
        # =====================================

        save_model(
            model
        )

        logger.info("=" * 60)
        logger.info("PIPELINE FINISHED")
        logger.info("=" * 60)

    except Exception as e:

        logger.exception(e)

        raise


if __name__ == "__main__":
    main()
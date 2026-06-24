
import pandas as pd

from logger import logger

final_df = pd.read_csv("bank_data_encoded.csv")

from feature_engineering import (
    create_features
)

from data_split import (
    split_data
)

from model import (
    calculate_class_weight,
    build_model
)

from train import (
    train_model,
    save_model
)

from evaluation import (
    find_best_threshold,
    evaluate
)


def main():

    try:

        logger.info(
            "Pipeline Started"
        )

        df = create_features(
            final_df
        )

        (
            x_train,
            x_test,
            y_train,
            y_test
        ) = split_data(df)

        scale_pos_weight = (
            calculate_class_weight(
                y_train
            )
        )

        model = build_model(
            scale_pos_weight
        )

        model = train_model(
            model,
            x_train,
            y_train
        )

        probs = model.predict_proba(
            x_test
        )[:, 1]

        threshold = (
            find_best_threshold(
                probs,
                y_test
            )
        )

        y_pred = (
            probs >= threshold
        ).astype(int)

        evaluate(
            y_test,
            y_pred
        )

        save_model(
            model
        )

        logger.info(
            "Pipeline Finished"
        )

    except Exception as e:

        logger.exception(e)


if __name__ == "__main__":
    main()
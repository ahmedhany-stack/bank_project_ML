import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from logger import logger


def find_best_threshold(
    probs,
    y_test
):

    best_threshold = 0.5
    best_recall = 0

    for t in np.arange(
        0.10,
        0.60,
        0.01
    ):

        preds = (
            probs >= t
        ).astype(int)

        recall = recall_score(
            y_test,
            preds
        )

        if recall > best_recall:

            best_recall = recall
            best_threshold = t

    return best_threshold


def evaluate(
    y_test,
    y_pred
):

    logger.info(
        f"Accuracy={accuracy_score(y_test,y_pred):.4f}"
    )

    logger.info(
        f"Precision={precision_score(y_test,y_pred):.4f}"
    )

    logger.info(
        f"Recall={recall_score(y_test,y_pred):.4f}"
    )

    logger.info(
        f"F1={f1_score(y_test,y_pred):.4f}"
    )

    logger.info(
        f"\n{confusion_matrix(y_test,y_pred)}"
    )

    logger.info(
        f"\n{classification_report(y_test,y_pred)}"
    )
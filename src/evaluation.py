import numpy as np

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report
)

from src.logger import logger


def find_best_threshold(
    probs,
    y_true
):

    logger.info("Searching best threshold...")

    best_threshold = 0.5
    best_f1 = 0

    for threshold in np.arange(
        0.10,
        0.90,
        0.01
    ):

        predictions = (
            probs >= threshold
        ).astype(int)

        f1 = f1_score(
            y_true,
            predictions
        )

        if f1 > best_f1:

            best_f1 = f1
            best_threshold = threshold

    logger.info(
        f"Best Threshold = {best_threshold:.2f}"
    )

    logger.info(
        f"Best F1 Score = {best_f1:.4f}"
    )

    return best_threshold


def evaluate(
    y_true,
    y_pred
):

    accuracy = accuracy_score(
        y_true,
        y_pred
    )

    precision = precision_score(
        y_true,
        y_pred
    )

    recall = recall_score(
        y_true,
        y_pred
    )

    f1 = f1_score(
        y_true,
        y_pred
    )

    cm = confusion_matrix(
        y_true,
        y_pred
    )

    report = classification_report(
        y_true,
        y_pred
    )

    logger.info("=" * 60)
    logger.info("MODEL EVALUATION")
    logger.info("=" * 60)

    logger.info(f"Accuracy  : {accuracy:.4f}")
    logger.info(f"Precision : {precision:.4f}")
    logger.info(f"Recall    : {recall:.4f}")
    logger.info(f"F1 Score  : {f1:.4f}")

    logger.info("\nConfusion Matrix")
    logger.info(f"\n{cm}")

    logger.info("\nClassification Report")
    logger.info(f"\n{report}")

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1
    }
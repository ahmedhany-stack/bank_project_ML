import joblib

from src.logger import logger
from src.config import MODEL_FILE


def train_model(
    model,
    x_train,
    y_train
):

    logger.info("Training model...")

    model.fit(
        x_train,
        y_train
    )

    logger.info("Model training completed.")

    return model


def save_model(
    model
):

    logger.info(f"Saving model to {MODEL_FILE}")

    joblib.dump(
        model,
        MODEL_FILE
    )

    logger.info("Model saved successfully.")
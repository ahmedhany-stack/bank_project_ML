import joblib

from config import MODEL_FILE

def train_model(
    model,
    x_train,
    y_train
):

    model.fit(
        x_train,
        y_train
    )

    return model


def save_model(model):

    joblib.dump(
        model,
        MODEL_FILE
    )
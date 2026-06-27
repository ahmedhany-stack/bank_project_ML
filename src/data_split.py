from sklearn.model_selection import train_test_split

from src.config import (
    TEST_SIZE,
    RANDOM_STATE
)


def split_data(
    X,
    y
):

    return train_test_split(

        X,

        y,

        test_size=TEST_SIZE,

        random_state=RANDOM_STATE,

        stratify=y

    )
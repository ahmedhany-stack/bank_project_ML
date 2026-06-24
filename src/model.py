from xgboost import (
    XGBClassifier
)

def calculate_class_weight(
    y_train
):

    neg = (
        y_train == 0
    ).sum()

    pos = (
        y_train == 1
    ).sum()

    return (
        neg / pos
    ) * 2.0


def build_model(
    scale_pos_weight
):

    return XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        max_delta_step=2,
        learning_rate=0.05,
        n_estimators=600,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        tree_method="hist",
        n_jobs=-1
    )
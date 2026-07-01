import os
import joblib
import pandas as pd

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.feature_engineering import create_features
from src.encoding import transform_prediction_data
import sys
import os


app = FastAPI(
    title="Bank Deposit Prediction API",
    version="1.0.0"
)

MODEL_PATH = "models/xgboost_model.pkl"


def load_model():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


model = load_model()
print(type(model))
print(model.feature_names_in_)

class BankCustomer(BaseModel):

    age: float
    job: str
    marital: str
    education: str
    default_flag: str
    balance: float
    housing: str
    loan: str
    contact: str
    day: int
    month: str
    duration: float
    campaign: float
    pdays: float
    previous: float
    poutcome: str


@app.get("/")
def home():

    return {
        "message": "Bank Deposit Prediction API is running"
    }


@app.post("/predict")
def predict(customer: BankCustomer):

    global model

    try:

        # -----------------------------
        # Raw Data
        # -----------------------------

        raw_df = pd.DataFrame(
            [customer.model_dump()]
        )

        # -----------------------------
        # Feature Engineering
        # -----------------------------

        feature_df = create_features(
            raw_df
        )

        # -----------------------------
        # Preprocessing
        # -----------------------------

        final_df = transform_prediction_data(
            feature_df
        )

        # -----------------------------
        # Prediction
        # -----------------------------
        print(final_df.columns.tolist())

        prediction = model.predict(
            final_df
        )[0]

        probability = model.predict_proba(
            final_df
        )[0][1]

        label = (
            "yes"
            if prediction == 1
            else "no"
        )

        return {

            "prediction": int(prediction),

            "deposit": label,

            "probability": round(
                float(probability),
                4
            )

        }

    except Exception as e:

        raise HTTPException(
            status_code=500,
            detail=str(e)
        )
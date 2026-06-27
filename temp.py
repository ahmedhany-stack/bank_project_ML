import os
import pickle
import pandas as pd
import numpy as np
from fastapi import  HTTPException
from pydantic import BaseModel
from src.feature_engineering import create_features
from src.encoding import encode_categorical_features,scale_features


# 1. الاستدعاء الصحيح والمطلق لجميع الأدوات والموديلات

 # 👈 استدعاء البايبلاين بتاعتك!



MODEL_PATH = "models/xgboost_model.pkl"

def load_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"الموديل غير موجود في: {MODEL_PATH}")
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


model = load_model()


# تعريف المدخلات بناءً على أعمدة البنك الحقيقية اللي بعتها
class BankCustomerData(BaseModel):
    age: float
    job: str
    marital: str
    education: str
    default_flag: str
    balance: float
    housing: str
    loan: str
    contact: str
    day: float
    month: str
    duration: float
    campaign: float
    pdays: float
    previous: float
    poutcome: str


def predict(customer: BankCustomerData):
    global model
    if model is None:
        try:
            model = load_model()
        except Exception:
            raise HTTPException(status_code=503, detail="الموديل لسه مجهز ش على السيرفر.")
    
    try:
        # 1. تحويل البيانات القادمة من الـ API إلى Pandas DataFrame
        input_dict = customer.dict()
        raw_df = pd.DataFrame([input_dict])
        
        # 2. تمرير الـ DataFrame على دالة الـ Feature Engineering بتاعتك
        df = create_features(raw_df)
        df = encode_categorical_features(
            df
        )

        final_df = scale_features(
            df
        )
        prediction=model.predict(final_df)

        
        # 3. تعديل سحري: تحويل أي عمود نصي باقي إلى نوع 'category' عشان الـ XGBoost يفهمه

        
        return {
            "prediction": int(prediction[0]),

 
        }
        
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"خطأ أثناء معالجة البيانات: {str(e)}")
    
#docker compose up -d --build fastapi-app

test = BankCustomerData(
    age=58,
    job="management",
    marital="married",
    education="tertiary",
    default_flag="no",
    balance=2143,
    housing="yes",
    loan="no",
    contact="unknown",
    day=5,
    month="may",
    duration=261,
    campaign=1,
    pdays=-1,
    previous=0,
    poutcome="unknown"
)
print(predict(test))
    
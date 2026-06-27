# 1. استخدام نسخة بايثون خفيفة ومستقرة
FROM python:3.10-slim

# 2. تحديد الفولدر الرئيسي جوه الكونتينر
WORKDIR /app

# 3. تحديث الـ pip وتثبيت كل المكتبات المطلوبة للموديل والـ API دفعة واحدة
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
    fastapi \
    uvicorn \
    pandas \
    numpy \
    xgboost \
    scikit-learn

# 4. الأمر اللي بيشغل الـ FastAPI أول ما الكونتينر يقوم
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]
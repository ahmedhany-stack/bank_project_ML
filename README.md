# Bank Marketing Deposit Prediction

## Project Overview

This project focuses on predicting whether a bank customer will subscribe to a term deposit using Machine Learning.

The dataset used is the Bank Marketing Dataset, which contains customer information, campaign details, and historical contact records.

The project follows a complete Machine Learning pipeline:

* Exploratory Data Analysis (EDA)
* Data Cleaning
* Outlier Handling
* Skewness Transformation
* Feature Engineering
* Feature Encoding
* Feature Scaling
* Model Training
* Model Evaluation
* Model Saving

---

## Project Structure

```text
bank_project/

│
├── data/
│   ├── raw/
│   ├── interim/
│   └── processed/
│
├── logs/
│
├── models/
│
├── plots/
│
├── reports/
│
├── src/
│   ├── discover_data.py
│   ├── preprocessing_outliers.py
│   ├── skewed_handling.py
│   ├── encoding.py
│   ├── feature_engineering.py
│   ├── data_split.py
│   ├── model.py
│   ├── evaluation.py
│   ├── train.py
│   ├── logger.py
│   ├── config.py
│   └── main.py
│
├── requirements.txt
└── README.md
```

---

## Dataset

The dataset contains information about:

* Customer demographics
* Job information
* Marital status
* Education
* Bank balance
* Previous marketing campaigns
* Call duration
* Number of contacts
* Housing loans
* Personal loans

Target Variable:

```text
deposit
```

* yes → Customer subscribed
* no → Customer did not subscribe

---

## Exploratory Data Analysis

The EDA pipeline automatically generates:

* Dataset summary
* Missing values report
* Duplicate analysis
* Numerical statistics
* Target distribution
* Correlation analysis
* Histograms
* Boxplots
* Target vs Numerical Features
* Target vs Categorical Features

Generated files:

```text
reports/eda_report.txt

plots/
├── correlation_heatmap.png
├── histograms.png
├── boxplots/
├── target_numeric/
└── target_categorical/
```

---

## Data Preprocessing

### Outlier Handling

Outliers are handled using the IQR Method.

```text
Lower Bound = Q1 - 1.5 × IQR
Upper Bound = Q3 + 1.5 × IQR
```

Extreme values are capped instead of removed.

---

### Skewness Handling

The following features are transformed using Yeo-Johnson Transformation:

* age
* balance
* duration
* campaign
* pdays
* previous

This helps create more normal distributions.

---

### Encoding

Categorical features are encoded using:

* Label Encoding (binary features)
* One-Hot Encoding (multi-category features)

---

### Feature Scaling

Numerical features are scaled using:

```text
MinMaxScaler
```

Range:

```text
[0, 1]
```

---

## Feature Engineering

Additional features were created:

```python
balance_log
balance_high

campaign_log
campaign_high

pdays_contacted

previous_log

duration_log
duration_high

age_group
```

These features improve model performance by capturing hidden patterns.

---

## Model

Algorithm:

```text
XGBoost Classifier
```

Configuration:

```python
learning_rate=0.05
n_estimators=600
max_depth=6
subsample=0.8
colsample_bytree=0.8
scale_pos_weight=balanced
```

---

## Evaluation Metrics

The model is evaluated using:

* Accuracy
* Precision
* Recall
* F1 Score
* Confusion Matrix
* Classification Report

Threshold tuning is applied to maximize Recall.

---

## Logging

All pipeline steps are logged automatically.

Generated logs:

```text
logs/

├── eda_pipeline.log
├── outlier_pipeline.log
├── transformation_pipeline.log
├── encoding_pipeline.log
└── model_training.log
```

---

## Saved Model

Trained models are stored in:

```text
models/
```

Example:

```text
models/xgboost_model.pkl
```

---

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd bank_project
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Run

Execute the full pipeline:

```bash
python main.py
```

---

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-Learn
* XGBoost
* Joblib

---

## Future Improvements

* Hyperparameter Tuning
* Cross Validation
* MLflow Integration
* Airflow Orchestration
* Docker Deployment
* FastAPI Inference API
* CI/CD Pipeline

---

## Author

Ahmed Hany

Faculty of Artificial Intelligence

Machine Learning & AI Enthusiast

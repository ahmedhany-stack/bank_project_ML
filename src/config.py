import os

LOG_DIR = "logs"
MODEL_DIR = "models"

os.makedirs(LOG_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)

LOG_FILE = os.path.join(
    LOG_DIR,
    "model_training.log"
)

MODEL_FILE = os.path.join(
    MODEL_DIR,
    "xgboost_model.pkl"
)

RANDOM_STATE = 42

TEST_SIZE = 0.2
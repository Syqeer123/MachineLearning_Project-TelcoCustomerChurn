# config.py
from pathlib import Path

# Directory Paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "Telco-Customer-Churn.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(exist_ok=True) # Automatically creates the 'models' folder if it doesn't exist

# Model Parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2

# Feature Groupings
TARGET_COL = "Churn"
IGNORE_COLS = ["customerID"]

NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]

BINARY_COLS = [
    "gender", "SeniorCitizen", "Partner", "Dependents", 
    "PhoneService", "PaperlessBilling"
]

MULTICLASS_COLS = [
    "MultipleLines", "InternetService", "OnlineSecurity", 
    "OnlineBackup", "DeviceProtection", "TechSupport", 
    "StreamingTV", "StreamingMovies", "Contract", "PaymentMethod"
]
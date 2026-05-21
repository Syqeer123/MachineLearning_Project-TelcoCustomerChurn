# src/data_preprocessing.py
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import sys
from pathlib import Path

# Link to config file in the parent directory
sys.path.append(str(Path(__file__).resolve().parents[1]))
import config

def load_and_clean_data(file_path: Path) -> pd.DataFrame:
    """Loads dataset and fixes structural anomalies like whitespace strings."""
    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found at: {file_path}. Make sure the CSV is in the root folder.")
        
    df = pd.read_csv(file_path)
    
    # Handle blank spaces in TotalCharges
    if "TotalCharges" in df.columns:
        df["TotalCharges"] = df["TotalCharges"].replace(r"^\s*$", np.nan, regex=True)
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        # Fill missing with median
        df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())
        
    return df

def get_train_test_splits(df: pd.DataFrame):
    """Splits features and target into training and validation sets."""
    X = df.drop(columns=[config.TARGET_COL] + config.IGNORE_COLS, errors="ignore")
    # Convert 'Yes'/'No' string in Target to 1/0
    y = df[config.TARGET_COL].apply(lambda x: 1 if str(x).strip().lower() == "yes" else 0)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=config.TEST_SIZE, 
        random_state=config.RANDOM_STATE, 
        stratify=y
    )
    return X_train, X_test, y_train, y_test
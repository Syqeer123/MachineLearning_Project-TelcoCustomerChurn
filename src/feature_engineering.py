# src/feature_engineering.py
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from imblearn.over_sampling import SMOTE
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config

def build_preprocessor() -> ColumnTransformer:
    """Creates a consistent column transformer for scaling and encoding."""
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), config.NUMERIC_COLS),
            ("cat", OneHotEncoder(drop="first", handle_unknown="ignore"), config.MULTICLASS_COLS + config.BINARY_COLS)
        ]
    )
    return preprocessor

def apply_smote(X_train_processed, y_train):
    """Applies SMOTE exclusively to training data."""
    smote = SMOTE(random_state=config.RANDOM_STATE)
    X_resampled, y_resampled = smote.fit_resample(X_train_processed, y_train)
    return X_resampled, y_resampled
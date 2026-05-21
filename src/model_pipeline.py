# src/model_pipeline.py
import joblib
import logging
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, accuracy_score
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parents[1]))
import config
from src.data_preprocessing import load_and_clean_data, get_train_test_splits
from src.feature_engineering import build_preprocessor, apply_smote

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

def run_pipeline():
    logging.info("Starting Machine Learning Pipeline...")
    
    # 1. Load & Clean
    df = load_and_clean_data(config.DATA_PATH)
    X_train, X_test, y_train, y_test = get_train_test_splits(df)
    
    # 2. Preprocess (Scale & Encode)
    preprocessor = build_preprocessor()
    X_train_mod = preprocessor.fit_transform(X_train)
    X_test_mod = preprocessor.transform(X_test)
    
    # 3. Handle Imbalance with SMOTE
    logging.info("Applying SMOTE to training data...")
    X_train_res, y_train_res = apply_smote(X_train_mod, y_train)
    
    # 4. Train Model
    logging.info("Training XGBoost Classifier...")
    model = XGBClassifier(random_state=config.RANDOM_STATE, eval_metric="logloss")
    model.fit(X_train_res, y_train_res)
    
    # 5. Evaluate
    preds = model.predict(X_test_mod)
    logging.info(f"Test Accuracy: {accuracy_score(y_test, preds):.4f}")
    
    # 6. Save Artifacts for Streamlit
    joblib.dump(preprocessor, config.MODEL_DIR / "preprocessor.joblib")
    joblib.dump(model, config.MODEL_DIR / "xgboost_model.joblib")
    logging.info(f"Pipeline complete! Models saved in: {config.MODEL_DIR}")

if __name__ == "__main__":
    run_pipeline()
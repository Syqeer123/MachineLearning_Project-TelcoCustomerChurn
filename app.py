# app.py
import streamlit as st
import pandas as pd
import joblib
from pathlib import Path
import config

st.set_page_config(page_title="Telco Churn Predictor", layout="wide")

st.title("📊 Telco Customer Churn Predictor")
st.write("Enter customer details below to predict the probability of churn using our trained XGBoost model.")

# Load the saved model and preprocessor
@st.cache_resource
def load_artifacts():
    preprocessor = joblib.load(config.MODEL_DIR / "preprocessor.joblib")
    model = joblib.load(config.MODEL_DIR / "xgboost_model.joblib")
    return preprocessor, model

try:
    preprocessor, model = load_artifacts()
except FileNotFoundError:
    st.error("⚠️ Model files not found! Please run `python src/model_pipeline.py` first.")
    st.stop()

# Build the UI layout
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Demographics")
    gender = st.selectbox("Gender", ["Female", "Male"])
    senior_citizen = st.selectbox("Senior Citizen", ["No", "Yes"])
    partner = st.selectbox("Partner", ["Yes", "No"])
    dependents = st.selectbox("Dependents", ["Yes", "No"])
    tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, value=12)

with col2:
    st.subheader("Services")
    phone_service = st.selectbox("Phone Service", ["Yes", "No"])
    multiple_lines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
    internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
    online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])
    device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])

with col3:
    st.subheader("Account")
    tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
    streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
    streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
    contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
    paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])
    payment_method = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
    monthly_charges = st.number_input("Monthly Charges ($)", min_value=0.0, value=65.0)
    total_charges = st.number_input("Total Charges ($)", min_value=0.0, value=780.0)

st.markdown("---")

if st.button("Predict Churn Risk", type="primary"):
    # Create DataFrame from inputs
    input_data = pd.DataFrame([{
        "gender": gender, "SeniorCitizen": 1 if senior_citizen == "Yes" else 0, "Partner": partner, "Dependents": dependents,
        "tenure": tenure, "PhoneService": phone_service, "MultipleLines": multiple_lines,
        "InternetService": internet_service, "OnlineSecurity": online_security, "OnlineBackup": online_backup,
        "DeviceProtection": device_protection, "TechSupport": tech_support, "StreamingTV": streaming_tv,
        "StreamingMovies": streaming_movies, "Contract": contract, "PaperlessBilling": paperless_billing,
        "PaymentMethod": payment_method, "MonthlyCharges": monthly_charges, "TotalCharges": total_charges
    }])
    
    # Predict
    processed_input = preprocessor.transform(input_data)
    prediction = model.predict(processed_input)[0]
    probability = model.predict_proba(processed_input)[0][1]
    
    # Output Result
    if prediction == 1:
        st.error(f"🚨 **High Churn Risk!** Probability: **{probability:.2%}**")
    else:
        st.success(f"✅ **Customer is likely to stay.** Churn Probability: **{probability:.2%}**")
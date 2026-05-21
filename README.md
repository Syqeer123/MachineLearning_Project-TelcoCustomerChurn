# Telco Customer Churn Predictor

![Python](https://img.shields.io/badge/Python-3.x-blue)
![XGBoost](https://img.shields.io/badge/Model-XGBoost-orange)
![Streamlit](https://img.shields.io/badge/App-Streamlit-red)
![Status](https://img.shields.io/badge/Status-Live-brightgreen)

> Predicts the probability of a telecom customer churning using an XGBoost 
> classifier, with SHAP explainability to show which factors drive each prediction.

🔗 **Live Demo:** [syqeer-telco-churn-predictor.streamlit.app](https://syqeer-telco-churn-predictor.streamlit.app/)

---

## Problem Statement
Customer acquisition in the telecommunications industry is significantly more expensive than customer retention. This project leverages machine learning to proactively identify customers at high risk of churning, allowing the business to deploy targeted retention strategies, optimize marketing spend, and reduce overall revenue leakage.

---

## Dataset
- **Source:** IBM Telco Customer Churn Dataset
- **Size:** 7,043 rows, 21 features
- **Target:** `Churn` — binary classification (Yes/No)

---

## Approach

### 1. Exploratory Data Analysis
An analysis of the dataset revealed a baseline churn rate of approximately 26.5%. Key drivers of churn consistently point toward contract types and service configurations, with customers on Month-to-Month contracts and those utilizing Fiber Optic internet showing a historically higher propensity to leave. 

### 2. Preprocessing
- **Scaling:** Applied `StandardScaler` to numerical columns (`tenure`, `MonthlyCharges`, `TotalCharges`) to normalize the data distribution.
- **Data Split:** The dataset was partitioned using a strict 80/20 train-test ratio (`test_size=0.2`, `random_state=42`) to ensure robust, unbiased validation.

### 3. Model Selection
Multiple algorithms were evaluated to find the optimal balance between accuracy and the ability to distinguish between classes (ROC-AUC). While Logistic Regression had a marginally higher raw test accuracy, XGBoost achieved the highest Cross-Validation ROC-AUC score, making it the most robust and reliable choice for handling the dataset's underlying patterns.

| Model | Test Accuracy | CV ROC-AUC |
|---|---|---|
| Logistic Regression | 0.7875 | 0.8446 |
| Random Forest | 0.7854 | 0.8401 |
| **XGBoost** | **0.7775** | **0.8466** |

### 4. Explainability
SHAP (SHapley Additive exPlanations) is utilized to eliminate the "black box" nature of the model. A dynamic waterfall chart is generated for every prediction, calculating the exact weight of each feature and illustrating how variables pushed the individual customer's churn probability up or down.

---

## App Features
- Input customer demographics, services, and account details.
- Predicts churn probability utilizing the trained XGBoost model.
- Generates dynamic SHAP waterfall charts to explain the "why" behind every individual prediction.
- Deployed live on Streamlit Cloud with an automated execution pipeline.

---

## Project Structure
```text
TelcoCustomerChurn_ML-Project/
├── app.py                  # Streamlit application front-end
├── models/
│   ├── preprocessor.joblib # Data scaling and encoding pipeline
│   └── xgboost_model.joblib# Trained XGBoost binary
├── notebook/               # Jupyter notebooks for EDA and training
├── src/
│   └── model_pipeline.py   # Automated training and export script
├── requirements.txt        # Cloud deployment dependencies
└── README.md
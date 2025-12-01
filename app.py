import streamlit as st
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

# ================================
# 1. Load data + train model
# ================================
@st.cache_resource
def load_model():
    # Read the Telco churn data from the CSV in this repo
    df = pd.read_csv("Telco_customer_churn.csv")

    data = df.copy()

    # Keep only rows where churn value is present
    data = data.dropna(subset=["Churn Value"])

    # Make Total Charges numeric and drop rows where it's missing
    data["Total Charges"] = pd.to_numeric(data["Total Charges"], errors="coerce")
    data = data.dropna(subset=["Total Charges"])

    # Rename columns so they match what we use in the app
    data = data.rename(
        columns={
            "Tenure Months": "tenure",
            "Monthly Charges": "MonthlyCharges",
            "Total Charges": "TotalCharges",
            "Payment Method": "PaymentMethod",
            "Internet Service": "InternetService",
        }
    )

    # Only keep the 6 features + target
    cols_needed = [
        "tenure",
        "MonthlyCharges",
        "TotalCharges",
        "Contract",
        "PaymentMethod",
        "InternetService",
        "Churn Value",
    ]
    data = data[cols_needed].dropna()

    # Target and features
    y = data["Churn Value"].astype(int)
    X = data[
        [
            "tenure",
            "MonthlyCharges",
            "TotalCharges",
            "Contract",
            "PaymentMethod",
            "InternetService",
        ]
    ]

    # Numeric and categorical feature lists
    numeric_features = ["tenure", "MonthlyCharges", "TotalCharges"]
    categorical_features = ["Contract", "PaymentMethod", "InternetService"]

    # Preprocessing steps
    numeric_transformer = Pipeline(steps=[("scaler", StandardScaler())])
    categorical_transformer = Pipeline(
        steps=[("onehot", OneHotEncoder(handle_unknown="ignore"))]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )

    # Logistic Regression pipeline (same idea as in Databricks)
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("classifier", LogisticRegression(max_iter=1000)),
        ]
    )

    # Fit the model
    model.fit(X, y)

    return model


# Load the model once (cached)
model = load_model()

# ================================
# 2. Streamlit user interface
# ================================
st.title("📊 Customer Churn Prediction App")
st.write("Enter customer information to predict the probability of churn.")

with st.form("input_form"):
    st.subheader("Enter Customer Information")

    tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, value=12)
    monthly_charges = st.number_input(
        "Monthly Charges", min_value=0.0, max_value=200.0, value=70.0
    )
    total_charges = st.number_input(
        "Total Charges", min_value=0.0, max_value=10000.0, value=800.0
    )
    contract = st.selectbox(
        "Contract Type", ["Month-to-month", "One year", "Two year"]
    )
    payment_method = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer", "Credit card"],
    )
    internet_service = st.selectbox(
        "Internet Service", ["DSL", "Fiber optic", "None"]
    )

    submitted = st.form_submit_button("Predict")

# ================================
# 3. Run prediction
# ================================
if submitted:
    # Build a one-row DataFrame with the same column names as training
    input_data = pd.DataFrame(
        {
            "tenure": [tenure],
            "MonthlyCharges": [monthly_charges],
            "TotalCharges": [total_charges],
            "Contract": [contract],
            "PaymentMethod": [payment_method],
            "InternetService": [internet_service],
        }
    )

    # Probability of churn (class 1)
    churn_prob = float(model.predict_proba(input_data)[0][1])

    st.subheader("📌 Prediction Result")
    st.write(f"**Churn Probability:** {churn_prob:.2f}")

    if churn_prob > 0.5:
        st.error("⚠️ High Risk of Churn")
    else:
        st.success("✅ Low Risk of Churn")

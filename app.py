import streamlit as st
import pandas as pd
import pickle

MODEL_PATH = "model.pkl"

# Load the sklearn pipeline that you logged from Databricks
@st.cache_resource
def load_model():
    with open(MODEL_PATH, "rb") as f:
        model = pickle.load(f)
    return model

model = load_model()

st.title("📊 Customer Churn Prediction App")
st.write("Upload customer information to predict churn probability.")

# --- User input form ---
with st.form("input_form"):
    st.subheader("Enter Customer Information")
    
    tenure = st.number_input("Tenure (Months)", min_value=0, max_value=100, value=12)
    monthly_charges = st.number_input("Monthly Charges", min_value=0.0, max_value=200.0, value=70.0)
    total_charges = st.number_input("Total Charges", min_value=0.0, max_value=10000.0, value=900.0)

    # These category values MUST match the Telco dataset exactly
    contract = st.selectbox(
        "Contract Type",
        ["Month-to-month", "One year", "Two year"]
    )
    payment_method = st.selectbox(
        "Payment Method",
        ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    )
    internet_service = st.selectbox(
        "Internet Service",
        ["DSL", "Fiber optic", "No"]
    )
    
    submitted = st.form_submit_button("Predict")

# --- Run prediction ---
if submitted:
    # IMPORTANT: column names must match what you used in Databricks
    input_data = pd.DataFrame({
        "tenure": [tenure],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges],
        "Contract": [contract],
        "PaymentMethod": [payment_method],
        "InternetService": [internet_service],
    })

    # Pipeline handles preprocessing + model
    pred_prob = model.predict_proba(input_data)[0][1]

    st.subheader("📌 Prediction Result")
    st.write(f"**Churn Probability:** {pred_prob:.2f}")

    if pred_prob > 0.5:
        st.error("⚠️ High Risk of Churn")
    else:
        st.success("✅ Low Risk of Churn")

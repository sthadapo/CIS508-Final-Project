📊 CIS508 Final Project — Customer Churn Prediction

This project predicts whether a telecom customer is likely to cancel their service.
I built the model in Databricks, tracked everything with MLflow, and created an interactive Streamlit app so anyone can test the model.


Project Goal

The goal is to help a business understand which customers are at high risk of churn so they can take action early (discounts, offers, outreach, etc.).


What I Did

Cleaned and prepared the Telco Customer Churn dataset

Trained several machine learning models

Compared performance using accuracy, precision, recall, F1, and AUC

Logged everything in MLflow

Chose the best model and saved it

Built a Streamlit web app to make predictions


Best Model

The best model was a Logistic Regression pipeline.

It had:

AUC: ~0.83

Accuracy: ~0.80

Precision: ~0.64

Recall: ~0.54

It performed the most balanced overall and works well for business decisions.


Streamlit App

The app asks for a few customer details:

Tenure

Monthly Charges

Total Charges

Contract Type

Payment Method

Internet Service

Then it returns a churn probability score and a simple message:

High Risk of Churn ⚠️

Low Risk of Churn ✅


To run the app:
pip install streamlit pandas mlflow scikit-learn
streamlit run app.py


Files in This Repository
app.py                # Streamlit web app
model.pkl             # Saved machine learning model
conda.yaml            # Optional environment file
Telco_customer_churn.csv   # Dataset


Final Submission Includes

Streamlit app

Video presentation

Slide deck

MLflow experiment link

GitHub repository (this one)

import streamlit as st
import joblib
import numpy as np

model = joblib.load("loan_default_model.pkl")
scaler = joblib.load("scaler.pkl")

st.title("🏦 Loan Default Prediction System")

income = st.number_input("Customer Income")
loan_amount = st.number_input("Loan Amount")
interest = st.number_input("Interest Rate")
credit_history = st.number_input("Credit History Length")
loan_percent_income = st.number_input("Loan Percent of Income")

if st.button("Check Loan Risk"):
    data = np.array([[income, loan_amount, interest, credit_history, loan_percent_income]])
    data = scaler.transform(data)
    prediction = model.predict(data)

    if prediction[0] == 1:
        st.error("❌ HIGH RISK – Loan will Default")
    else:
        st.success("✅ LOW RISK – Loan can be Approved")

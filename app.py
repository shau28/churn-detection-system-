import streamlit as st
import pandas as pd
import numpy as np
import pickle

# -------------------------------
# Load Model & Preprocessing
# -------------------------------
with open('model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('label_enc.pkl', 'rb') as obj:
    lb_enc = pickle.load(obj)

with open('ohe_enc.pkl', 'rb') as obj:
    ohe_enc = pickle.load(obj)

with open('scaler.pkl', 'rb') as obj:
    scaler = pickle.load(obj)

# -------------------------------
# App Title
# -------------------------------
st.title("Customer Churn Prediction")

st.write("Enter customer details to predict churn.")

# -------------------------------
# User Inputs
# -------------------------------
credit_score = st.number_input("Credit Score", 300, 900, 600)
age = st.number_input("Age", 18, 100, 30)
tenure = st.number_input("Tenure (Years)", 0, 10, 3)
balance = st.number_input("Balance", 0.0, 250000.0, 50000.0)
num_products = st.number_input("Number of Products", 1, 4, 1)
has_cr_card = st.selectbox("Has Credit Card", [0, 1])
is_active_member = st.selectbox("Is Active Member", [0, 1])
estimated_salary = st.number_input("Estimated Salary", 0.0, 200000.0, 50000.0)

geography = st.selectbox("Geography", ["France", "Germany", "Spain"])
gender = st.selectbox("Gender", ["Male", "Female"])

# -------------------------------
# Prediction
# -------------------------------
if st.button("Predict"):

    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary],
        'Geography': [geography],
        'Gender': [gender]
    })

    # Encode Gender
    input_data['Gender'] = lb_enc.transform(input_data['Gender'])

    # One-hot encode Geography
    geo_encoded = ohe_enc.transform(input_data[['Geography']]).toarray()
    geo_df = pd.DataFrame(geo_encoded, columns=ohe_enc.get_feature_names_out(['Geography']))

    # Drop original Geography
    input_data = input_data.drop('Geography', axis=1)

    # Combine
    input_data = pd.concat([input_data, geo_df], axis=1)

    # Scale
    input_data = scaler.transform(input_data)

    # Predict
    pred = model.predict(input_data)

    # Output
    if pred[0] == 1:
        st.error("Customer is likely to churn ❌")
    else:
        st.success("Customer is not likely to churn ✅")

# -------------------------------
# Footer
# -------------------------------
st.markdown("""
    <hr>
    <p style='text-align: right;'>Created by Shaurya Pal</p>
""", unsafe_allow_html=True)

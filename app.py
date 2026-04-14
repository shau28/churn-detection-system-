import streamlit as st
import pandas as pd
import numpy as np
import pickle

# -------------------------------
# Load Model & Preprocessing
# -------------------------------
model = pickle.load(open('model.pkl', 'rb'))

with open('label_enc.pkl', 'rb') as obj:
    lb_enc = pickle.load(obj)

with open('ohe_enc.pkl', 'rb') as obj:
    ohe_enc = pickle.load(obj)

with open('scaler.pkl', 'rb') as obj:
    scaler = pickle.load(obj)

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(page_title="Churn Prediction", page_icon="📊")

# -------------------------------
# Title
# -------------------------------
st.title("📊 Customer Churn Prediction System")
st.markdown("### 🇮🇳 Indian Banking Customer Analysis")

# -------------------------------
# Input Fields
# -------------------------------
CreditScore = st.number_input("💳 Credit Score", min_value=300, max_value=900, value=650)

city = st.selectbox("🏙️ City", ["Delhi", "Mumbai", "Bangalore"])

gender = st.selectbox("👤 Gender", lb_enc.classes_)

age = st.slider("🎂 Age", 18, 90, 30)

tenure = st.slider("📅 Tenure (Years)", 0, 10, 5)

balance = st.number_input("💰 Account Balance", value=50000.0)

nbrprod = st.slider("📦 Number of Products", 1, 4, 1)

credit_card = st.selectbox("💳 Has Credit Card", [0, 1])

is_active = st.selectbox("⚡ Active Member", [0, 1])

salary = st.number_input("💵 Estimated Salary", value=50000.0)

# -------------------------------
# Map Indian Cities to Model Data
# -------------------------------
if city == "Delhi":
    geography = "France"
elif city == "Mumbai":
    geography = "Germany"
else:
    geography = "Spain"

# -------------------------------
# Prediction Button
# -------------------------------
if st.button("🔍 Predict Churn"):

    # Create DataFrame
    input_data = pd.DataFrame({
        'CreditScore': [CreditScore],
        'Geography': [geography],
        'Gender': [gender],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [nbrprod],
        'HasCrCard': [credit_card],
        'IsActiveMember': [is_active],
        'EstimatedSalary': [salary]
    })

    # Encode Gender
    input_data['Gender'] = lb_enc.transform(input_data['Gender'])

    # One-Hot Encode Geography
    geo_encoded = ohe_enc.transform([[geography]]).toarray()
    geo_df = pd.DataFrame(geo_encoded, columns=ohe_enc.get_feature_names_out())

    # Combine Data
    input_data = pd.concat([input_data.drop('Geography', axis=1), geo_df], axis=1)

    # Align columns
    input_data = input_data.reindex(columns=scaler.feature_names_in_, fill_value=0)

    # Scale data
    input_scaled = scaler.transform(input_data)

    # Prediction
    prediction = model.predict(input_scaled)

    # Handle output safely
    try:
        prob = prediction[0][0]
    except:
        prob = prediction[0]

    # Output
    if prob > 0.5:
        st.error(f"⚠️ Customer is likely to CHURN\n\nProbability: {prob*100:.2f}%")
    else:
        st.success(f"✅ Customer is NOT likely to churn\n\nProbability: {prob*100:.2f}%")

    st.info("This prediction is based on a Machine Learning model.")

# -------------------------------
# Footer (Added by you)
# -------------------------------
st.markdown("""
    <hr>
    <p style='text-align: right;'>Created by Shaurya Pal</p>
""", unsafe_allow_html=True)

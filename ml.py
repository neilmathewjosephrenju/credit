import streamlit as st
import joblib
import numpy as np
import pandas as pd

# Load the trained model
with open("SouthGermanCredit_English.pkl", "rb") as model_file:
    model = joblib.load(model_file)

# Define feature names manually (avoid reloading dataset)
feature_names = [
    "status", "duration", "credit_history", "purpose", "amount", "savings", "employment",
    "installment_rate", "personal_status_sex", "other_debtors", "present_residence",
    "property", "age", "other_installment_plans", "housing", "existing_credits",
    "job", "people_liable", "telephone", "foreign_worker"
]

# Streamlit UI
st.title("Credit Risk Prediction App")
st.write("Enter the customer details below:")

# Collect user input dynamically
user_data = []
for feature in feature_names:
    value = st.number_input(f"Enter {feature}:", value=0.0)
    user_data.append(value)

# Convert input into a NumPy array
input_array = np.array(user_data).reshape(1, -1)

# Prediction
if st.button("Predict Credit Risk"):
    prediction = model.predict(input_array)
    st.subheader("Prediction Result:")
    if prediction[0] == 1:
        st.error("High Credit Risk ❌")
    else:
        st.success("Low Credit Risk ✅")

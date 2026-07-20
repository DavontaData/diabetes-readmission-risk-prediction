import streamlit as st
import pandas as pd
import joblib


model = joblib.load("diabetes_readmission_model.pkl")
model_features = joblib.load("model_features.pkl")


st.title("🏥 Diabetes 30-Day Readmission Risk Prediction")

st.markdown("""
This application estimates the likelihood of 30-day hospital readmission among
diabetic inpatient encounters with emergency admission using a machine-learning
model trained on the UCI Diabetes 130-US Hospitals dataset.

**Disclaimer:** This tool is intended for educational and research purposes
and should support—not replace—clinical judgment.
""")


age = st.number_input(
    "Age",
    min_value=5,
    max_value=100,
    value=50
)

number_inpatient_visits = st.number_input(
    "Number of Prior Inpatient Visits",
    min_value=0,
    value=0
)

number_emergency = st.number_input(
    "Number of Emergency Visits",
    min_value=0,
    value=0
)

time_in_hospital = st.number_input(
    "Time in Hospital (Days)",
    min_value=1,
    value=3
)

a1c_result = st.selectbox(
    "A1C Result",
    ["None", "Norm", ">7", ">8"]
)

max_glu_serum = st.selectbox(
    "Maximum Glucose Serum",
    ["None", "Norm", ">200", ">300"]
)

insulin = st.selectbox(
    "Insulin Treatment",
    ["No", "Steady", "Up", "Down"]
)

change_medication = st.selectbox(
    "Medication Change",
    ["No", "Yes"]
)


if st.button("Predict Readmission Risk"):

    # Create one row with the exact feature columns used by the trained model.
    input_data = pd.DataFrame(0, index=[0], columns=model_features)

    # Numeric features
    input_data["age"] = age
    input_data["number_inpatient_visits"] = number_inpatient_visits
    input_data["number_emergency"] = number_emergency
    input_data["time_in_hospital"] = time_in_hospital

    # A1C Result
    # >7 is the reference/baseline category, so all A1C dummy variables remain 0.
    if a1c

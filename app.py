import streamlit as st
import pandas as pd
import joblib


model = joblib.load("diabetes_readmission_model.pkl")
model_features = joblib.load("model_features.pkl")


st.title("🏥 Diabetes 30-Day Readmission Risk Prediction")


st.markdown("""
This application predicts the likelihood of 30-day hospital readmission among
diabetic inpatient encounters admitted through the emergency department using
a machine learning model trained on the UCI Diabetes 130-US Hospitals dataset.

**Disclaimer:** This tool is intended for educational and research purposes
and should not replace clinical judgment.
""")


age = st.number_input(
    "Age",
    min_value=18,
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
    ["None", "Norm", ">8"]
)

max_glu_serum = st.selectbox(
    "Maximum Glucose Serum",
    ["None", "Norm", ">300"]
)

insulin = st.selectbox(
    "Insulin Treatment",
    ["No", "Steady", "Up"]
)

change_medication = st.selectbox(
    "Medication Change",
    ["No", "Yes"]
)


if st.button("Predict Readmission Risk"):

  
    input_data = pd.DataFrame(0, index=[0], columns=model_features)

   
    input_data["age"] = age
    input_data["number_inpatient_visits"] = number_inpatient_visits
    input_data["number_emergency"] = number_emergency
    input_data["time_in_hospital"] = time_in_hospital

   
    if a1c_result == ">8":
        input_data["a1c_result_>8"] = 1
    elif a1c_result == "None":
        input_data["a1c_result_None"] = 1
    elif a1c_result == "Norm":
        input_data["a1c_result_Norm"] = 1

   
    if max_glu_serum == ">300":
        input_data["max_glu_serum_>300"] = 1
    elif max_glu_serum == "None":
        input_data["max_glu_serum_None"] = 1
    elif max_glu_serum == "Norm":
        input_data["max_glu_serum_Norm"] = 1

   
    if insulin == "No":
        input_data["insulin_No"] = 1
    elif insulin == "Steady":
        input_data["insulin_Steady"] = 1
    elif insulin == "Up":
        input_data["insulin_Up"] = 1

   
    if change_medication == "No":
        input_data["change_medication_No"] = 1

  
    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

   
    st.divider()

    st.subheader("Prediction Results")

    if prediction == 1:
        st.error("🔴 Predicted Class: High Risk of 30-Day Readmission")
    else:
        st.success("🟢 Predicted Class: No 30-Day Readmission")

    st.metric(
        label="Estimated Readmission Probability",
        value=f"{probability:.2%}"
    )

    st.divider()

    if probability >= 0.50:
        st.warning(
            """
            **Clinical Interpretation**

            This patient demonstrates characteristics associated with an
            increased likelihood of a 30-day hospital readmission.

            This prediction should be interpreted alongside clinical
            assessment and should not replace clinician judgment.
            """
        )

    else:
        st.info(
            """
            **Clinical Interpretation**

            This patient demonstrates a lower predicted probability of
            30-day hospital readmission based on the trained machine
            learning model.

            Clinical judgment should always be considered when making
            patient care decisions.
            """
        )

import streamlit as st
import pandas as pd
import joblib




model = joblib.load("diabetes_readmission_model.pkl")
model_features = joblib.load("model_features.pkl")
final_threshold = joblib.load("diabetes_readmission_threshold.pkl")



st.title("Diabetes 30-Day Readmission Risk Prediction")

st.write(
    "Predict 30-day hospital readmission risk among diabetic emergency inpatient "
    "admissions using a tuned Logistic Regression machine learning model. "
    "Enter patient characteristics to receive a predicted readmission "
    "classification and estimated risk probability."
)


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

number_diagnoses = st.number_input(
    "Number of Diagnoses",
    min_value=1,
    value=5
)

number_medications = st.number_input(
    "Number of Medications",
    min_value=0,
    value=10
)

number_lab_procedures = st.number_input(
    "Number of Laboratory Procedures",
    min_value=0,
    value=40
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

 
    input_data = pd.DataFrame(
        0,
        index=[0],
        columns=model_features
    )

    input_data["age"] = age
    input_data["number_inpatient_visits"] = number_inpatient_visits
    input_data["number_emergency"] = number_emergency
    input_data["time_in_hospital"] = time_in_hospital
    input_data["number_diagnoses"] = number_diagnoses
    input_data["number_medications"] = number_medications
    input_data["number_lab_procedures"] = number_lab_procedures

   
    input_data["total_acute_visits"] = (
        number_inpatient_visits + number_emergency
    )

    input_data["diagnoses_per_day"] = (
        number_diagnoses / max(time_in_hospital, 1)
    )

    input_data["medications_per_day"] = (
        number_medications / max(time_in_hospital, 1)
    )

    input_data["labs_per_day"] = (
        number_lab_procedures / max(time_in_hospital, 1)
    )


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


    input_data = input_data[model_features]


    

    probability = model.predict_proba(input_data)[0][1]



    prediction = int(probability >= final_threshold)


    st.divider()

    st.subheader("Prediction Results")

    if prediction == 1:

        st.error(
            "Predicted Class: Higher Risk of 30-Day Readmission"
        )

    else:

        st.success(
            "Predicted Class: Lower Risk of 30-Day Readmission"
        )


    st.metric(
        label="Estimated Readmission Probability",
        value=f"{probability:.2%}"
    )

    st.caption(
        f"Classification threshold: {final_threshold:.2f}"
    )



    st.divider()

    if prediction == 1:

        st.warning(
            """
            **Risk Interpretation**

            The model estimates that this encounter has a higher
            predicted probability of 30-day hospital readmission
            based on the information provided.

            This prediction is intended for risk stratification
            and should not replace clinical judgment.
            """
        )

    else:

        st.info(
            """
            **Risk Interpretation**

            The model estimates that this encounter has a lower
            predicted probability of 30-day hospital readmission
            based on the information provided.

            This prediction is intended for risk stratification
            and should not replace clinical judgment.
            """
        )

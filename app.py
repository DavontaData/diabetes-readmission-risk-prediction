import streamlit as st
import pandas as pd
import joblib


# ============================================================
# Load Model Artifacts
# ============================================================

model = joblib.load("diabetes_readmission_model.pkl")
model_features = joblib.load("model_features.pkl")
final_threshold = joblib.load("diabetes_readmission_threshold.pkl")


# ============================================================
# Page Configuration
# ============================================================

st.set_page_config(
    page_title="Diabetes Readmission Risk Prediction",
    page_icon="🏥",
    layout="centered"
)


# ============================================================
# Application Title
# ============================================================

st.title("Diabetes 30-Day Readmission Risk Prediction")

st.write(
    "Predict 30-day hospital readmission risk among diabetic emergency "
    "inpatient admissions using a Logistic Regression machine learning model. "
    "Enter patient characteristics to receive a predicted readmission "
    "classification and estimated risk probability."
)


# ============================================================
# Intended Use Disclaimer
# ============================================================

st.warning(
    """
    **Research / Educational Prototype**

    This application is intended for research, educational, and portfolio
    demonstration purposes only. It has not been externally validated and
    should not be used for clinical decision-making.
    """
)


# ============================================================
# Patient / Encounter Inputs
# ============================================================

st.subheader("Patient Encounter Information")


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


# ============================================================
# Clinical Categorical Inputs
# ============================================================

st.subheader("Clinical Characteristics")


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


# ============================================================
# Prediction
# ============================================================

if st.button("Predict Readmission Risk"):

    # --------------------------------------------------------
    # Create input dataframe using exact model feature structure
    # --------------------------------------------------------

    input_data = pd.DataFrame(
        0,
        index=[0],
        columns=model_features
    )


    # --------------------------------------------------------
    # Numerical Features
    # --------------------------------------------------------

    input_data["age"] = age

    input_data["number_emergency"] = number_emergency

    input_data["time_in_hospital"] = time_in_hospital

    input_data["number_diagnoses"] = number_diagnoses

    input_data["number_medications"] = number_medications

    input_data["number_lab_procedures"] = number_lab_procedures


    # --------------------------------------------------------
    # Derived Features
    # --------------------------------------------------------

    input_data["medications_per_day"] = (
        number_medications / max(time_in_hospital, 1)
    )

    input_data["labs_per_day"] = (
        number_lab_procedures / max(time_in_hospital, 1)
    )

    input_data["diagnoses_per_day"] = (
        number_diagnoses / max(time_in_hospital, 1)
    )


    # --------------------------------------------------------
    # A1C One-Hot Encoding
    # --------------------------------------------------------

    if a1c_result == ">8":

        if "a1c_result_>8" in input_data.columns:
            input_data["a1c_result_>8"] = 1

    elif a1c_result == ">7":

        if "a1c_result_>7" in input_data.columns:
            input_data["a1c_result_>7"] = 1

    elif a1c_result == "None":

        if "a1c_result_None" in input_data.columns:
            input_data["a1c_result_None"] = 1

    elif a1c_result == "Norm":

        if "a1c_result_Norm" in input_data.columns:
            input_data["a1c_result_Norm"] = 1


    # --------------------------------------------------------
    # Maximum Glucose Serum One-Hot Encoding
    # --------------------------------------------------------

    if max_glu_serum == ">300":

        if "max_glu_serum_>300" in input_data.columns:
            input_data["max_glu_serum_>300"] = 1

    elif max_glu_serum == ">200":

        if "max_glu_serum_>200" in input_data.columns:
            input_data["max_glu_serum_>200"] = 1

    elif max_glu_serum == "None":

        if "max_glu_serum_None" in input_data.columns:
            input_data["max_glu_serum_None"] = 1

    elif max_glu_serum == "Norm":

        if "max_glu_serum_Norm" in input_data.columns:
            input_data["max_glu_serum_Norm"] = 1


    # --------------------------------------------------------
    # Insulin One-Hot Encoding
    # --------------------------------------------------------

    if insulin == "No":

        if "insulin_No" in input_data.columns:
            input_data["insulin_No"] = 1

    elif insulin == "Steady":

        if "insulin_Steady" in input_data.columns:
            input_data["insulin_Steady"] = 1

    elif insulin == "Up":

        if "insulin_Up" in input_data.columns:
            input_data["insulin_Up"] = 1

    elif insulin == "Down":

        if "insulin_Down" in input_data.columns:
            input_data["insulin_Down"] = 1


    # --------------------------------------------------------
    # Medication Change One-Hot Encoding
    # --------------------------------------------------------

    if change_medication == "No":

        if "change_medication_No" in input_data.columns:
            input_data["change_medication_No"] = 1

    elif change_medication == "Yes":

        if "change_medication_Yes" in input_data.columns:
            input_data["change_medication_Yes"] = 1


    # --------------------------------------------------------
    # Ensure Exact Feature Ordering
    # --------------------------------------------------------

    input_data = input_data[model_features]


    # --------------------------------------------------------
    # Generate Prediction Probability
    # --------------------------------------------------------

    probability = model.predict_proba(input_data)[0][1]


    # --------------------------------------------------------
    # Apply Final Classification Threshold
    # --------------------------------------------------------

    prediction = int(
        probability >= final_threshold
    )


    # ========================================================
    # Display Results
    # ========================================================

    st.divider()

    st.subheader("Prediction Results")


    # --------------------------------------------------------
    # Classification Result
    # --------------------------------------------------------

    if prediction == 1:

        st.error(
            "Predicted Class: Higher Risk of 30-Day Readmission"
        )

    else:

        st.success(
            "Predicted Class: Lower Risk of 30-Day Readmission"
        )


    # --------------------------------------------------------
    # Probability
    # --------------------------------------------------------

    st.metric(
        label="Estimated Readmission Probability",
        value=f"{probability:.2%}"
    )


    st.caption(
        f"Classification threshold: {final_threshold:.2f}"
    )


    # ========================================================
    # Risk Interpretation
    # ========================================================

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


    # ========================================================
    # Model Information
    # ========================================================

    st.divider()

    with st.expander("Model Information"):

        st.write("**Model:** Logistic Regression")

        st.write("**Classification Threshold:** 0.50")

        st.write("**Recall:** 59.0%")

        st.write("**Precision:** 20.9%")

        st.write("**F1-Score:** 30.9%")

        st.write("**ROC-AUC:** 69.8%")

        st.write("**PR-AUC:** 25.1%")

        st.caption(
            "Performance metrics are based on the held-out test set."
        )

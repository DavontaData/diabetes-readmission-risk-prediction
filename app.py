import streamlit as st
import pandas as pd
import joblib


# ============================================================
# Load Final Model Artifacts
# ============================================================

model = joblib.load(
    "final_calibrated_logistic_model.pkl"
)

model_features = joblib.load(
    "final_model_features.pkl"
)

final_threshold = joblib.load(
    "final_threshold.pkl"
)


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

st.title(
    "Diabetes 30-Day Readmission Risk Prediction"
)

st.write(
    "Predict 30-day hospital readmission risk among diabetic "
    "emergency inpatient admissions using a calibrated Logistic "
    "Regression machine-learning model. Enter patient encounter "
    "characteristics to receive an estimated readmission probability "
    "and risk classification."
)


# ============================================================
# Intended Use Disclaimer
# ============================================================

st.warning(
    """
    **Research / Educational Prototype**

    This application is intended for research, educational, and
    portfolio demonstration purposes only. The model was developed
    using historical hospital data from 1999–2008 and has not been
    externally validated using contemporary healthcare data.

    This application should not be used for autonomous clinical
    decision-making or as a substitute for professional clinical
    judgment.
    """
)


# ============================================================
# Patient Encounter Information
# ============================================================

st.subheader(
    "Patient Encounter Information"
)


age = st.number_input(
    "Age",
    min_value=5,
    max_value=100,
    value=50
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
# Clinical Characteristics
# ============================================================

st.subheader(
    "Clinical Characteristics"
)


a1c_result = st.selectbox(
    "A1C Result",
    [
        "None",
        "Norm",
        ">7",
        ">8"
    ]
)


max_glu_serum = st.selectbox(
    "Maximum Glucose Serum",
    [
        "None",
        "Norm",
        ">200",
        ">300"
    ]
)


insulin = st.selectbox(
    "Insulin Treatment",
    [
        "No",
        "Steady",
        "Up",
        "Down"
    ]
)


change_medication = st.selectbox(
    "Medication Change",
    [
        "No",
        "Yes"
    ]
)


# ============================================================
# Prediction
# ============================================================

if st.button(
    "Predict Readmission Risk"
):

    # ========================================================
    # Create Input DataFrame
    # ========================================================

    input_data = pd.DataFrame(
        0,
        index=[0],
        columns=model_features
    )


    # ========================================================
    # Numerical Features
    # ========================================================

    if "age" in input_data.columns:
        input_data["age"] = age


    if "number_emergency" in input_data.columns:
        input_data["number_emergency"] = number_emergency


    if "time_in_hospital" in input_data.columns:
        input_data["time_in_hospital"] = time_in_hospital


    if "number_diagnoses" in input_data.columns:
        input_data["number_diagnoses"] = number_diagnoses


    if "number_medications" in input_data.columns:
        input_data["number_medications"] = number_medications


    if "number_lab_procedures" in input_data.columns:
        input_data["number_lab_procedures"] = (
            number_lab_procedures
        )


    # ========================================================
    # Engineered Features
    # ========================================================

    medications_per_day = (
        number_medications /
        max(time_in_hospital, 1)
    )

    labs_per_day = (
        number_lab_procedures /
        max(time_in_hospital, 1)
    )

    diagnoses_per_day = (
        number_diagnoses /
        max(time_in_hospital, 1)
    )


    if "medications_per_day" in input_data.columns:
        input_data["medications_per_day"] = (
            medications_per_day
        )


    if "labs_per_day" in input_data.columns:
        input_data["labs_per_day"] = (
            labs_per_day
        )


    if "diagnoses_per_day" in input_data.columns:
        input_data["diagnoses_per_day"] = (
            diagnoses_per_day
        )


    # ========================================================
    # A1C One-Hot Encoding
    # ========================================================

    a1c_column = f"a1c_result_{a1c_result}"

    if a1c_column in input_data.columns:
        input_data[a1c_column] = 1


    # ========================================================
    # Maximum Glucose Serum One-Hot Encoding
    # ========================================================

    glucose_column = (
        f"max_glu_serum_{max_glu_serum}"
    )

    if glucose_column in input_data.columns:
        input_data[glucose_column] = 1


    # ========================================================
    # Insulin One-Hot Encoding
    # ========================================================

    insulin_column = (
        f"insulin_{insulin}"
    )

    if insulin_column in input_data.columns:
        input_data[insulin_column] = 1


    # ========================================================
    # Medication Change One-Hot Encoding
    # ========================================================

    medication_column = (
        f"change_medication_{change_medication}"
    )

    if medication_column in input_data.columns:
        input_data[medication_column] = 1


    # ========================================================
    # Ensure Exact Feature Order
    # ========================================================

    input_data = input_data.reindex(
        columns=model_features,
        fill_value=0
    )


    # ========================================================
    # Generate Probability
    # ========================================================

    probability = model.predict_proba(
        input_data
    )[0, 1]


    # ========================================================
    # Apply Final Classification Threshold
    # ========================================================

    prediction = int(
        probability >= final_threshold
    )


    # ========================================================
    # Prediction Results
    # ========================================================

    st.divider()

    st.subheader(
        "Prediction Results"
    )


    st.metric(
        label="Estimated 30-Day Readmission Probability",
        value=f"{probability:.2%}"
    )


    st.caption(
        f"Final classification threshold: "
        f"{final_threshold:.2f}"
    )


    if prediction == 1:

        st.error(
            "Predicted Class: Higher Risk of "
            "30-Day Readmission"
        )

    else:

        st.success(
            "Predicted Class: Lower Risk of "
            "30-Day Readmission"
        )


    # ========================================================
    # Risk Tier
    # ========================================================

    st.divider()

    st.subheader(
        "Risk Tier"
    )


    if probability < 0.10:

        risk_tier = "Lower Risk"

    elif probability < 0.30:

        risk_tier = "Moderate Risk"

    elif probability < 0.50:

        risk_tier = "High Risk"

    else:

        risk_tier = "Very High Risk"


    st.write(
        f"**Risk Tier:** {risk_tier}"
    )


    # ========================================================
    # Risk Interpretation
    # ========================================================

    st.divider()

    st.subheader(
        "Risk Interpretation"
    )


    if prediction == 1:

        st.warning(
            """
            The model estimates that this encounter has a
            predicted probability at or above the final
            0.10 classification threshold.

            The result should be interpreted as a
            risk-stratification output and not as a clinical
            diagnosis.
            """
        )

    else:

        st.info(
            """
            The model estimates that this encounter has a
            predicted probability below the final 0.10
            classification threshold.

            The result should be interpreted as a
            risk-stratification output and not as a clinical
            diagnosis.
            """
        )


    # ========================================================
    # Model Information
    # ========================================================

    st.divider()

    with st.expander(
        "Model Information"
    ):

        st.write(
            "**Model:** Calibrated Logistic Regression"
        )

        st.write(
            "**Calibration:** Sigmoid / Platt Scaling"
        )

        st.write(
            "**Cross-Validation:** 5-Fold"
        )

        st.write(
            f"**Classification Threshold:** "
            f"{final_threshold:.2f}"
        )

        st.write(
            "**Accuracy:** 45.4%"
        )

        st.write(
            "**Precision:** 13.9%"
        )

        st.write(
            "**Recall:** 73.5%"
        )

        st.write(
            "**F1-Score:** 0.234"
        )

        st.write(
            "**ROC-AUC:** 0.626"
        )

        st.write(
            "**PR-AUC:** 0.175"
        )

        st.caption(
            "Performance metrics are based on the held-out "
            "test set."
        )


# ============================================================
# Application Footer
# ============================================================

st.divider()

st.caption(
    "Research / Educational Prototype — "
    "Not for autonomous clinical decision-making."
)

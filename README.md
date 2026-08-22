# Predicting 30-Day Hospital Readmission Risk Among Diabetic Emergency Inpatient Admissions Using Machine Learning

### An End-to-End Healthcare Data Science Project Using SQL, Python, Machine Learning, Tableau, and Streamlit

**Live Streamlit Application:**  
https://diabetes-readmission-prediction-x7rqhrfmcyz3r2t88rkhzz.streamlit.app/

**Tableau Public Dashboard:**  
https://public.tableau.com/views/Diabetes_Readmission_Risk_Dashboard/DiabetesReadmissionRiskDashboard

---

## Project Article

For a detailed walkthrough of the clinical problem, data preparation, exploratory analysis, feature engineering, machine learning, model evaluation, interpretation, probability calibration, threshold optimization, and deployment:

**Read the Full Project Walkthrough on Medium:**  
https://medium.com/@davontacarruth/building-a-machine-learning-model-to-predict-30-day-hospital-readmissions-for-patients-with-83f11cbb3ab9

---

# Clinical Problem

Hospital readmissions within 30 days are an important healthcare challenge because they may indicate opportunities to improve care coordination, medication management, patient education, and chronic disease management.

Patients with diabetes may experience complex clinical and healthcare-utilization patterns that contribute to repeated hospital encounters.

This project focuses on identifying diabetic emergency inpatient encounters associated with **30-day hospital readmission** using demographic, clinical, and healthcare-utilization data.

---

# Business Problem

Healthcare organizations need analytical tools that can help identify encounters associated with elevated readmission risk so that care teams can potentially prioritize follow-up and care-management efforts.

This project demonstrates how healthcare data and machine learning can be used to:

- Analyze readmission patterns
- Identify predictive signals
- Address class imbalance
- Evaluate different classification strategies
- Generate encounter-level risk predictions
- Demonstrate a risk-stratification and clinical decision-support prototype

The model is intended for **research, educational, and portfolio demonstration purposes**, not autonomous clinical decision-making.

---

# Study Objective

To identify demographic, clinical, and healthcare-utilization factors associated with 30-day hospital readmission among diabetic emergency inpatient admissions and develop a machine-learning workflow for predicting readmission risk.

---

# Research Question

> **Which demographic, clinical, and healthcare-utilization factors are associated with 30-day hospital readmission among diabetic emergency inpatient admissions?**

---

# Dataset Overview

This project uses the **Diabetes 130-US Hospitals Dataset**, which contains historical inpatient hospital encounter data from patients with diabetes collected from 130 U.S. hospitals between 1999 and 2008.

The original dataset contains **101,766 hospital encounters** and **50 variables**, including demographic, clinical, medication, admission, laboratory, and healthcare-utilization information.

A SQL-based clinical cohort was created to focus specifically on **diabetic emergency inpatient admissions**.

Encounters were identified using diabetes-related ICD-9 diagnosis codes beginning with `250` and filtered to emergency admissions.

### Dataset Information

| Attribute | Value |
|---|---:|
| Original dataset | 101,766 encounters |
| Hospitals | 130 U.S. hospitals |
| Time period | 1999–2008 |
| Original variables | 50 |
| Final analytical cohort | 19,689 encounters |
| Final analytical variables | 26 |
| Population | Diabetic emergency inpatient admissions |
| Prediction outcome | 30-day hospital readmission |

### Outcome Definition

The original readmission variable contained:

- `NO`
- `>30`
- `<30`

For this project:

- `<30` = **1**, readmitted within 30 days
- `NO` or `>30` = **0**, not readmitted within 30 days

The final cohort contained:

- **17,458 non-readmission encounters (88.67%)**
- **2,231 30-day readmission encounters (11.33%)**

This produced a substantially imbalanced classification problem.

---

# Technology Stack

- Python
- SQL
- Microsoft SQL Server
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Jupyter Notebook
- VS Code
- Tableau Public
- Streamlit
- Joblib
- GitHub

The project combines SQL Server and Python to create an end-to-end healthcare data science workflow.

The notebook connects to SQL Server, extracts the clinical cohort using SQL, performs data cleaning and validation, conducts exploratory analysis, engineers machine-learning features, trains and evaluates models, interprets model results, and saves deployment artifacts.

---

# Clinical Cohort Definition

The study population consisted of diabetic inpatient encounters admitted through the emergency department.

The cohort was defined using diabetes-related ICD-9 diagnosis codes beginning with `250%` in:

- `diag_1`
- `diag_2`
- `diag_3`

and:

```sql
admission_type_id = 1

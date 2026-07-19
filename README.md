# Diabetes 30-Day Readmission Risk Prediction

Machine learning pipeline to predict 30-day hospital readmission risk among patients with diabetes using clinical and healthcare utilization data.

Live Streamlit Application:
https://diabetes-readmission-prediction-x7rqhrfmcyz3r2t88rkhzz.streamlit.app/

## Project Overview

This project develops a machine learning pipeline to identify factors associated with 30-day hospital readmission among diabetic inpatient encounters admitted through the emergency department.

## Study Objective

The objective of this study was to identify demographic, clinical, and healthcare utilization factors associated with 30-day readmission among diabetic inpatient encounters admitted through the emergency department and develop a machine-learning-ready dataset for predictive modeling.

The goal was to analyze demographic, clinical, and healthcare utilization characteristics associated with readmission risk and construct a machine-learning-ready dataset for predictive modeling.

## Research Question

Which demographic, clinical, and healthcare utilization factors are associated with 30-day hospital readmission among diabetic inpatient encounters admitted through the emergency department?

## Dataset

Dataset:
UCI Diabetes 130-US Hospitals for Years 1999-2008

The dataset contains hospital encounters for patients with diabetes and includes:

* Demographic information
* Laboratory measurements
* Medication information
* Healthcare utilization variables
* Admission characteristics
* Prediction Target

### Outcome	Definition
- 1	Readmitted within 30 days
- 0	Not readmitted within 30 days

## Clinical Cohort Definition

The study population consisted of diabetic inpatient encounters admitted through the emergency department.

Patients were identified using ICD-9 diabetes diagnosis codes beginning with "250%" and filtered to emergency admission encounters.

### Patients were identified using:

* ICD-9 diabetes diagnosis codes beginning with "250%"
* Admission characteristics relevant to acute care utilization

## Machine Learning Workflow

The project followed an end-to-end clinical machine learning workflow:

1. Data extraction and cleaning
2. Missing value and data quality assessment
3. Exploratory data analysis
4. Feature engineering
5. Train/test split
6. Model development
7. Model evaluation
8. Clinical interpretation
9. Streamlit deployment
10. Feature Engineering

Final model features included:

### Patient Characteristics
* Age
  
### Healthcare Utilization

* Prior inpatient visits
* Emergency visits
* Length of hospital stay
  
### Clinical Indicators

* A1C results
* Maximum glucose serum
* Insulin treatment
* Medication changes

Categorical variables were transformed using one-hot encoding before model training.

## Machine Learning Model

The final prediction model used:

### Logistic Regression

Selected because:

* Provides probability-based predictions
* Offers interpretability
* Is appropriate for clinical risk prediction applications
  
## Model Evaluation

The model was evaluated using classification performance metrics including:

* Accuracy
* Precision
* Recall
* ROC-AUC (if calculated)

Evaluation results are available in the project notebook.

(Update this section later with your exact numbers after we review the notebook.)


## Streamlit Deployment

The trained model was deployed as an interactive Streamlit application.

Users can enter patient characteristics including:

* Age
* Prior inpatient visits
* Emergency visits
* Hospital stay duration
* A1C status
* Glucose measurement
* Insulin treatment
* Medication changes

The application provides:

* Predicted readmission class
* Estimated probability of 30-day readmission

## Limitations

* The dataset represents historical hospital encounters and may not reflect current healthcare practices.
* External validation would be required before clinical implementation.
* Predictions should support, not replace, clinical judgment.

## Future Improvements

Potential future work:

* Validate model using newer healthcare datasets
* Incorporate clinical notes using NLP
* Add explainability methods such as SHAP
* Compare additional machine learning approaches
* Explore integration into clinical decision-support workflows

## Technologies Used

* Python
* Pandas
* NumPy
* Scikit-learn
* SQL
* Jupyter Notebook
* Streamlit
* GitHub

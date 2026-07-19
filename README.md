# Diabetes 30-Day Readmission Risk Prediction

Machine learning pipeline to predict 30-day hospital readmission risk among patients with diabetes using clinical and healthcare utilization data.

Live Streamlit Application:
https://diabetes-readmission-prediction-x7rqhrfmcyz3r2t88rkhzz.streamlit.app/

## Project Overview

This project develops a machine learning pipeline to identify factors associated with 30-day hospital readmission among diabetic inpatient encounters with the emergency department.

## Study Objective

The objective of this study was to identify demographic, clinical, and healthcare utilization factors associated with 30-day readmission among diabetic inpatient encounters admitted through the emergency department and develop a machine-learning-ready dataset for predictive modeling.

The goal was to analyze demographic, clinical, and healthcare utilization characteristics associated with readmission risk and construct a machine-learning-ready dataset for predictive modeling.

## Research Question

Which demographic, clinical, and healthcare utilization factors are associated with 30-day hospital readmission among diabetic inpatient encounters admitted through the emergency department?

## Supporting Questions

1. What patient and hospitalization characteristics are associated with increased readmission risk?

2. How does prior healthcare utilization (previous inpatient and emergency visits) relate to 30-day readmission?

3. Which clinical features contribute most to predicting readmission risk?

4. How well do different machine learning models perform in identifying high-risk patients?


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

## Clinical Cohort Definition

The study population consisted of diabetic inpatient encounters admitted through the emergency department.

Patients were identified using ICD-9 diabetes diagnosis codes beginning with "250%" and filtered to emergency admission encounters.

### Outcome	Definition
- 1	Readmitted within 30 days
- 0	Not readmitted within 30 days

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

## Final Model Performance

Logistic Regression was selected as the final model because it had the strongest ROC-AUC and provided interpretable risk estimates.

| Metric | Result |
|---|---:|
| Accuracy | 69.8% |
| Precision | 18.9% |
| Recall | 50.4% |
| F1 Score | 0.275 |
| ROC-AUC | 0.672 |

Because 30-day readmissions were relatively uncommon, accuracy was interpreted alongside recall, precision, F1 score, and ROC-AUC. The model identified approximately half of readmissions, but its positive predictions had limited precision.

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

## Repository Structure

├── app.py
├── Diabetes_Readmission_Project.ipynb
├── diabetes_readmission_model.pkl
├── model_features.pkl
├── diabetes_final_ml_dataset.csv
├── requirements.txt
├── images/
└── README.md

## Streamlit Application

The final machine learning model was deployed as an interactive Streamlit application.

### Prediction Interface

![Streamlit Input](images/streamlit_input.png)

![Streamlit Input](images/streamlit_input.png.2)

### Prediction Output

![Prediction Result](images/streamlit_prediction.png)

## Tableau Dashboard

The Tableau dashboard summarizes readmission patterns by age, prior inpatient visits,
emergency visits, A1C result, and medication-change status.

<img width="1359" height="709" alt="Screenshot 2026-07-19 181250" src="https://github.com/user-attachments/assets/0f4a1f89-9529-426d-b2dc-66da440d4999" />


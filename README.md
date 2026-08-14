# Predicting 30-Day Hospital Readmission Risk Among Diabetic Emergency Inpatient Admissions Using Machine Learning

Live Streamlit Application:
https://diabetes-readmission-prediction-x7rqhrfmcyz3r2t88rkhzz.streamlit.app/

Tableau Public link:
https://public.tableau.com/views/Diabetes_Readmission_Risk_Dashboard/DiabetesReadmissionRiskDashboard?:language=en-US&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link

## Project Article

For a detailed walkthrough of the project, including the clinical problem, data preparation, exploratory analysis, feature engineering, machine learning models, evaluation, interpretation, and deployment:

**[Read the Full Project Walkthrough on Medium](https://medium.com/@davontacarruth/building-a-machine-learning-model-to-predict-30-day-hospital-readmissions-for-patients-with-83f11cbb3ab9)**

## Clinical Problem

Hospital readmissions within 30 days are an important healthcare challenge because they may indicate opportunities to improve care coordination, medication management, patient education, and chronic disease management.

Patients with diabetes may experience complex clinical and healthcare-utilization patterns that can contribute to repeated hospital encounters.

This project focuses on identifying diabetic emergency inpatient encounters associated with increased risk of 30-day hospital readmission using demographic, clinical, and healthcare-utilization data.

## Business Problem

Healthcare organizations need analytical tools that can help identify encounters associated with higher readmission risk so that care teams can potentially prioritize follow-up and care-management efforts.

This project demonstrates how healthcare data and machine learning can be used to analyze readmission patterns, identify predictive signals, and develop a risk-stratification prototype that could support population health and clinical analytics use cases.

## Study Objective

To identify demographic, clinical, and healthcare-utilization factors associated with 30-day hospital readmission among diabetic emergency inpatient admissions and construct a machine-learning-ready dataset for predictive modeling.

## Research Question

Which demographic, clinical, and healthcare-utilization factors are associated with 30-day hospital readmission among diabetic emergency inpatient admissions?

## Dataset Overview

This project uses the **Diabetes 130-US Hospitals dataset**, which contains historical inpatient hospital encounter data from patients with diabetes collected from 130 U.S. hospitals between 1999 and 2008.

The original dataset contains **101,766 hospital encounters** and includes demographic, clinical, medication, admission, laboratory, and healthcare-utilization variables.

For this project, a SQL-based clinical cohort was created to focus specifically on **diabetic emergency inpatient admissions**. Encounters were identified using diabetes-related ICD-9 diagnosis codes beginning with **250** and filtered to emergency admissions.

### Dataset Information

* **Original dataset:** 101,766 hospital encounters
* **Hospitals:** 130 U.S. hospitals
* **Time period:** 1999–2008
* **Original variables:** 50
* **Final analytical cohort:** 19,689 hospital encounters
* **Final analytical variables:** 26
* **Population:** Diabetic emergency inpatient admissions
* **Prediction outcome:** 30-day hospital readmission
* **Target:** `<30` readmission = 1; `NO` or `>30` = 0

The final analytical cohort contained **19,689 encounters and 26 variables** after the SQL cohort extraction. :contentReference[oaicite:6]{index=6}

The dataset was used to examine demographic, clinical, and healthcare-utilization factors associated with 30-day readmission and to develop a machine-learning-ready dataset for predictive modeling.

## Tools Used

- Python
- SQL Server
- SQL
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Jupyter Notebook
- VS Code
- Tableau Public
- Streamlit
- GitHub
- Joblib

The project combines SQL Server and Python to create an end-to-end clinical data science workflow.

The notebook connects to SQL Server, extracts the cohort using SQL, performs data cleaning and validation, conducts exploratory analysis, engineers machine-learning features, trains and evaluates models, interprets results, and saves the final model artifacts.

The dataset contains:

* Demographic information
* Laboratory measurements
* Medication information
* Healthcare-utilization variables
* Admission characteristics
* Clinical indicators
* The 30-day readmission prediction target

## Clinical Cohort Definition

The study population consisted of diabetic inpatient encounters admitted through the emergency department.

Patients were identified using ICD-9 diabetes diagnosis codes beginning with `250%` in `diag_1`, `diag_2`, or `diag_3` and filtered to emergency admissions using:

`admission_type_id = 1`

This produced a cohort of:

**19,689 emergency inpatient encounters**

### Outcome Definition

| Outcome | Definition |
|---|---|
| 1 | Readmitted within 30 days |
| 0 | Not readmitted within 30 days |

The final outcome distribution was:

* **17,458 non-readmission encounters (88.67%)**
* **2,231 30-day readmission encounters (11.33%)**

This class imbalance was considered throughout model development and evaluation.

## Project Workflow

This project follows an end-to-end clinical data science and machine-learning workflow:

1. Connect to SQL Server
2. Perform data quality checks and validation
3. Clean and standardize the data
4. Extract the diabetes emergency-inpatient cohort
5. Perform exploratory data analysis (EDA)
6. Analyze demographic, clinical, and healthcare-utilization patterns
7. Engineer features for machine learning
8. Create the machine-learning dataset
9. Split data into training and testing sets
10. Build baseline predictive models
11. Perform hyperparameter tuning
12. Evaluate classification thresholds
13. Compare model performance
14. Interpret the final model
15. Save model artifacts
16. Deploy the model using Streamlit

## Data Cleaning

The data-preprocessing workflow included:

* Handling missing values and placeholder values
* Standardizing categorical variables
* Converting variables to appropriate data types
* Converting age ranges into numeric representations
* Grouping diagnosis information into broader clinical categories
* Reviewing missingness and data quality
* Reviewing potential numerical outliers
* Removing sparse or potentially problematic variables
* Evaluating potential information leakage

Potential numerical outliers were reviewed using the Interquartile Range (IQR) approach. Clinically plausible extreme values were retained rather than automatically removed.

A particularly important modeling decision involved potential temporal leakage. A patient-level observed-encounter feature was evaluated but was not used as a final ML predictor because the available encounter information did not provide sufficient temporal detail to confirm whether those encounters occurred before the prediction point.

## Exploratory Data Analysis

Exploratory analysis examined:

* Age
* Gender
* Race
* Hospital length of stay
* Emergency utilization
* Inpatient utilization
* Number of diagnoses
* Number of medications
* Laboratory procedures
* A1C results
* Maximum glucose
* Insulin treatment
* Medication changes

Healthcare-utilization patterns showed important differences between encounters with and without 30-day readmission.

### Observed Encounter Utilization

The analysis found:

* **1.51 average observed inpatient encounters** among encounters without 30-day readmission
* **2.81 average observed inpatient encounters** among encounters with 30-day readmission

The median was also higher among readmitted encounters:

* **2 vs. 1 observed encounters**

These values represent **observed encounters within the study cohort**, not confirmed prior encounters, because encounter timestamps were not available to establish whether the encounters occurred before or after the current hospitalization. :contentReference[oaicite:7]{index=7}

## Feature Engineering

Feature engineering was used to transform the cleaned healthcare data into variables suitable for machine learning.

The final machine-learning feature set contained **13 predictors before one-hot encoding**:

### Patient Characteristics

* Age

### Healthcare Utilization and Clinical Complexity

* Number of emergency visits
* Length of hospital stay
* Number of diagnoses
* Number of medications
* Medications per day
* Number of laboratory procedures
* Labs per day
* Diagnoses per day

### Clinical and Treatment Indicators

* A1C results
* Maximum serum glucose
* Insulin treatment
* Medication changes

The final feature list used by the notebook is explicitly defined as these 13 predictors. :contentReference[oaicite:8]{index=8}

Categorical variables were transformed using one-hot encoding.

After encoding, the feature matrix contained:

**19,689 observations × 23 predictors**

The 23 encoded predictors included the numerical variables above along with one-hot encoded A1C, maximum glucose, insulin, and medication-change categories. :contentReference[oaicite:9]{index=9}

## Machine Learning Dataset

The final machine-learning dataset contained:

* **19,689 observations**
* **13 predictors before encoding**
* **23 predictors after one-hot encoding**
* **1 binary target variable**

The target variable was:

`readmission_flag`

where:

* `0` = no 30-day readmission
* `1` = 30-day readmission

## Train/Test Split

The dataset was divided using an 80/20 stratified train/test split.

### Training Set

**15,751 encounters**

### Testing Set

**3,938 encounters**

Stratification was used to preserve the class distribution between the training and testing datasets.

## Machine Learning Models

Three supervised classification algorithms were evaluated:

* Logistic Regression
* Random Forest
* XGBoost

Logistic Regression was used because of its interpretability and ability to provide probability-based predictions.

Random Forest was evaluated as a nonlinear ensemble method.

XGBoost was evaluated as a gradient-boosting approach for structured healthcare data.

## Handling Class Imbalance

Because 30-day readmission represented only **11.33%** of the cohort, class imbalance was a major modeling consideration.

The project evaluated:

* Class weighting
* Positive-class weighting
* Precision
* Recall
* F1-score
* ROC-AUC
* PR-AUC
* Classification-threshold performance

Accuracy was not treated as the only measure of model quality.

## Baseline Model Performance

The baseline models produced the following results on the held-out test dataset:

| Model | Accuracy | Precision | Recall | F1 Score | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 70.1% | 20.9% | 59.0% | 0.309 | 0.698 | 0.251 |
| Random Forest | 88.5% | 31.6% | 1.3% | 0.026 | 0.638 | 0.182 |
| XGBoost | 76.0% | 19.5% | 35.9% | 0.253 | 0.641 | 0.180 |

The baseline model comparison demonstrates why accuracy alone can be misleading for an imbalanced healthcare classification problem.

For example, Random Forest achieved approximately **88.5% accuracy**, but recall for the minority readmission class was only approximately **1.3%**.

Logistic Regression provided substantially stronger minority-class recall and F1-score.

The notebook's baseline model comparison confirms these results. :contentReference[oaicite:10]{index=10}

## Hyperparameter Tuning

Hyperparameter tuning was performed for:

* Logistic Regression
* Random Forest
* XGBoost

The tuned models were compared against their baseline configurations.

An important finding was that hyperparameter tuning did not consistently improve minority-class performance.

For Logistic Regression:

* Baseline recall: **59.0%**
* Tuned recall: **57.0%**
* Baseline F1: **0.309**
* Tuned F1: **0.246**

For Random Forest:

* Baseline recall: **1.3%**
* Tuned recall: **30.0%**
* Baseline F1: **0.026**
* Tuned F1: **0.212**

For XGBoost:

* Baseline recall: **35.9%**
* Tuned recall: **0.2%**
* Baseline F1: **0.253**
* Tuned F1: **0.004**

The results demonstrated that tuning improved some minority-class metrics for Random Forest but reduced recall for Logistic Regression and XGBoost. :contentReference[oaicite:11]{index=11}

## Final Model Selection

The **baseline Logistic Regression model** was selected as the final classification model.

The final model was selected because it provided:

* The highest recall among the evaluated configurations
* The highest F1-score among the evaluated configurations
* Strong ROC-AUC performance
* Greater interpretability compared with the tree-based models

The model-selection process placed greater emphasis on recall and F1-score because 30-day readmission was the minority class and the primary objective was to identify encounters at higher risk of readmission.

The final classification threshold was:

**0.50**

The notebook explicitly identifies baseline Logistic Regression as the final recommended model. :contentReference[oaicite:12]{index=12}

## Final Model Performance

The final baseline Logistic Regression model achieved the following performance on the held-out test set:

| Metric | Result |
|---|---:|
| Accuracy | **70.1%** |
| Precision | **20.9%** |
| Recall | **59.0%** |
| F1 Score | **0.309** |
| ROC-AUC | **0.698** |
| PR-AUC | **0.251** |

The model's performance demonstrates moderate discrimination while identifying approximately 59% of the actual 30-day readmission encounters.

However, the relatively low precision means that a substantial number of encounters predicted as readmissions did not actually experience readmission.

## Final Confusion Matrix

The test set contained:

**446 actual 30-day readmission encounters**

The final model:

* Correctly identified **263** readmission encounters
* Missed **183** readmission encounters
* Correctly identified **2,499** non-readmission encounters
* Produced **993 false-positive predictions**

| | Predicted No Readmission | Predicted Readmission |
|---|---:|---:|
| Actual No Readmission | 2,499 | 993 |
| Actual Readmission | 183 | 263 |

The model therefore identified approximately **59% of actual readmission encounters**.

Because precision was approximately 20.9%, the model should be viewed as a **risk-stratification and decision-support prototype**, not as a definitive clinical predictor. :contentReference[oaicite:13]{index=13}

## Model Interpretation

Logistic Regression was selected partly because its coefficients provide an interpretable way to examine predictive relationships.

The strongest positive associations in the final model were related to:

* Emergency healthcare utilization
* Number of diagnoses
* Hospital length of stay

The notebook specifically identifies emergency visits, number of diagnoses, and time in hospital as important positive associations in the final Logistic Regression model. :contentReference[oaicite:14]{index=14}

These relationships represent associations within the predictive model and should not be interpreted as causal effects.

## Probability Calibration

The project also evaluated probability calibration.

The baseline Logistic Regression model had a Brier score of:

**0.2376**

After calibration, the Brier score improved to:

**0.0986**

Lower Brier scores indicate better probability calibration.

This analysis demonstrated that the predicted probabilities could potentially be improved through calibration, but the calibrated model was not used to replace the final baseline Logistic Regression configuration for this project. :contentReference[oaicite:15]{index=15} :contentReference[oaicite:16]{index=16}

## Streamlit Deployment

The trained model was deployed as an interactive Streamlit application.

Users can enter encounter characteristics including:

* Age
* Emergency visits
* Hospital stay duration
* Number of diagnoses
* Number of medications
* Number of laboratory procedures
* A1C status
* Maximum glucose measurement
* Insulin treatment
* Medication changes

The application provides:

* Predicted readmission classification
* Estimated probability of 30-day readmission

The trained Logistic Regression model, model feature structure, and classification threshold were saved using Joblib so the application can load the trained model without retraining it.

The Streamlit application demonstrates how the machine learning workflow can be moved from the Jupyter Notebook environment into an interactive prediction interface.

## Tableau Dashboard

A Tableau dashboard was also created to complement the machine learning workflow.

The dashboard provides exploratory analysis of clinical and healthcare-utilization patterns associated with diabetes readmission.

Key areas include:

* A1C categories
* Maximum serum glucose
* Observed inpatient utilization
* Emergency utilization
* Hospital length of stay
* Medication changes

The Tableau dashboard provides a descriptive/exploratory perspective, while the Streamlit application provides a predictive interface.

## Modeling Limitations

The dataset exhibits substantial class imbalance, with 30-day readmissions representing only **11.33%** of the final cohort.

This imbalance makes minority-class prediction challenging and limits the usefulness of accuracy as a standalone evaluation metric.

The dataset also represents historical hospital encounters from **1999–2008**, so model performance may not generalize to modern healthcare environments.

The final model also produced a substantial number of false-positive predictions, resulting in relatively low precision.

Additional limitations include:

* Historical data
* Limited clinical variables
* Lack of external validation
* Lack of prospective validation
* Potential differences between historical and current healthcare workflows
* Lack of clinical workflow integration
* No established clinical intervention associated with a high-risk prediction

The model is intended for **educational and research purposes** and should support—not replace—clinical judgment.

## Future Improvements

Future work could focus on:

* External validation using an independent healthcare dataset
* Validation using more recent hospital data
* Probability calibration evaluation
* Clinically meaningful threshold selection
* Fairness and subgroup performance analysis
* Evaluation across demographic groups
* Prospective monitoring
* Clinical workflow integration
* Evaluation of false positives and false negatives with clinical stakeholders
* Defining an appropriate intervention associated with a high-risk prediction

Before any clinical implementation, the model would require additional validation and evaluation in an appropriate healthcare environment.

## Conclusion

This project demonstrates an end-to-end clinical data science workflow for investigating and predicting 30-day hospital readmission among diabetic emergency inpatient admissions.

The workflow included:

**SQL Server → Data Cleaning → Clinical Cohort Definition → EDA → Feature Engineering → Machine Learning → Model Comparison → Hyperparameter Tuning → Threshold Evaluation → Model Interpretation → Model Saving → Streamlit Deployment**

The final model was **baseline Logistic Regression** using a **0.50 classification threshold**.

It achieved:

* **70.1% accuracy**
* **20.9% precision**
* **59.0% recall**
* **0.309 F1-score**
* **0.698 ROC-AUC**
* **0.251 PR-AUC**

The model correctly identified **263 of 446 actual 30-day readmission encounters** while missing 183.

Logistic Regression was selected because it provided the strongest combination of minority-class recall, F1-score, ROC-AUC performance, and interpretability among the evaluated model configurations.

The project demonstrates that healthcare machine learning requires more than selecting an algorithm. Cohort definition, data quality, feature engineering, class imbalance, model evaluation, interpretability, calibration, and clinical validation are all important components of a responsible predictive modeling workflow.

## Repository Structure

Diabetes-Readmission-Prediction/
│
├── .devcontainer/
│   └── devcontainer.json
│
├── images/
│   ├── streamlit_overview_1.png
│   ├── streamlit_overview_2.png
│   ├── streamlit_prediction_3.png
│   ├── tableau_dashboard_1.png
│   └── tableau_dashboard_2.png
│
├── README.md
├── app.py
├── clinical_reference_diabetes_project.xlsx
├── diabetes_clean_dataset.xls
├── diabetes_final_logistic_regression.pkl
├── diabetes_final_ml_dataset.xls
├── diabetes_final_model_features.pkl
├── diabetes_final_threshold.pkl
├── diabetes_readmission_project.ipynb
├── diabetes_readmission_risk_dashboard.twbx
├── diabetes_readmission_risk_prediction_presentation.pptx
└── requirements.txt

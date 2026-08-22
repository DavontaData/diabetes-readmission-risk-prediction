# Predicting 30-Day Hospital Readmission Risk Among Diabetic Emergency Inpatient Admissions Using Machine Learning

### End-to-End Healthcare Data Science Project Using SQL Server, Python, Machine Learning, Tableau, and Streamlit

**Live Streamlit Application:**  
https://diabetes-readmission-prediction-x7rqhrfmcyz3r2t88rkhzz.streamlit.app/

**Tableau Public Dashboard:**  
https://public.tableau.com/views/Diabetes_Readmission_Risk_Dashboard/DiabetesReadmissionRiskDashboard

**Medium Project Walkthrough:**  
https://medium.com/@davontacarruth/building-a-machine-learning-model-to-predict-30-day-hospital-readmissions-for-patients-with-83f11cbb3ab9

---

## Project Overview

Hospital readmissions within 30 days are an important healthcare challenge because they may indicate opportunities to improve care coordination, medication management, patient education, and chronic disease management.

Patients with diabetes can experience complex clinical and healthcare-utilization patterns that contribute to repeated hospital encounters.

This project investigates 30-day hospital readmission among diabetic inpatient encounters following emergency admission and develops a machine-learning workflow for identifying encounters with elevated predicted readmission risk.

The project combines SQL Server, Python, statistical analysis, machine learning, probability calibration, classification-threshold optimization, model interpretation, encounter-level risk tracking, Tableau visualization, and Streamlit deployment.

### End-to-End Workflow


SQL Server
    ↓
Data Extraction
    ↓
Data Quality Assessment
    ↓
Data Cleaning
    ↓
Clinical Cohort Definition
    ↓
Exploratory Data Analysis
    ↓
Feature Engineering
    ↓
Machine Learning Dataset
    ↓
Train/Test Split
    ↓
Baseline Models
    ↓
Hyperparameter Tuning
    ↓
Probability Calibration
    ↓
Threshold Optimization
    ↓
Final Model Selection
    ↓
Model Interpretation
    ↓
Encounter-Level Risk Tracking
    ↓
Model Deployment
    ↓
Streamlit Application
```

---

# Clinical Problem

Hospital readmissions within 30 days are an important healthcare challenge because they may indicate opportunities for improved care coordination, medication management, patient education, and chronic disease management.

This project focuses on diabetic inpatient encounters admitted through the emergency department and investigates whether demographic, clinical, and healthcare-utilization characteristics can help identify encounters associated with 30-day readmission.

---

# Business Problem

Healthcare organizations need analytical tools that can help identify encounters associated with elevated readmission risk so that care teams could potentially prioritize follow-up and care-management resources.

This project demonstrates how healthcare data and machine learning can be used to:

- Analyze readmission patterns
- Identify predictive signals
- Address class imbalance
- Compare multiple machine-learning algorithms
- Evaluate precision-recall trade-offs
- Calibrate predicted probabilities
- Optimize classification thresholds
- Generate encounter-level risk predictions
- Demonstrate a clinical decision-support prototype

The current application is intended for **research, educational, and portfolio demonstration purposes** and is not intended for autonomous clinical decision-making.

---

# Study Objective

The objective of this project is to identify demographic, clinical, and healthcare-utilization factors associated with 30-day hospital readmission among diabetic emergency inpatient admissions and develop a machine-learning workflow for predicting readmission risk.

---

# Research Question

> **Which demographic, clinical, and healthcare-utilization factors are associated with 30-day hospital readmission among diabetic emergency inpatient admissions?**

---

# Dataset Overview

This project uses the **Diabetes 130-US Hospitals Dataset**, a historical dataset containing inpatient encounters involving patients with diabetes.

The original dataset contains:

- **101,766 hospital encounters**
- **50 variables**
- Data from **130 U.S. hospitals**
- Historical data collected between **1999 and 2008**

A SQL-based clinical cohort was created to focus specifically on diabetic inpatient encounters admitted through the emergency department.

---

# Clinical Cohort Definition

The study population was defined using diabetes-related ICD-9 diagnosis codes and emergency admission status.

Diabetes-related encounters were identified when any of the following diagnosis fields began with `250`:

```text
diag_1
diag_2
diag_3
```

Emergency admissions were identified using:

```sql
admission_type_id = 1
```

The cohort query was:

```sql
SELECT *
FROM dbo.diabetes_data_clean
WHERE (
    diag_1 LIKE '250%'
    OR diag_2 LIKE '250%'
    OR diag_3 LIKE '250%'
)
AND admission_type_id = 1;
```

### Final Cohort

The final cohort contained:

- **19,689 emergency inpatient encounters**
- **26 cohort variables**
- **2,231 30-day readmission encounters**

The cohort definition and resulting dimensions were validated against the notebook workflow.

---

# Outcome Definition

The original dataset contains three readmission categories:

| Original Value | Meaning |
|---|---|
| `NO` | No readmission |
| `>30` | Readmitted after 30 days |
| `<30` | Readmitted within 30 days |

For machine learning, the outcome was converted into a binary variable:

```text
readmission_flag
```

### Target Definition

| Target | Definition |
|---:|---|
| `0` | No 30-day readmission |
| `1` | Readmission within 30 days |

The final cohort contained:

- **17,458 encounters (88.67%)** without 30-day readmission
- **2,231 encounters (11.33%)** with 30-day readmission

This created a substantially imbalanced classification problem.

---

# Data Quality and Preprocessing

The data-preparation workflow included:

- Data quality assessment
- Missing-value assessment
- Placeholder-value handling
- Data-type validation
- Categorical-variable standardization
- Age transformation
- Diagnosis grouping
- Review of numerical distributions
- Outlier assessment
- Leakage assessment
- Preparation of categorical variables for machine learning

Potential numerical outliers were reviewed using the Interquartile Range (IQR) approach.

Clinically plausible extreme values were not automatically removed simply because they were statistically unusual.

---

# Data Leakage Assessment

Potential information leakage was specifically considered during feature engineering.

Some healthcare-utilization variables were useful for exploratory analysis but were not appropriate for the final predictive model because the available data could not establish whether the information occurred before or after the prediction point.

This was an important modeling consideration because:

> A feature can be predictive and still be inappropriate for a machine-learning model if the information would not actually be available at prediction time.

Potentially leakage-prone variables were therefore excluded from the final predictive feature set when temporal availability could not be established.

---

# Exploratory Data Analysis

Exploratory analysis examined demographic, clinical, and healthcare-utilization characteristics including:

- Age
- Gender
- Race
- Hospital length of stay
- Emergency utilization
- Inpatient utilization
- Number of diagnoses
- Number of medications
- Laboratory procedures
- A1C results
- Maximum glucose
- Insulin treatment
- Medication changes

The analysis was used to understand the structure of the cohort, identify potential predictive signals, evaluate class imbalance, and guide feature engineering.

---

# Feature Engineering

Feature engineering transformed the cleaned healthcare data into variables appropriate for machine learning.

The final feature set included:

### Demographic

- Age

### Healthcare Utilization and Clinical Complexity

- Number of emergency visits
- Time in hospital
- Number of diagnoses
- Number of medications
- Number of laboratory procedures

### Derived Features

- Medications per day
- Laboratory procedures per day
- Diagnoses per day

### Clinical and Treatment Variables

- A1C result
- Maximum glucose
- Insulin
- Medication change

Categorical variables were converted into numerical representations using one-hot encoding.

The final feature set contained **13 predictors before one-hot encoding**.

After one-hot encoding, the final feature matrix contained:

**19,689 observations × 23 predictors**

---

# Machine Learning Dataset

The final machine-learning dataset contained:

- **19,689 observations**
- **13 predictors before encoding**
- **23 predictors after one-hot encoding**
- **1 binary target variable**

The target variable was:

```text
readmission_flag
```

where:

```text
0 = No 30-day readmission
1 = 30-day readmission
```

---

# Train/Test Split

The dataset was divided using an **80/20 stratified train/test split**.

### Training Set

**15,751 encounters**

### Testing Set

**3,938 encounters**

Stratification was used to maintain a similar class distribution between the training and testing datasets.

The test set was held out for model evaluation.

---

# Machine Learning Models

Three supervised classification algorithms were evaluated:

## Logistic Regression

Logistic Regression was selected as an interpretable statistical classification model capable of estimating the probability of 30-day readmission.

## Random Forest

Random Forest was evaluated because it can model nonlinear relationships and interactions among predictors.

## XGBoost

XGBoost was evaluated as a gradient-boosting approach capable of modeling complex relationships within structured healthcare data.

Comparing the three approaches allowed evaluation of the trade-off between interpretability and more complex machine-learning methods.

---

# Handling Class Imbalance

Only **11.33%** of encounters resulted in 30-day readmission.

Class imbalance was therefore an important modeling consideration.

For Logistic Regression and Random Forest:

```text
class_weight = "balanced"
```

The baseline XGBoost model used positive-class weighting.

The purpose of class weighting was to give greater importance to the minority readmission class during model training.

Because of the class imbalance, accuracy was not treated as the only measure of model quality.

The project emphasized:

- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Confusion matrices
- Probability calibration
- Classification-threshold performance

---

# Baseline Model Performance

The baseline models produced substantially different results.

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 70.5% | 20.6% | 56.1% | 0.301 | 0.710 | 0.259 |
| Random Forest | 88.5% | 35.3% | 1.3% | 0.026 | 0.612 | 0.171 |
| XGBoost | 74.4% | 15.7% | 28.7% | 0.203 | 0.589 | 0.159 |

The Random Forest result illustrates why accuracy alone can be misleading in an imbalanced healthcare classification problem.

Although Random Forest achieved approximately **88.5% accuracy**, it identified only **1.3%** of actual readmission cases.

Logistic Regression provided substantially stronger minority-class recall and F1-score.

---

# Hyperparameter Tuning

Hyperparameter tuning was performed for:

- Logistic Regression
- Random Forest
- XGBoost

### Logistic Regression

The tuning process evaluated combinations of:

- `C`
- Class weighting
- Solver

### Random Forest

The tuning process evaluated:

- Number of trees
- Maximum depth
- Minimum samples per leaf
- Class weighting

### XGBoost

The tuning process evaluated:

- Number of estimators
- Maximum depth
- Learning rate
- Subsampling
- Column sampling
- Class weighting

The models were tuned using:

```text
GridSearchCV
3-fold cross-validation
ROC-AUC optimization
```

Tuning did not consistently improve minority-class performance across all models.

---

# Probability Calibration

After comparing the baseline and tuned models, Logistic Regression was retained as the leading interpretable model for additional analysis.

Probability calibration was then performed using:

**5-fold sigmoid calibration (Platt scaling).**

Calibration evaluates whether predicted probabilities correspond reasonably to observed outcome frequencies.

The Brier score improved from:

| Model | Brier Score |
|---|---:|
| Baseline Logistic Regression | 0.2376 |
| Calibrated Logistic Regression | 0.0986 |

The lower Brier score indicated substantially improved probability reliability after calibration.

---

# Classification Threshold Optimization

A predicted probability does not automatically have to use a classification threshold of 0.50.

Because this project prioritizes identification of potential 30-day readmission encounters, the classification threshold was evaluated as a separate modeling decision.

Threshold optimization evaluated the trade-off between:

- Precision
- Recall
- F1-score

The calibrated Logistic Regression model was evaluated using out-of-fold probabilities.

The final operating threshold was selected as:

## **0.10**

At the 0.10 threshold using the out-of-fold predictions:

- **Precision:** 0.135
- **Recall:** 0.728
- **F1-score:** 0.228

The threshold was locked before evaluation on the held-out test set.

---

# Final Model Selection

The final classification configuration was:

## Calibrated Logistic Regression + 0.10 Threshold

The final model was selected because the project's primary objective was to identify as many potential 30-day readmission encounters as possible.

Although the baseline 0.50 configuration produced stronger precision, F1-score, ROC-AUC, PR-AUC, and accuracy, the calibrated 0.10 configuration produced substantially higher recall.

Therefore, the final decision prioritized **recall** over overall accuracy.

This was an objective-driven model-selection decision rather than simply selecting the model with the highest accuracy.

---

# Final Model

### Model


CalibratedClassifierCV

with:

Estimator:
LogisticRegression(
    class_weight="balanced",
    max_iter=1000,
    random_state=42
)

Calibration:
5-fold sigmoid calibration / Platt scaling

Classification threshold:
0.10
```

---

# Final Test-Set Performance

The calibrated Logistic Regression model with the locked **0.10 classification threshold** was evaluated on the held-out test set.

| Metric | Final Result |
|---|---:|
| Accuracy | **45.4%** |
| Precision | **13.9%** |
| Recall | **73.5%** |
| F1-Score | **0.234** |
| ROC-AUC | **0.626** |
| PR-AUC | **0.175** |

The model correctly identified approximately **74% of actual 30-day readmission encounters**.

However, the relatively low precision indicates that the lower threshold produced a substantial number of false-positive classifications.

This demonstrates the trade-off between increasing sensitivity to the minority class and generating additional false positives.

---

# Baseline vs. Final Configuration

| Metric | Baseline Logistic Regression | Final Calibrated Logistic Regression |
|---|---:|---:|
| Threshold | 0.50 | **0.10** |
| Accuracy | 70.5% | **45.4%** |
| Precision | 20.6% | **13.9%** |
| Recall | 56.1% | **73.5%** |
| F1-Score | 0.301 | **0.234** |
| ROC-AUC | 0.710 | **0.626** |
| PR-AUC | 0.259 | **0.175** |

The final configuration sacrificed overall classification performance on several metrics in exchange for substantially higher recall.

This trade-off was consistent with the project's objective of identifying more potential readmission encounters.

---

# Model Interpretation

Logistic Regression provides interpretable coefficients that can be examined using odds ratios.

The interpretation process examined:

- Model coefficients
- Odds ratios
- Direction of association
- Relative magnitude of associations

An odds ratio greater than 1 indicates higher estimated odds of readmission as the feature increases, holding the other model features constant.

An odds ratio below 1 indicates lower estimated odds.

These relationships represent **model-based associations and should not be interpreted as causal effects**.

The strongest positive associations were related to areas such as:

- Emergency healthcare utilization
- Clinical complexity
- Hospital length of stay

In particular, number of emergency visits, number of diagnoses, and time in hospital were important predictive signals.

---

# Encounter-Level Risk Tracking

After finalizing the model, predictions were generated for the held-out test encounters.

The prediction output included:

- `patient_nbr`
- `encounter_id`
- `Predicted_Readmission_Probability`
- `Predicted_Readmission_Risk`

Patient and encounter identifiers were used for tracking and reporting purposes and were not used as predictive model features.

The final prediction results were saved as:

```text
diabetes_final_prediction_results.xls
```

---

# Risk Tiers

Predicted readmission probabilities were grouped into descriptive risk tiers.

| Predicted Probability | Risk Tier |
|---|---|
| `< 0.10` | Lower Risk |
| `0.10 – < 0.30` | Moderate Risk |
| `0.30 – < 0.50` | High Risk |
| `≥ 0.50` | Very High Risk |

These risk tiers are **descriptive portfolio-level categories and are not clinically validated risk classifications**.

The binary classification threshold and risk tiers serve different purposes:

- **0.10 threshold:** determines the binary model classification.
- **Risk tiers:** provide a more granular descriptive interpretation of predicted probability.

---

# Patient-Level Risk Summary

The project also aggregated encounter-level predictions by `patient_nbr`.

The patient-level summary included:

- Number of encounters
- Highest predicted readmission probability
- Number of high-risk encounters
- Patient-level risk tier

This demonstrates how encounter-level predictions could potentially be summarized for downstream risk-stratification and care-management workflows.

This patient-level analysis is a demonstration of a potential analytics workflow and is not a clinically validated longitudinal risk model.

---

# Deployment

The final model was saved as reusable machine-learning artifacts.

### `final_calibrated_logistic_model.pkl`

Contains the trained calibrated Logistic Regression model.

### `final_model_features.pkl`

Contains the ordered feature structure required by the model.

### `final_threshold.pkl`

Contains the final locked classification threshold:

```text
0.10
```

These artifacts allow the trained model to be loaded and reused without retraining.

---

# Deployment Architecture

The deployment workflow follows:

```text
New Patient Encounter Data
        ↓
Input Validation
        ↓
Feature Transformation
        ↓
Feature Alignment
        ↓
Calibrated Logistic Regression
        ↓
Predicted Readmission Probability
        ↓
0.10 Classification Threshold
        ↓
Readmission Risk Classification
```

The deployment application reproduces the feature structure used during model development and applies the saved classification threshold.

---

# Streamlit Application

The trained model was integrated into an interactive Streamlit application.

The application allows users to enter patient encounter characteristics and receive:

- Estimated probability of 30-day readmission
- Binary readmission-risk classification
- Classification threshold
- Risk interpretation
- Model information
- Clinical-use disclaimer

The application loads the saved model artifacts rather than retraining the model.

### Streamlit Application

https://diabetes-readmission-prediction-x7rqhrfmcyz3r2t88rkhzz.streamlit.app/

---

# Tableau Dashboard

A Tableau Public dashboard was created to complement the machine-learning workflow.

The dashboard provides descriptive and exploratory analysis of clinical and healthcare-utilization patterns associated with readmission.

### Tableau Public Dashboard

https://public.tableau.com/views/Diabetes_Readmission_Risk_Dashboard/DiabetesReadmissionRiskDashboard

The Tableau dashboard provides the **descriptive analytics layer**, while the Streamlit application provides the **predictive modeling layer**.

---

# Clinical and Business Implications

The model demonstrates the potential value of using routinely collected healthcare encounter data to identify encounters with elevated estimated readmission risk.

Because the final configuration prioritizes recall, it could potentially support:

- Risk stratification
- Follow-up prioritization
- Care-management workflows
- Transitional-care resource allocation
- Clinical decision-support prototypes

However, the final model's relatively low precision means that using the predictions as an automatic clinical alert could generate a substantial number of false-positive alerts.

Therefore, the model would be more appropriate as one component of a broader clinical decision-support or care-management workflow rather than as a standalone clinical decision-maker.

---

# Limitations

## Historical Data

The model was developed using historical inpatient diabetes encounters from **1999–2008**.

Changes in clinical practice, treatment patterns, patient populations, healthcare utilization, and coding practices may limit the model's applicability to contemporary healthcare settings.

## Class Imbalance

Only **11.33%** of encounters experienced 30-day readmission.

This makes minority-class prediction challenging and limits the usefulness of accuracy as a standalone metric.

## Low Precision

The final 0.10-threshold configuration achieved:

**13.9% precision**

Although the lower threshold improved recall, it also resulted in a substantial number of false-positive predictions.

## Limited Clinical Variables

The dataset does not capture many potentially important predictors of readmission, including detailed:

- Social determinants of health
- Socioeconomic conditions
- Insurance information
- Outpatient follow-up
- Medication adherence
- Comprehensive disease-severity measures

## No External Validation

The model has not been externally validated using an independent healthcare population.

## No Prospective Clinical Evaluation

The model has not been evaluated prospectively in an operational clinical environment.

## No Clinical Intervention

A validated clinical intervention associated with a high-risk prediction has not been established.

---

# What Would Be Required Before Clinical Use?

Before real-world implementation, the model would require:

1. External validation using an independent healthcare dataset.
2. Evaluation using contemporary healthcare data.
3. Probability calibration assessment.
4. Clinically appropriate threshold selection.
5. Review of false positives and false negatives with clinical stakeholders.
6. Subgroup and fairness evaluation.
7. Prospective clinical evaluation.
8. Clinical workflow integration.
9. Definition of an appropriate intervention associated with a high-risk prediction.
10. Prospective monitoring of model performance.

---

# Technology Stack

## Data & Database

- Microsoft SQL Server
- SQL
- PyODBC
- SQLAlchemy

## Data Science

- Python
- Pandas
- NumPy
- Scikit-learn

## Machine Learning

- Logistic Regression
- Random Forest
- XGBoost
- GridSearchCV
- Probability Calibration
- Classification Threshold Optimization

## Visualization

- Matplotlib
- Seaborn
- Tableau Public

## Deployment

- Streamlit
- Joblib

## Development

- Jupyter Notebook
- VS Code
- GitHub

---

# Repository Structure


diabetes-readmission-risk-prediction/
│
├── .devcontainer/
│   └── devcontainer.json
│
├── images/
│   ├── streamlit_overview_1.png
│   ├── streamlit_overview_2.png
│   ├── streamlit_prediction_3.png
│   ├── streamlit_prediction_4.png
│   ├── tableau_dashboard_1.png
│   └── tableau_dashboard_2.png
│
├── README.md
├── app.py
├── requirements.txt
│
├── diabetes_eeadmission_project.ipynb
│
├── diabetes_clean _dataset.xls
├── diabetes_final_ml_dataset.xls
├── diabetes_final_prediction_results.xls
│
├── clinical reference_diabetes_project.xlsx
├── diabetes_readmission_clinical_stakeholder_presentation.pptx
├── diabetes_readmission_risk_dashboard twbx
│
├── final_calibrated_logistic_model.pkl
├── final_model_features.pkl
└── final_threshold.pkl
```

---

# Key Takeaways

This project demonstrates that healthcare machine learning requires more than selecting an algorithm.

The workflow incorporated:

- Clinical cohort definition
- Healthcare data cleaning
- Data quality assessment
- Leakage assessment
- Exploratory data analysis
- Feature engineering
- Class-imbalance handling
- Baseline model comparison
- Hyperparameter tuning
- Probability calibration
- Classification-threshold optimization
- Model interpretation
- Encounter-level prediction
- Patient-level risk summarization
- Model artifact creation
- Streamlit deployment

One of the most important modeling decisions was recognizing that the classification threshold should reflect the project's objective.

The baseline Logistic Regression model at the 0.50 threshold produced stronger overall performance on several metrics.

However, the calibrated Logistic Regression model using the locked **0.10 threshold** increased recall from **56.1% to 73.5%**.

This came at the cost of lower precision, F1-score, accuracy, ROC-AUC, and PR-AUC.

The final model therefore represents an explicit **precision-recall trade-off driven by the project's objective of identifying more potential readmission encounters**.

---

# Conclusion

This project demonstrates an end-to-end healthcare machine-learning workflow for predicting 30-day hospital readmission among diabetic inpatient encounters following emergency admission.

The workflow included:

- Data understanding
- Data quality assessment
- Data cleaning
- Clinical cohort definition
- Exploratory data analysis
- Feature engineering
- Machine-learning dataset development
- Class-imbalance handling
- Baseline Logistic Regression, Random Forest, and XGBoost modeling
- Hyperparameter tuning
- Baseline versus tuned model comparison
- Probability calibration
- Classification-threshold optimization
- Final model selection
- Model evaluation
- Logistic Regression interpretation
- Encounter-level risk prediction
- Patient-level risk summarization
- Model deployment

The **calibrated Logistic Regression model with the locked 0.10 threshold** was selected as the final classification configuration because its **73.5% recall** aligned with the project's primary objective of identifying as many potential 30-day readmission encounters as possible.

Although the final configuration achieved relatively low precision (**13.9%**), the project demonstrates the practical trade-off between identifying more true readmissions and generating additional false-positive predictions.

Overall, this project demonstrates that responsible healthcare machine learning requires consideration of:

- Clinical objectives
- Cohort definition
- Data quality
- Information leakage
- Class imbalance
- Precision-recall trade-offs
- Probability reliability
- Model interpretability
- External validation
- Clinical workflow integration
- Prospective monitoring

The current system is therefore best viewed as a **portfolio-level healthcare risk-stratification and clinical decision-support prototype**, not a production clinical decision-making system.

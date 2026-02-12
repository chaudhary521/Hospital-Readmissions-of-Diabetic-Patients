Hospital Readmission Prediction for Diabetic Patients
A machine learning project using Logistic Regression to predict the likelihood of hospital readmission among diabetic patients. Built as part of the Statistics and Optimisation module at the National College of Ireland.



Overview
Hospital readmissions are a key performance indicator in healthcare, particularly for diabetic patients where readmission within 30 days can signal inadequate care at discharge. This project builds a binary classification model to predict whether a diabetic patient will be readmitted to hospital, enabling earlier and more targeted clinical interventions.
The target variable readmitted was originally a three-class label (No, Within30Days, After30Days) and was converted into a binary outcome:

1 → Readmitted (within or after 30 days)
0 → Not readmitted


Dataset
The dataset is sourced from Strack et al. (2014) and is available via the UCI Machine Learning Repository.
PropertyValueSourceB. Strack et al., 2014 (UCI Repository)Rows101,763Features47Target Variablereadmitted (binarised)Time Period1999–2008Hospitals130 US hospitals
Key Features Used
FeatureDescriptiontime_in_hospitalNumber of days admittednum_lab_proceduresNumber of lab tests performednum_proceduresNumber of non-lab proceduresnum_medicationsNumber of distinct medications administerednumber_outpatientNumber of outpatient visits in the prior yearnumber_emergencyNumber of emergency visits in the prior yearnumber_inpatientNumber of inpatient visits in the prior yearnumber_diagnosesNumber of diagnoses entered into the system
Patient Demographics

Gender: 54,708 Female / 47,055 Male
Most common age group: 70–80 years (26,066 patients)
Most common ethnicity: Caucasian (76,099 patients)
Readmission rate: ~46% of patients were readmitted


Project Structure
├── hospitaldata.csv           # Raw dataset
├── cleaned_db_data.csv        # Cleaned/preprocessed dataset
├── x23332794_code.pdf         # Full annotated Python code
├── x23332794_Stats_Report.pdf # Written research report
└── README.md

Data Preprocessing

Binary target encoding — readmitted column converted to binary (readmitted_att)
Missing value handling — Null values filled with 0
Non-numeric feature removal — Categorical columns (race, gender, age, medications, etc.) dropped to prepare data for sklearn's Logistic Regression
Feature selection — Columns encounter_id and patient_nbr excluded as they carry no predictive information
Train/test split — 70% training / 30% testing with random_state=42


Model
Algorithm: Logistic Regression (sklearn + statsmodels)
pythonfrom sklearn.linear_model import LogisticRegression

log_reg_model = LogisticRegression(max_iter=1000, random_state=42)
log_reg_model.fit(Independent_cols_train, Dependent_cols_train)
A statsmodels Logit model was also fit on the full dataset to obtain statistically interpretable coefficients, confidence intervals, and p-values.

Results
Model Performance (sklearn)
MetricValueAccuracy61.73%ROC-AUC Score0.65Weighted F1-Score0.60
Confusion Matrix
              Predicted: 0    Predicted: 1
Actual: 0        13,418          2,958
Actual: 1         8,724          5,429
Classification Report
ClassPrecisionRecallF1-Score0 (Not Readmitted)0.610.820.701 (Readmitted)0.650.380.48
Statistically Significant Predictors (statsmodels Logit)
VariableCoefficientp-valueInterpretationnumber_inpatient+0.365< 0.001Strongest positive predictornumber_emergency+0.217< 0.001Strong positive predictornumber_outpatient+0.087< 0.001Positive predictortime_in_hospital+0.014< 0.001Slight positive effectnum_procedures−0.060< 0.001Negative predictornumber_diagnoses−0.028< 0.001Negative predictornum_lab_procedures−0.003< 0.001Slight negative effectnum_medications−0.0010.500Not statistically significant

Note: num_medications was retained in the final model as a clinically relevant variable despite not reaching statistical significance (p > 0.05).


Visualisations
The project includes five key visualisations:

Correlation Heatmap — Numerical feature correlations (strong positive correlation found between surgeries/hospitalisation and lab tests/drugs)
Readmission Bar Chart — ~54% not readmitted vs ~46% readmitted
Logistic Regression Coefficients Plot — With 95% confidence intervals; number_inpatient is the dominant predictor
ROC Curve — AUC = 0.65, indicating moderate discriminative ability
Predicted Probability Distribution — Most patients cluster between 0.4–0.5 predicted probability, with a small high-risk tail near 0.9


Requirements
pandas
numpy
scikit-learn
statsmodels
matplotlib
seaborn
Install all dependencies:
bashpip install pandas numpy scikit-learn statsmodels matplotlib seaborn



The script will output:

Descriptive statistics
Model accuracy, confusion matrix, and classification report
ROC-AUC score
Statsmodels logit summary table
All 5 visualisation plots



References

Strack, B., Ventura, S., Cios, K. J., DeShazo, J. P., Gennings, C., Olmo, J. L., & Clore, J. N. (2014). HbA1c measurement on hospital readmission rates. BioMed Research International, Article ID 781670.



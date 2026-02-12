Predicting Hospital Readmissions of Diabetic Patients
A Logistic Regression Analysis using the Strack et al. (2014) Dataset

Project Overview
This repository contains a full statistical and machine‑learning analysis of diabetic patient readmissions using the UCI Diabetes 130‑US Hospitals Dataset (Strack et al., 2014).
The project explores factors influencing hospital readmission, performs extensive data preprocessing, and builds a logistic regression model to predict whether a patient will be readmitted.

The repository includes:
Raw dataset (hospitaldata.csv)
Cleaned dataset (cleaned_db_data.csv)
Python code for preprocessing, EDA, and model building (x23332794_code.ipynb)

Objective
To build a predictive model that identifies diabetic patients at high risk of hospital readmission, enabling healthcare providers to intervene early and improve patient outcomes.

Repository Structure
Code
├── cleaned_db_data.csv              # Cleaned dataset used for modeling
├── hospitaldata.csv                 # Original dataset (UCI)
├── x23332794_code.pdf               # Python code for analysis & modeling
└── README.md                        # Project documentation

Methodology
1. Data Preprocessing
Converted readmitted into a binary variable (readmitted_att)
Handled missing values using fillna()
Removed non‑numeric columns for logistic regression
Split data into training/testing sets (70/30)
Normalized and cleaned numerical features

2. Exploratory Data Analysis (EDA)
Descriptive statistics for all numeric variables
Frequency counts for categorical variables
Correlation heatmap to identify relationships
Visualizations for readmission distribution

3. Model Development
Logistic Regression using:
scikit-learn for prediction and evaluation
statsmodels for coefficient interpretation
Evaluated using:
Accuracy
Confusion Matrix
Classification Report
ROC Curve & AUC Score

Key Results
Metric	Value
Accuracy	0.617
ROC‑AUC	0.652
Best Predictors	number_inpatient, number_emergency, number_outpatient
Least Significant	num_medications (p > 0.05)

Insights
Patients with more inpatient, emergency, or outpatient visits are significantly more likely to be readmitted.
The dataset is imbalanced, with more non‑readmitted cases.
Model performance is moderate; improvements could include:
SMOTE / oversampling
Feature engineering
Tree‑based models (Random Forest, XGBoost)

Visualizations Included
Correlation Heatmap
Readmission Percentage Bar Plot
Logistic Regression Coefficients
ROC Curve
Predicted Probability Distribution

Technologies Used
Python
Pandas
NumPy
Scikit‑Learn
Statsmodels
Matplotlib
Seaborn

Reference
Strack, B. et al. (2014). Impact of HbA1c Measurement on Hospital Readmission Rates. BioMed Research International.

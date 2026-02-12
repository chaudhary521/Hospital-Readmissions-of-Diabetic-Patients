# Hospital Readmission Prediction for Diabetic Patients

A machine learning project using **Logistic Regression** to predict the likelihood of hospital readmission among diabetic patients. Built as part of the Statistics and Optimisation module at the **National College of Ireland**.


## Overview

Hospital readmissions are a key performance indicator in healthcare, particularly for diabetic patients where readmission within 30 days can signal inadequate care at discharge. This project builds a binary classification model to predict whether a diabetic patient will be readmitted to hospital, enabling earlier and more targeted clinical interventions.

The target variable `readmitted` was originally a three-class label (`No`, `Within30Days`, `After30Days`) and was converted into a binary outcome:
- `1` → Readmitted (within or after 30 days)
- `0` → Not readmitted

---

## Dataset

The dataset is sourced from **Strack et al. (2014)** and is available via the UCI Machine Learning Repository.

| Property | Value |
|---|---|
| Source | B. Strack et al., 2014 (UCI Repository) |
| Rows | 101,763 |
| Features | 47 |
| Target Variable | `readmitted` (binarised) |
| Time Period | 1999–2008 |
| Hospitals | 130 US hospitals |

### Key Features Used

| Feature | Description |
|---|---|
| `time_in_hospital` | Number of days admitted |
| `num_lab_procedures` | Number of lab tests performed |
| `num_procedures` | Number of non-lab procedures |
| `num_medications` | Number of distinct medications administered |
| `number_outpatient` | Number of outpatient visits in the prior year |
| `number_emergency` | Number of emergency visits in the prior year |
| `number_inpatient` | Number of inpatient visits in the prior year |
| `number_diagnoses` | Number of diagnoses entered into the system |

### Patient Demographics

- **Gender:** 54,708 Female / 47,055 Male
- **Most common age group:** 70–80 years (26,066 patients)
- **Most common ethnicity:** Caucasian (76,099 patients)
- **Readmission rate:** ~46% of patients were readmitted

---

## Project Structure

```
├── hospitaldata.csv           # Raw dataset
├── cleaned_db_data.csv        # Cleaned/preprocessed dataset
├── x23332794_code.pdf         # Full annotated Python code
├── x23332794_Stats_Report.pdf # Written research report
└── README.md
```

---

## Data Preprocessing

1. **Binary target encoding** — `readmitted` column converted to binary (`readmitted_att`)
2. **Missing value handling** — Null values filled with `0`
3. **Non-numeric feature removal** — Categorical columns (race, gender, age, medications, etc.) dropped to prepare data for sklearn's Logistic Regression
4. **Feature selection** — Columns `encounter_id` and `patient_nbr` excluded as they carry no predictive information
5. **Train/test split** — 70% training / 30% testing with `random_state=42`

---

## Model

**Algorithm:** Logistic Regression (`sklearn` + `statsmodels`)

```python
from sklearn.linear_model import LogisticRegression

log_reg_model = LogisticRegression(max_iter=1000, random_state=42)
log_reg_model.fit(Independent_cols_train, Dependent_cols_train)
```

A `statsmodels` Logit model was also fit on the full dataset to obtain statistically interpretable coefficients, confidence intervals, and p-values.

---

## Results

### Model Performance (sklearn)

| Metric | Value |
|---|---|
| Accuracy | 61.73% |
| ROC-AUC Score | 0.65 |
| Weighted F1-Score | 0.60 |

### Confusion Matrix

```
              Predicted: 0    Predicted: 1
Actual: 0        13,418          2,958
Actual: 1         8,724          5,429
```

### Classification Report

| Class | Precision | Recall | F1-Score |
|---|---|---|---|
| 0 (Not Readmitted) | 0.61 | 0.82 | 0.70 |
| 1 (Readmitted) | 0.65 | 0.38 | 0.48 |

### Statistically Significant Predictors (statsmodels Logit)

| Variable | Coefficient | p-value | Interpretation |
|---|---|---|---|
| `number_inpatient` | +0.365 | < 0.001 | Strongest positive predictor |
| `number_emergency` | +0.217 | < 0.001 | Strong positive predictor |
| `number_outpatient` | +0.087 | < 0.001 | Positive predictor |
| `time_in_hospital` | +0.014 | < 0.001 | Slight positive effect |
| `num_procedures` | −0.060 | < 0.001 | Negative predictor |
| `number_diagnoses` | −0.028 | < 0.001 | Negative predictor |
| `num_lab_procedures` | −0.003 | < 0.001 | Slight negative effect |
| `num_medications` | −0.001 | 0.500 | Not statistically significant |

> **Note:** `num_medications` was retained in the final model as a clinically relevant variable despite not reaching statistical significance (p > 0.05).

---

## Visualisations

The project includes five key visualisations:

1. **Correlation Heatmap** — Numerical feature correlations (strong positive correlation found between surgeries/hospitalisation and lab tests/drugs)
2. **Readmission Bar Chart** — ~54% not readmitted vs ~46% readmitted
3. **Logistic Regression Coefficients Plot** — With 95% confidence intervals; `number_inpatient` is the dominant predictor
4. **ROC Curve** — AUC = 0.65, indicating moderate discriminative ability
5. **Predicted Probability Distribution** — Most patients cluster between 0.4–0.5 predicted probability, with a small high-risk tail near 0.9

---

## Requirements

```
pandas
numpy
scikit-learn
statsmodels
matplotlib
seaborn
```

Install all dependencies:

```bash
pip install pandas numpy scikit-learn statsmodels matplotlib seaborn
```

---


## References

1. Strack, B., Ventura, S., Cios, K. J., DeShazo, J. P., Gennings, C., Olmo, J. L., & Clore, J. N. (2014). *HbA1c measurement on hospital readmission rates*. BioMed Research International, Article ID 781670.

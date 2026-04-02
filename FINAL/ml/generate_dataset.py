"""
generate_dataset.py
-------------------
Generates a synthetic Loan Prediction dataset that mirrors the real
Kaggle 'Loan Prediction Problem Dataset' (Analytics Vidhya).
Run this ONCE to create loan_data.csv before training.

Real dataset link:
https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 614  # same size as the real Kaggle dataset

gender         = np.random.choice(['Male', 'Female'], N, p=[0.80, 0.20])
married        = np.random.choice(['Yes', 'No'],      N, p=[0.65, 0.35])
dependents     = np.random.choice(['0', '1', '2', '3+'], N, p=[0.57, 0.17, 0.16, 0.10])
education      = np.random.choice(['Graduate', 'Not Graduate'], N, p=[0.78, 0.22])
self_employed  = np.random.choice(['Yes', 'No'], N, p=[0.14, 0.86])
applicant_inc  = np.random.lognormal(mean=8.5, sigma=0.6, size=N).astype(int)
coapplicant_inc= np.random.choice(
    [0] * 300 + list(np.random.lognormal(7.5, 0.7, N - 300).astype(int)), N)
loan_amount    = np.random.lognormal(mean=4.9, sigma=0.5, size=N).astype(int)
loan_term      = np.random.choice([360, 180, 480, 300, 240, 84, 120, 60, 36], N,
                                   p=[0.68, 0.10, 0.07, 0.05, 0.04, 0.02, 0.02, 0.01, 0.01])
credit_history = np.random.choice([1.0, 0.0], N, p=[0.84, 0.16])
property_area  = np.random.choice(['Urban', 'Semiurban', 'Rural'], N, p=[0.38, 0.37, 0.25])

# Loan status: influenced by credit history, income, education
prob_approved = (
    0.35
    + 0.40 * credit_history
    + 0.05 * (applicant_inc > 4000)
    + 0.05 * (education == 'Graduate')
    + 0.05 * (married == 'Yes')
    - 0.05 * (loan_amount > 150)
)
prob_approved = np.clip(prob_approved, 0.05, 0.95)
loan_status = np.where(np.random.rand(N) < prob_approved, 'Y', 'N')

# Introduce ~8% missing values (realistic)
def add_missing(arr, pct=0.08):
    arr = arr.astype(object)
    idx = np.random.choice(len(arr), int(len(arr) * pct), replace=False)
    arr[idx] = np.nan
    return arr

df = pd.DataFrame({
    'Loan_ID':           [f'LP{str(i).zfill(6)}' for i in range(1, N + 1)],
    'Gender':            add_missing(gender),
    'Married':           add_missing(married),
    'Dependents':        add_missing(dependents),
    'Education':         education,
    'Self_Employed':     add_missing(self_employed),
    'ApplicantIncome':   applicant_inc,
    'CoapplicantIncome': coapplicant_inc,
    'LoanAmount':        add_missing(loan_amount.astype(object)),
    'Loan_Amount_Term':  add_missing(loan_term.astype(object)),
    'Credit_History':    add_missing(credit_history.astype(object)),
    'Property_Area':     property_area,
    'Loan_Status':       loan_status,
})

df.to_csv('ml/loan_data.csv', index=False)
print(f"Dataset saved → ml/loan_data.csv  ({N} rows)")
print(df['Loan_Status'].value_counts())

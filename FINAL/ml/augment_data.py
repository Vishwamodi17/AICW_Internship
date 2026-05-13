"""
augment_data.py
---------------
Generates 1000 additional realistic synthetic records that mirror
the real Kaggle Loan Prediction dataset distribution.
Appends to ml/loan_data.csv and saves the combined dataset.

Run: python ml/augment_data.py
"""

import os
import numpy as np
import pandas as pd

np.random.seed(2024)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CSV_PATH = os.path.join(BASE_DIR, 'loan_data.csv')

# ── Load existing data ─────────────────────────────────────────
df_orig = pd.read_csv(CSV_PATH)
print(f"Original dataset: {df_orig.shape}")
print(df_orig['Loan_Status'].value_counts())

N = 1000  # new records to generate

# ── Generate features matching real distribution ───────────────
gender         = np.random.choice(['Male','Female'], N, p=[0.80, 0.20])
married        = np.random.choice(['Yes','No'],      N, p=[0.65, 0.35])
dependents     = np.random.choice(['0','1','2','3+'],N, p=[0.57, 0.17, 0.16, 0.10])
education      = np.random.choice(['Graduate','Not Graduate'], N, p=[0.78, 0.22])
self_employed  = np.random.choice(['Yes','No'], N, p=[0.14, 0.86])

# Income: log-normal to match real skewed distribution
applicant_inc  = np.random.lognormal(mean=8.5, sigma=0.65, size=N).astype(int)
# Co-applicant: ~40% have co-applicant income
coapplicant_inc= np.where(
    np.random.rand(N) < 0.40,
    np.random.lognormal(7.5, 0.7, N).astype(int),
    0
).astype(float)

# Loan amount: log-normal (in thousands, matching original)
loan_amount    = np.random.lognormal(mean=4.9, sigma=0.55, size=N)
loan_amount    = np.clip(loan_amount, 9, 700).round(0)

# Loan term: realistic distribution
loan_term      = np.random.choice(
    [360, 180, 480, 300, 240, 84, 120, 60, 36],
    N, p=[0.68, 0.10, 0.07, 0.05, 0.04, 0.02, 0.02, 0.01, 0.01]
).astype(float)

# Credit history: 84% good
credit_history = np.random.choice([1.0, 0.0], N, p=[0.84, 0.16])

property_area  = np.random.choice(['Urban','Semiurban','Rural'], N, p=[0.38, 0.37, 0.25])

# ── Loan status: realistic probability model ───────────────────
# Factors: credit history (strongest), income, education, EMI ratio
emi_proxy      = (loan_amount / loan_term) * 1000
income_total   = applicant_inc + coapplicant_inc
emi_ratio      = np.where(income_total > 0, emi_proxy / income_total, 1.0)

prob_approved = (
    0.30
    + 0.42 * credit_history                          # credit history dominant
    + 0.06 * (income_total > 5000).astype(float)
    + 0.05 * (education == 'Graduate').astype(float)
    + 0.04 * (married == 'Yes').astype(float)
    + 0.04 * (property_area != 'Rural').astype(float)
    - 0.08 * (emi_ratio > 0.5).astype(float)         # high EMI ratio hurts
    - 0.05 * (loan_amount > 300).astype(float)
    - 0.04 * (self_employed == 'Yes').astype(float)
)
prob_approved  = np.clip(prob_approved, 0.05, 0.95)
loan_status    = np.where(np.random.rand(N) < prob_approved, 'Y', 'N')

# ── Introduce realistic missing values (~5%) ───────────────────
def add_missing(arr, pct=0.05):
    arr = arr.astype(object)
    idx = np.random.choice(len(arr), int(len(arr) * pct), replace=False)
    arr[idx] = np.nan
    return arr

# Start Loan_ID from max existing
max_id = int(df_orig['Loan_ID'].str.replace('LP','').astype(int).max())

df_new = pd.DataFrame({
    'Loan_ID':           [f'LP{str(max_id + i + 1).zfill(6)}' for i in range(N)],
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

# ── Combine and save ───────────────────────────────────────────
df_combined = pd.concat([df_orig, df_new], ignore_index=True)
df_combined.to_csv(CSV_PATH, index=False)

print(f"\nNew records added : {N}")
print(f"Combined dataset  : {df_combined.shape}")
print(f"\nLoan Status distribution:")
print(df_combined['Loan_Status'].value_counts())
print(f"\nApproval rate: {df_combined['Loan_Status'].eq('Y').mean()*100:.1f}%")
print(f"\nSaved → {CSV_PATH}")

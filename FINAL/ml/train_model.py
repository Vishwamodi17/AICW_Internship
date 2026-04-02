"""
train_model.py
--------------
Trains a Gradient Boosting Classifier on the REAL Kaggle Loan Prediction dataset.
Dataset: https://www.kaggle.com/datasets/altruistdelhite04/loan-prediction-problem-dataset

Steps:
  1. Load & explore data
  2. Clean / impute missing values
  3. Feature engineering
  4. Encode categorical features
  5. Train/test split (stratified)
  6. Hyperparameter tuning (RandomizedSearchCV, 5-fold CV)
  7. Evaluate (accuracy, confusion matrix, classification report)
  8. Feature importance plot
  9. Save model pipeline as loan_model.pkl
"""

import os
import pickle
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.ensemble        import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, RandomizedSearchCV, StratifiedKFold
from sklearn.preprocessing   import LabelEncoder
from sklearn.metrics         import (accuracy_score, confusion_matrix,
                                     classification_report, ConfusionMatrixDisplay)

warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ─────────────────────────────────────────────
# 1. LOAD DATA
# ─────────────────────────────────────────────
df = pd.read_csv(os.path.join(BASE_DIR, 'loan_data.csv'))

print("=" * 60)
print("REAL KAGGLE DATASET")
print("=" * 60)
print(f"Shape  : {df.shape}")
print(f"\nMissing values:\n{df.isnull().sum()}")
print(f"\nTarget distribution:\n{df['Loan_Status'].value_counts()}")

# Drop Loan_ID — not a feature
df.drop(columns=['Loan_ID'], inplace=True)

# ─────────────────────────────────────────────
# 2. CLEAN / IMPUTE MISSING VALUES
# ─────────────────────────────────────────────
# Categorical → mode
for col in ['Gender', 'Married', 'Self_Employed']:
    df[col] = df[col].fillna(df[col].mode()[0])

# Dependents: fix '3+' → '3', then mode-fill
df['Dependents'] = df['Dependents'].replace('3+', '3')
df['Dependents'] = df['Dependents'].fillna(df['Dependents'].mode()[0])

# Numeric → mode / median
df['Credit_History']   = df['Credit_History'].fillna(df['Credit_History'].mode()[0])
df['Loan_Amount_Term'] = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0])
df['LoanAmount']       = df['LoanAmount'].fillna(df['LoanAmount'].median())

print(f"\nMissing after imputation : {df.isnull().sum().sum()}")

# ─────────────────────────────────────────────
# 3. FEATURE ENGINEERING
# ─────────────────────────────────────────────
df['TotalIncome']     = df['ApplicantIncome'] + df['CoapplicantIncome']
df['TotalIncome_log'] = np.log1p(df['TotalIncome'])
df['LoanAmount_log']  = np.log1p(df['LoanAmount'])
df['EMI']             = df['LoanAmount'] / df['Loan_Amount_Term'].astype(float)
df['Balance_Income']  = df['TotalIncome'] - (df['EMI'] * 1000)

# ─────────────────────────────────────────────
# 4. ENCODE CATEGORICAL FEATURES
# ─────────────────────────────────────────────
le = LabelEncoder()
for col in ['Gender', 'Married', 'Education', 'Self_Employed', 'Property_Area']:
    df[col] = le.fit_transform(df[col].astype(str))

df['Dependents']  = df['Dependents'].astype(float).astype(int)
df['Loan_Status'] = df['Loan_Status'].map({'Y': 1, 'N': 0})

# ─────────────────────────────────────────────
# 5. FEATURES & TARGET
# ─────────────────────────────────────────────
FEATURES = [
    'Gender', 'Married', 'Dependents', 'Education', 'Self_Employed',
    'ApplicantIncome', 'CoapplicantIncome', 'LoanAmount',
    'Loan_Amount_Term', 'Credit_History', 'Property_Area',
    'TotalIncome_log', 'LoanAmount_log', 'EMI', 'Balance_Income',
]

X = df[FEATURES]
y = df['Loan_Status']

print(f"\nFeatures used : {len(FEATURES)}")
print(f"Class balance : {y.value_counts().to_dict()}")

# ─────────────────────────────────────────────
# 6. TRAIN / TEST SPLIT
# ─────────────────────────────────────────────
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42, stratify=y
)
print(f"\nTrain : {X_train.shape[0]}  |  Test : {X_test.shape[0]}")

# ─────────────────────────────────────────────
# 7. HYPERPARAMETER TUNING
# ─────────────────────────────────────────────
param_dist = {
    'n_estimators':      [100, 200, 300, 400, 500],
    'learning_rate':     [0.01, 0.05, 0.08, 0.1, 0.15, 0.2],
    'max_depth':         [3, 4, 5],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf':  [1, 2, 4],
    'subsample':         [0.7, 0.8, 0.9, 1.0],
    'max_features':      ['sqrt', 'log2', None],
}

gbc = GradientBoostingClassifier(random_state=42)
cv  = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

search = RandomizedSearchCV(
    gbc, param_dist,
    n_iter=80,
    scoring='accuracy',
    cv=cv,
    n_jobs=-1,
    random_state=42,
    verbose=1,
)

print("\nRunning RandomizedSearchCV (80 iterations × 5-fold CV)…")
search.fit(X_train, y_train)

best_model = search.best_estimator_
print(f"\nBest params  : {search.best_params_}")
print(f"Best CV acc  : {search.best_score_:.4f}  ({search.best_score_*100:.2f}%)")

# ─────────────────────────────────────────────
# 8. EVALUATE ON TEST SET
# ─────────────────────────────────────────────
y_pred = best_model.predict(X_test)
acc    = accuracy_score(y_test, y_pred)

print("\n" + "=" * 60)
print("MODEL EVALUATION ON TEST SET")
print("=" * 60)
print(f"Test Accuracy : {acc:.4f}  ({acc*100:.2f}%)")
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=['Rejected (N)', 'Approved (Y)']))

# Confusion matrix
cm  = confusion_matrix(y_test, y_pred)
fig, ax = plt.subplots(figsize=(6, 5))
ConfusionMatrixDisplay(cm, display_labels=['Rejected', 'Approved']).plot(
    ax=ax, colorbar=False, cmap='Blues')
ax.set_title('Confusion Matrix – Gradient Boosting', fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'confusion_matrix.png'), dpi=150)
print("\nConfusion matrix → ml/confusion_matrix.png")

# ─────────────────────────────────────────────
# 9. FEATURE IMPORTANCE PLOT
# ─────────────────────────────────────────────
feat_df = (pd.DataFrame({'Feature': FEATURES,
                          'Importance': best_model.feature_importances_})
             .sort_values('Importance', ascending=True))

fig, ax = plt.subplots(figsize=(9, 6))
colors  = sns.color_palette('viridis', len(feat_df))
ax.barh(feat_df['Feature'], feat_df['Importance'], color=colors)
ax.set_xlabel('Importance Score', fontsize=12)
ax.set_title('Feature Importance – Gradient Boosting Classifier',
             fontsize=13, fontweight='bold')
ax.grid(axis='x', linestyle='--', alpha=0.5)
for i, (val, _) in enumerate(zip(feat_df['Importance'], feat_df['Feature'])):
    ax.text(val + 0.001, i, f'{val:.4f}', va='center', fontsize=8)
plt.tight_layout()
plt.savefig(os.path.join(BASE_DIR, 'feature_importance.png'), dpi=150)
print("Feature importance  → ml/feature_importance.png")

# ─────────────────────────────────────────────
# 10. SAVE MODEL
# ─────────────────────────────────────────────
model_path = os.path.join(BASE_DIR, 'loan_model.pkl')
with open(model_path, 'wb') as f:
    pickle.dump({'model': best_model, 'features': FEATURES}, f)

print(f"\nModel saved → {model_path}")
print("=" * 60)
print("Training complete.")

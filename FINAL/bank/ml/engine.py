"""
bank/ml/engine.py
-----------------
ML Engine: trains Logistic Regression, Decision Tree, Random Forest.
Selects best model. Saves to bank/ml/best_model.pkl.
Also exposes predict(), explain(), emi_calc() for views.
"""

import os, pickle, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import shap


from sklearn.linear_model     import LogisticRegression
from sklearn.tree             import DecisionTreeClassifier
from sklearn.ensemble         import RandomForestClassifier
from sklearn.model_selection  import train_test_split, cross_val_score
from sklearn.preprocessing    import LabelEncoder, StandardScaler
from sklearn.metrics          import (accuracy_score, confusion_matrix,
                                      classification_report)
from sklearn.pipeline         import Pipeline

warnings.filterwarnings('ignore')

BASE   = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH  = os.path.join(BASE, 'best_model.pkl')
SCALER_PATH = os.path.join(BASE, 'scaler.pkl')
DATA_PATH   = os.path.join(BASE, '..', '..', 'ml', 'loan_data.csv')

FEATURES = [
    'Gender','Married','Dependents','Education','Self_Employed',
    'ApplicantIncome','CoapplicantIncome','LoanAmount',
    'Loan_Amount_Term','Credit_History','Property_Area',
    'TotalIncome_log','LoanAmount_log','EMI','Balance_Income',
]

# Encoding maps (must match training)
GENDER_MAP   = {'Male':1,'Female':0}
MARRIED_MAP  = {'Yes':1,'No':0}
EDU_MAP      = {'Graduate':0,'Not Graduate':1}
EMP_MAP      = {'Yes':1,'No':0}
PROP_MAP     = {'Rural':0,'Semiurban':1,'Urban':2}

# ── Loan type rules ────────────────────────────────────────────
LOAN_TYPES = {
    'home': {
        'label':    'Home Loan',
        'rate':     8.5,   # % per annum
        'max_term': 360,
        'min_income': 30000,
        'description': 'For purchase or construction of residential property.',
    },
    'personal': {
        'label':    'Personal Loan',
        'rate':     13.5,
        'max_term': 60,
        'min_income': 15000,
        'description': 'Unsecured loan for personal expenses.',
    },
    'car': {
        'label':    'Car Loan',
        'rate':     9.5,
        'max_term': 84,
        'min_income': 20000,
        'description': 'For purchase of new or used vehicles.',
    },
}


# ══════════════════════════════════════════════════════════════
# TRAINING
# ══════════════════════════════════════════════════════════════
def train():
    df = pd.read_csv(DATA_PATH)
    df.drop(columns=['Loan_ID'], inplace=True, errors='ignore')

    # Impute
    for col in ['Gender','Married','Self_Employed']:
        df[col] = df[col].fillna(df[col].mode()[0])
    df['Dependents']        = df['Dependents'].replace('3+','3').fillna('0')
    df['Credit_History']    = df['Credit_History'].fillna(df['Credit_History'].mode()[0])
    df['Loan_Amount_Term']  = df['Loan_Amount_Term'].fillna(df['Loan_Amount_Term'].mode()[0])
    df['LoanAmount']        = df['LoanAmount'].fillna(df['LoanAmount'].median())

    # Engineer
    df['TotalIncome']     = df['ApplicantIncome'] + df['CoapplicantIncome']
    df['TotalIncome_log'] = np.log1p(df['TotalIncome'])
    df['LoanAmount_log']  = np.log1p(df['LoanAmount'])
    df['EMI']             = df['LoanAmount'] / df['Loan_Amount_Term'].astype(float)
    df['Balance_Income']  = df['TotalIncome'] - (df['EMI'] * 1000)

    # Encode
    le = LabelEncoder()
    for col in ['Gender','Married','Education','Self_Employed','Property_Area']:
        df[col] = le.fit_transform(df[col].astype(str))
    df['Dependents']  = df['Dependents'].astype(float).astype(int)
    df['Loan_Status'] = df['Loan_Status'].map({'Y':1,'N':0})

    X = df[FEATURES]
    y = df['Loan_Status']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y)

    # Scale
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s  = scaler.transform(X_test)

    # Models
    models = {
        'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
        'Decision Tree':       DecisionTreeClassifier(max_depth=5, random_state=42),
        'Random Forest':       RandomForestClassifier(n_estimators=200, random_state=42),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train_s, y_train)
        y_pred = model.predict(X_test_s)
        acc    = accuracy_score(y_test, y_pred)
        cv     = cross_val_score(model, X_train_s, y_train, cv=5).mean()
        cm     = confusion_matrix(y_test, y_pred)
        report = classification_report(y_test, y_pred,
                                       target_names=['Rejected','Approved'],
                                       output_dict=True)
        results[name] = {
            'model':    model,
            'accuracy': round(acc*100, 2),
            'cv_score': round(cv*100, 2),
            'cm':       cm.tolist(),
            'report':   report,
        }
        print(f"{name:25} Acc={acc*100:.2f}%  CV={cv*100:.2f}%")

    # Best model by accuracy
    best_name = max(results, key=lambda k: results[k]['accuracy'])
    best      = results[best_name]

    # Feature importance (RF only)
    rf_model = results['Random Forest']['model']
    importances = dict(zip(FEATURES, rf_model.feature_importances_))

    # Save
    payload = {
        'best_name':    best_name,
        'best_model':   best['model'],
        'scaler':       scaler,
        'features':     FEATURES,
        'importances':  importances,
        'all_results':  {k: {kk:vv for kk,vv in v.items() if kk!='model'}
                         for k,v in results.items()},
        'X_test':       X_test,
        'y_test':       y_test.tolist(),
        'df_sample':    df.head(200).to_dict(orient='records'),
    }
    with open(MODEL_PATH, 'wb') as f:
        pickle.dump(payload, f)

    print(f"\nBest model: {best_name} ({best['accuracy']}%)")
    print(f"Saved → {MODEL_PATH}")
    return payload


# ══════════════════════════════════════════════════════════════
# LOAD
# ══════════════════════════════════════════════════════════════
_cache = None

def load():
    with open(MODEL_PATH, 'rb') as f:
        pkg = pickle.load(f)
    
    best_model = pkg['best_model']  # ✅ now best_model exists
    
    # Add SHAP here, AFTER best_model is loaded
    explainer = shap.TreeExplainer(best_model)
    pkg['explainer'] = explainer   # store it in pkg so views can use it
    
    return pkg


# ══════════════════════════════════════════════════════════════
# PREDICT
# ══════════════════════════════════════════════════════════════
def predict(data: dict, loan_type: str = 'home') -> dict:
    """
    data: cleaned form dict
    Returns: result, approved_prob, rejected_prob, reasons, emi_info
    """
    pkg = load()
    model  = pkg['best_model']
    scaler = pkg['scaler']

    # Build feature vector
    applicant_inc   = float(data['applicant_income'])
    coapplicant_inc = float(data.get('coapplicant_income', 0))
    loan_amount     = float(data['loan_amount'])
    loan_term       = float(data['loan_amount_term'])
    credit_history  = float(data['credit_history'])

    total_income     = applicant_inc + coapplicant_inc
    total_income_log = np.log1p(total_income)
    loan_amount_log  = np.log1p(loan_amount)
    emi_val          = loan_amount / loan_term if loan_term > 0 else 0
    balance_income   = total_income - (emi_val * 1000)

    row = [
        GENDER_MAP.get(data['gender'], 1),
        MARRIED_MAP.get(data['married'], 0),
        int(str(data['dependents']).replace('3+','3')),
        EDU_MAP.get(data['education'], 0),
        EMP_MAP.get(data['self_employed'], 0),
        applicant_inc, coapplicant_inc, loan_amount,
        loan_term, credit_history,
        PROP_MAP.get(data['property_area'], 2),
        total_income_log, loan_amount_log, emi_val, balance_income,
    ]

    X = pd.DataFrame([row], columns=FEATURES)
    X_s = scaler.transform(X)

    pred     = model.predict(X_s)[0]
    proba    = model.predict_proba(X_s)[0]
    app_prob = round(float(proba[1]) * 100, 2)
    rej_prob = round(float(proba[0]) * 100, 2)

    # Loan type rule check
    lt       = LOAN_TYPES.get(loan_type, LOAN_TYPES['home'])
    rule_msg = None
    if total_income < lt['min_income']:
        pred     = 0
        app_prob = min(app_prob, 20.0)
        rej_prob = 100 - app_prob
        rule_msg = (f"Income ₹{total_income:,.0f} is below minimum "
                    f"₹{lt['min_income']:,} required for {lt['label']}.")

    result = 'Approved' if pred == 1 else 'Rejected'

    # EMI calculation (actual monthly EMI)
    rate_monthly = lt['rate'] / 12 / 100
    n            = int(loan_term)
    if rate_monthly > 0 and n > 0:
        emi_monthly = (loan_amount * 1000 * rate_monthly *
                       (1 + rate_monthly)**n) / ((1 + rate_monthly)**n - 1)
    else:
        emi_monthly = (loan_amount * 1000) / n if n > 0 else 0

    emi_info = {
        'monthly_emi':   round(emi_monthly, 2),
        'total_payment': round(emi_monthly * n, 2),
        'total_interest':round(emi_monthly * n - loan_amount * 1000, 2),
        'rate':          lt['rate'],
        'loan_label':    lt['label'],
    }

    reasons = _build_reasons(data, result, app_prob,
                             total_income, emi_val, loan_amount,
                             loan_term, pkg['importances'])
    if rule_msg:
        reasons.insert(0, {
            'label':  'Loan Type Rule',
            'detail': rule_msg,
            'status': 'negative',
            'impact': 'High',
            'icon':   'bi-exclamation-triangle-fill',
        })

    return {
        'result':       result,
        'approved_prob':app_prob,
        'rejected_prob':rej_prob,
        'reasons':      reasons,
        'emi_info':     emi_info,
        'loan_type':    lt,
        'model_used':   pkg['best_name'],
    }


# ══════════════════════════════════════════════════════════════
# REASON ANALYSIS
# ══════════════════════════════════════════════════════════════
def _impact(feat, importances):
    imp = importances.get(feat, 0)
    if imp >= 0.10: return 'High'
    if imp >= 0.04: return 'Medium'
    return 'Low'

def _build_reasons(data, result, app_prob,
                   total_income, emi_val, loan_amount,
                   loan_term, importances):
    credit_history = data['credit_history']
    education      = data['education']
    married        = data['married']
    property_area  = data['property_area']
    income_ratio   = (emi_val * 1000 / total_income * 100) if total_income > 0 else 100

    reasons = []

    # 1. Credit History
    if credit_history == '1':
        reasons.append({'label':'Credit History','status':'positive','impact':_impact('Credit_History',importances),
            'detail':'Good credit history — strongest positive signal for lenders.','icon':'bi-shield-fill-check'})
    else:
        reasons.append({'label':'Credit History','status':'negative','impact':_impact('Credit_History',importances),
            'detail':'Poor/no credit history — the #1 reason for rejection.','icon':'bi-shield-fill-x'})

    # 2. EMI vs Income
    if income_ratio <= 30:
        reasons.append({'label':'EMI vs Income','status':'positive','impact':_impact('Balance_Income',importances),
            'detail':f'EMI is {income_ratio:.1f}% of income — well within safe 40% limit.','icon':'bi-graph-up-arrow'})
    elif income_ratio <= 50:
        reasons.append({'label':'EMI vs Income','status':'neutral','impact':_impact('Balance_Income',importances),
            'detail':f'EMI is {income_ratio:.1f}% of income — slightly high, lenders prefer below 40%.','icon':'bi-graph-up'})
    else:
        reasons.append({'label':'EMI vs Income','status':'negative','impact':_impact('Balance_Income',importances),
            'detail':f'EMI is {income_ratio:.1f}% of income — exceeds safe threshold, high repayment risk.','icon':'bi-graph-down-arrow'})

    # 3. Total Income
    if total_income >= 8000:
        reasons.append({'label':'Total Income','status':'positive','impact':_impact('TotalIncome_log',importances),
            'detail':f'Combined income ₹{total_income:,.0f} is strong.','icon':'bi-currency-rupee'})
    elif total_income >= 4000:
        reasons.append({'label':'Total Income','status':'neutral','impact':_impact('TotalIncome_log',importances),
            'detail':f'Combined income ₹{total_income:,.0f} is moderate.','icon':'bi-currency-rupee'})
    else:
        reasons.append({'label':'Total Income','status':'negative','impact':_impact('TotalIncome_log',importances),
            'detail':f'Combined income ₹{total_income:,.0f} is low for requested loan.','icon':'bi-currency-rupee'})

    # 4. Loan Amount
    if loan_amount <= 100:
        reasons.append({'label':'Loan Amount','status':'positive','impact':_impact('LoanAmount_log',importances),
            'detail':f'₹{loan_amount:.0f}K is modest and manageable.','icon':'bi-bank'})
    elif loan_amount <= 200:
        reasons.append({'label':'Loan Amount','status':'neutral','impact':_impact('LoanAmount_log',importances),
            'detail':f'₹{loan_amount:.0f}K is within acceptable range.','icon':'bi-bank'})
    else:
        reasons.append({'label':'Loan Amount','status':'negative','impact':_impact('LoanAmount_log',importances),
            'detail':f'₹{loan_amount:.0f}K is high — consider reducing it.','icon':'bi-bank'})

    # 5. Property Area
    area_s = {'Urban':'positive','Semiurban':'positive','Rural':'neutral'}
    area_d = {'Urban':'Urban properties have high resale value, reducing lender risk.',
               'Semiurban':'Semiurban properties are viewed favourably.',
               'Rural':'Rural properties carry slightly higher risk.'}
    reasons.append({'label':'Property Area','status':area_s.get(property_area,'neutral'),
        'impact':_impact('Property_Area',importances),'detail':area_d.get(property_area,''),'icon':'bi-house-fill'})

    # 6. Education
    if education == 'Graduate':
        reasons.append({'label':'Education','status':'positive','impact':_impact('Education',importances),
            'detail':'Graduate status associated with higher earning potential.','icon':'bi-mortarboard-fill'})
    else:
        reasons.append({'label':'Education','status':'neutral','impact':_impact('Education',importances),
            'detail':'Non-graduate applicants carry slightly higher statistical risk.','icon':'bi-mortarboard'})

    # Sort
    order = {'negative':0,'neutral':1,'positive':2}
    if result == 'Rejected':
        reasons.sort(key=lambda r:(order[r['status']], r['impact']!='High'))
    else:
        reasons.sort(key=lambda r:(-order[r['status']], r['impact']!='High'))

    return reasons


# ══════════════════════════════════════════════════════════════
# EMI CALCULATOR (standalone)
# ══════════════════════════════════════════════════════════════
def emi_calc(principal_inr: float, annual_rate: float, months: int) -> dict:
    r = annual_rate / 12 / 100
    if r > 0 and months > 0:
        emi = (principal_inr * r * (1+r)**months) / ((1+r)**months - 1)
    else:
        emi = principal_inr / months if months > 0 else 0
    total   = emi * months
    interest= total - principal_inr
    return {
        'emi':      round(emi, 2),
        'total':    round(total, 2),
        'interest': round(interest, 2),
        'principal':round(principal_inr, 2),
    }


# ══════════════════════════════════════════════════════════════
# WHAT-IF ANALYSIS
# ══════════════════════════════════════════════════════════════
def whatif(base_data: dict, loan_type: str) -> list:
    """
    Returns list of scenarios varying income and loan amount.
    """
    scenarios = []
    base_income = float(base_data['applicant_income'])
    base_loan   = float(base_data['loan_amount'])

    for income_mult in [1.0, 1.25, 1.5, 2.0]:
        for loan_mult in [1.0, 0.75, 0.5]:
            d = dict(base_data)
            d['applicant_income'] = base_income * income_mult
            d['loan_amount']      = base_loan   * loan_mult
            r = predict(d, loan_type)
            scenarios.append({
                'income':       round(d['applicant_income']),
                'loan':         round(d['loan_amount']),
                'result':       r['result'],
                'approved_prob':r['approved_prob'],
            })
    return scenarios


if __name__ == '__main__':
    train()

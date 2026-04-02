"""
views.py
--------
Handles: landing, register, login, logout, apply (loan form), predict result.
Model is loaded once at module level for performance.
"""

import pickle
import numpy as np
import pandas as pd
from datetime import datetime

from django.shortcuts        import render, redirect
from django.contrib.auth     import authenticate, login, logout
from django.contrib.auth.forms import UserCreationForm
from django.contrib.auth.decorators import login_required
from django.contrib          import messages
from django.conf             import settings

from .forms import LoanApplicationForm

# ── Load ML model once at startup ─────────────────────────────
with open(settings.ML_MODEL_PATH, 'rb') as f:
    _model_data = pickle.load(f)

MODEL    = _model_data['model']
FEATURES = _model_data['features']

# Encoding maps — must match train_model.py
GENDER_MAP        = {'Male': 1, 'Female': 0}
MARRIED_MAP       = {'Yes': 1, 'No': 0}
EDUCATION_MAP     = {'Graduate': 0, 'Not Graduate': 1}
SELF_EMPLOYED_MAP = {'Yes': 1, 'No': 0}
PROPERTY_MAP      = {'Rural': 0, 'Semiurban': 1, 'Urban': 2}

HOW_IT_WORKS = [
    {'title': 'Create a Free Account',  'desc': 'Register in seconds — no credit card needed.'},
    {'title': 'Fill Your Details',      'desc': 'Enter your income, loan amount, credit history and more.'},
    {'title': 'Get Instant Prediction', 'desc': 'Our AI model returns your approval probability immediately.'},
]

# Feature importance from the trained model (used for reason ranking)
FEATURE_IMPORTANCES = dict(zip(FEATURES, MODEL.feature_importances_))


# ── Landing page ───────────────────────────────────────────────
def landing(request):
    return render(request, 'loan_app/landing.html', {'steps': HOW_IT_WORKS})


# ── Register ───────────────────────────────────────────────────
def register_view(request):
    if request.user.is_authenticated:
        return redirect('landing')
    form = UserCreationForm(request.POST or None)
    if request.method == 'POST':
        if form.is_valid():
            user = form.save()
            login(request, user)
            messages.success(request, 'Account created! Welcome aboard.')
            return redirect('apply')
    return render(request, 'loan_app/register.html', {'form': form})


# ── Login ──────────────────────────────────────────────────────
def login_view(request):
    if request.user.is_authenticated:
        return redirect('landing')
    error = None
    if request.method == 'POST':
        username = request.POST.get('username', '').strip()
        password = request.POST.get('password', '')
        user = authenticate(request, username=username, password=password)
        if user:
            login(request, user)
            next_url = request.GET.get('next', 'apply')
            return redirect(next_url)
        error = 'Invalid username or password. Please try again.'
    return render(request, 'loan_app/login.html', {'error': error})


# ── Logout ─────────────────────────────────────────────────────
def logout_view(request):
    logout(request)
    return redirect('landing')


# ── Apply (loan form) — login required ────────────────────────
@login_required(login_url='login')
def apply(request):
    form = LoanApplicationForm()
    return render(request, 'loan_app/index.html', {'form': form})


# ── Predict (form POST handler) — login required ──────────────
@login_required(login_url='login')
def predict(request):
    if request.method != 'POST':
        return redirect('apply')

    form = LoanApplicationForm(request.POST)
    if not form.is_valid():
        return render(request, 'loan_app/index.html', {'form': form})

    data          = form.cleaned_data
    features      = _preprocess(data)
    prediction    = MODEL.predict(features)[0]
    probability   = MODEL.predict_proba(features)[0]
    approved_prob = round(float(probability[1]) * 100, 2)
    rejected_prob = round(float(probability[0]) * 100, 2)
    result        = 'Approved' if prediction == 1 else 'Rejected'

    context = {
        'result':         result,
        'approved_prob':  approved_prob,
        'rejected_prob':  rejected_prob,
        'form_data':      data,
        'applicant_name': request.user.get_full_name() or request.user.username,
        'receipt_date':   datetime.now().strftime('%d %B %Y, %I:%M %p'),
        'receipt_id':     f'LN{request.user.id:04d}{datetime.now().strftime("%d%m%y%H%M")}',
        'reasons':        _build_reasons(data, result, approved_prob),
    }
    return render(request, 'loan_app/result.html', context)


# ── Reason Analysis ────────────────────────────────────────────
def _build_reasons(data, result, approved_prob):
    """
    Returns a list of reason dicts, each with:
      label   – human-readable factor name
      detail  – specific sentence about the applicant's value
      status  – 'positive' | 'negative' | 'neutral'
      impact  – 'High' | 'Medium' | 'Low'  (from model feature importance)
      icon    – Bootstrap icon class
    """
    applicant_inc   = float(data['applicant_income'])
    coapplicant_inc = float(data['coapplicant_income'])
    loan_amount     = float(data['loan_amount'])
    loan_term       = float(data['loan_amount_term'])
    credit_history  = data['credit_history']          # '1' or '0'
    education       = data['education']
    married         = data['married']
    self_employed   = data['self_employed']
    property_area   = data['property_area']
    dependents      = str(data['dependents'])

    total_income  = applicant_inc + coapplicant_inc
    emi           = (loan_amount / loan_term * 1000) if loan_term > 0 else 0
    income_ratio  = (emi / total_income * 100) if total_income > 0 else 100

    # Map feature importance to High / Medium / Low
    def _impact(feat):
        imp = FEATURE_IMPORTANCES.get(feat, 0)
        if imp >= 0.10: return 'High'
        if imp >= 0.04: return 'Medium'
        return 'Low'

    reasons = []

    # 1. Credit History — always the top factor
    if credit_history == '1':
        reasons.append({
            'label':  'Credit History',
            'detail': 'You have a good credit history. This is the strongest positive signal for lenders.',
            'status': 'positive',
            'impact': _impact('Credit_History'),
            'icon':   'bi-shield-fill-check',
        })
    else:
        reasons.append({
            'label':  'Credit History',
            'detail': 'Poor or no credit history detected. This is the single biggest reason for rejection.',
            'status': 'negative',
            'impact': _impact('Credit_History'),
            'icon':   'bi-shield-fill-x',
        })

    # 2. EMI-to-Income ratio
    if income_ratio <= 30:
        reasons.append({
            'label':  'EMI vs Income',
            'detail': f'Your estimated monthly EMI (₹{emi:,.0f}) is only {income_ratio:.1f}% of your total income — well within the safe 40% limit.',
            'status': 'positive',
            'impact': _impact('Balance_Income'),
            'icon':   'bi-graph-up-arrow',
        })
    elif income_ratio <= 50:
        reasons.append({
            'label':  'EMI vs Income',
            'detail': f'Your EMI (₹{emi:,.0f}) is {income_ratio:.1f}% of income. Slightly high — lenders prefer below 40%.',
            'status': 'neutral',
            'impact': _impact('Balance_Income'),
            'icon':   'bi-graph-up',
        })
    else:
        reasons.append({
            'label':  'EMI vs Income',
            'detail': f'Your EMI (₹{emi:,.0f}) is {income_ratio:.1f}% of income — exceeds the safe 40% threshold, raising repayment risk.',
            'status': 'negative',
            'impact': _impact('Balance_Income'),
            'icon':   'bi-graph-down-arrow',
        })

    # 3. Total Income
    if total_income >= 8000:
        reasons.append({
            'label':  'Total Income',
            'detail': f'Combined monthly income of ₹{total_income:,.0f} is strong and supports the requested loan.',
            'status': 'positive',
            'impact': _impact('TotalIncome_log'),
            'icon':   'bi-currency-rupee',
        })
    elif total_income >= 4000:
        reasons.append({
            'label':  'Total Income',
            'detail': f'Combined monthly income of ₹{total_income:,.0f} is moderate. A co-applicant could strengthen your profile.',
            'status': 'neutral',
            'impact': _impact('TotalIncome_log'),
            'icon':   'bi-currency-rupee',
        })
    else:
        reasons.append({
            'label':  'Total Income',
            'detail': f'Combined monthly income of ₹{total_income:,.0f} is low relative to the loan amount requested.',
            'status': 'negative',
            'impact': _impact('TotalIncome_log'),
            'icon':   'bi-currency-rupee',
        })

    # 4. Loan Amount
    if loan_amount <= 100:
        reasons.append({
            'label':  'Loan Amount',
            'detail': f'Requested amount of ₹{loan_amount:.0f}K is modest and manageable for your income level.',
            'status': 'positive',
            'impact': _impact('LoanAmount_log'),
            'icon':   'bi-bank',
        })
    elif loan_amount <= 200:
        reasons.append({
            'label':  'Loan Amount',
            'detail': f'Requested amount of ₹{loan_amount:.0f}K is within an acceptable range.',
            'status': 'neutral',
            'impact': _impact('LoanAmount_log'),
            'icon':   'bi-bank',
        })
    else:
        reasons.append({
            'label':  'Loan Amount',
            'detail': f'Requested amount of ₹{loan_amount:.0f}K is high. Consider reducing it to improve approval chances.',
            'status': 'negative',
            'impact': _impact('LoanAmount_log'),
            'icon':   'bi-bank',
        })

    # 5. Property Area
    area_map = {'Urban': 'positive', 'Semiurban': 'positive', 'Rural': 'neutral'}
    area_detail = {
        'Urban':     'Urban properties have higher resale value, reducing lender risk.',
        'Semiurban': 'Semiurban properties are viewed favourably by most lenders.',
        'Rural':     'Rural properties carry slightly higher risk for lenders due to lower liquidity.',
    }
    reasons.append({
        'label':  'Property Area',
        'detail': area_detail[property_area],
        'status': area_map[property_area],
        'impact': _impact('Property_Area'),
        'icon':   'bi-house-fill',
    })

    # 6. Education
    if education == 'Graduate':
        reasons.append({
            'label':  'Education',
            'detail': 'Graduate status is associated with higher earning potential and lower default risk.',
            'status': 'positive',
            'impact': _impact('Education'),
            'icon':   'bi-mortarboard-fill',
        })
    else:
        reasons.append({
            'label':  'Education',
            'detail': 'Non-graduate applicants are statistically at slightly higher default risk.',
            'status': 'neutral',
            'impact': _impact('Education'),
            'icon':   'bi-mortarboard',
        })

    # 7. Marital Status
    if married == 'Yes':
        reasons.append({
            'label':  'Marital Status',
            'detail': 'Married applicants often have dual income potential, which is viewed positively.',
            'status': 'positive',
            'impact': _impact('Married'),
            'icon':   'bi-people-fill',
        })
    else:
        reasons.append({
            'label':  'Marital Status',
            'detail': 'Single applicants rely on a single income stream.',
            'status': 'neutral',
            'impact': _impact('Married'),
            'icon':   'bi-person-fill',
        })

    # 8. Loan Term
    if loan_term >= 300:
        reasons.append({
            'label':  'Loan Term',
            'detail': f'A {int(loan_term)}-month term keeps your monthly EMI low, improving repayment capacity.',
            'status': 'positive',
            'impact': _impact('Loan_Amount_Term'),
            'icon':   'bi-calendar-check',
        })
    else:
        reasons.append({
            'label':  'Loan Term',
            'detail': f'A {int(loan_term)}-month term means higher monthly EMIs. A longer term could reduce repayment pressure.',
            'status': 'neutral',
            'impact': _impact('Loan_Amount_Term'),
            'icon':   'bi-calendar2',
        })

    # Sort: negatives first for rejected, positives first for approved
    order = {'negative': 0, 'neutral': 1, 'positive': 2}
    if result == 'Rejected':
        reasons.sort(key=lambda r: (order[r['status']], r['impact'] != 'High'))
    else:
        reasons.sort(key=lambda r: (-order[r['status']], r['impact'] != 'High'))

    return reasons


# ── Helper: build feature DataFrame ───────────────────────────
def _preprocess(data):
    gender          = GENDER_MAP[data['gender']]
    married         = MARRIED_MAP[data['married']]
    dependents      = int(str(data['dependents']).replace('3+', '3'))
    education       = EDUCATION_MAP[data['education']]
    self_employed   = SELF_EMPLOYED_MAP[data['self_employed']]
    applicant_inc   = float(data['applicant_income'])
    coapplicant_inc = float(data['coapplicant_income'])
    loan_amount     = float(data['loan_amount'])
    loan_term       = float(data['loan_amount_term'])
    credit_history  = float(data['credit_history'])
    property_area   = PROPERTY_MAP[data['property_area']]

    total_income     = applicant_inc + coapplicant_inc
    total_income_log = np.log1p(total_income)
    loan_amount_log  = np.log1p(loan_amount)
    emi              = loan_amount / loan_term if loan_term > 0 else 0
    balance_income   = total_income - (emi * 1000)

    row = [
        gender, married, dependents, education, self_employed,
        applicant_inc, coapplicant_inc, loan_amount,
        loan_term, credit_history, property_area,
        total_income_log, loan_amount_log, emi, balance_income,
    ]
    return pd.DataFrame([row], columns=FEATURES)

"""
forms.py
--------
Django form for collecting loan application inputs.
All fields mirror the ML model's expected features.
"""

from django import forms


# Loan type definitions with interest rates and rules
LOAN_TYPES = {
    'home':     {'label': 'Home Loan',     'rate': 8.5,  'icon': 'bi-house-fill',    'color': '#a78bfa', 'desc': 'For purchase or construction of property'},
    'personal': {'label': 'Personal Loan', 'rate': 13.5, 'icon': 'bi-person-fill',   'color': '#f472b6', 'desc': 'Unsecured loan for personal expenses'},
    'car':      {'label': 'Car Loan',      'rate': 9.5,  'icon': 'bi-car-front-fill','color': '#2dd4bf', 'desc': 'For purchase of new or used vehicles'},
    'business': {'label': 'Business Loan', 'rate': 12.0, 'icon': 'bi-briefcase-fill','color': '#fb923c', 'desc': 'For business expansion or working capital'},
    'education':{'label': 'Education Loan','rate': 10.5, 'icon': 'bi-mortarboard-fill','color':'#60a5fa', 'desc': 'For higher education and skill development'},
}


class LoanApplicationForm(forms.Form):
    # ── Loan Type ──────────────────────────────────────────────
    LOAN_TYPE_CHOICES = [(k, v['label']) for k, v in LOAN_TYPES.items()]

    loan_type = forms.ChoiceField(
        choices=LOAN_TYPE_CHOICES,
        widget=forms.HiddenInput(),
        initial='home',
        required=True,
    )

    # ── Personal Info ──────────────────────────────────────────
    GENDER_CHOICES = [('Male', 'Male'), ('Female', 'Female')]
    MARRIED_CHOICES = [('Yes', 'Yes'), ('No', 'No')]
    DEPENDENTS_CHOICES = [('0', '0'), ('1', '1'), ('2', '2'), ('3+', '3+')]
    EDUCATION_CHOICES = [('Graduate', 'Graduate'), ('Not Graduate', 'Not Graduate')]
    SELF_EMPLOYED_CHOICES = [('Yes', 'Yes'), ('No', 'No')]
    PROPERTY_CHOICES = [('Urban', 'Urban'), ('Semiurban', 'Semiurban'), ('Rural', 'Rural')]
    CREDIT_CHOICES = [('1', 'Good (1)'), ('0', 'Bad (0)')]

    gender = forms.ChoiceField(
        choices=GENDER_CHOICES,
        widget=forms.Select(attrs={'class': 'form-select'}),
        label='Gender'
    )
    married = forms.ChoiceField(
        choices=MARRIED_CHOICES,
        widget=forms.Select(attrs={'class': 'form-select'}),
        label='Marital Status'
    )
    dependents = forms.ChoiceField(
        choices=DEPENDENTS_CHOICES,
        widget=forms.Select(attrs={'class': 'form-select'}),
        label='Number of Dependents'
    )
    education = forms.ChoiceField(
        choices=EDUCATION_CHOICES,
        widget=forms.Select(attrs={'class': 'form-select'}),
        label='Education'
    )
    self_employed = forms.ChoiceField(
        choices=SELF_EMPLOYED_CHOICES,
        widget=forms.Select(attrs={'class': 'form-select'}),
        label='Self Employed'
    )

    # ── Financial Info ─────────────────────────────────────────
    applicant_income = forms.IntegerField(
        min_value=0,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'e.g. 50000'
        }),
        label='Applicant Monthly Income (₹)'
    )
    coapplicant_income = forms.FloatField(
        min_value=0,
        initial=0,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'e.g. 25000 (0 if none)'
        }),
        label='Co-Applicant Monthly Income (₹)'
    )
    loan_amount = forms.FloatField(
        min_value=1,
        widget=forms.NumberInput(attrs={
            'class': 'form-control',
            'placeholder': 'e.g. 1500000 (enter full amount in ₹)'
        }),
        label='Loan Amount (₹)'
    )
    loan_amount_term = forms.ChoiceField(
        choices=[
            ('360', '360 months (30 yrs)'),
            ('300', '300 months (25 yrs)'),
            ('240', '240 months (20 yrs)'),
            ('180', '180 months (15 yrs)'),
            ('120', '120 months (10 yrs)'),
            ('84',  '84 months (7 yrs)'),
            ('60',  '60 months (5 yrs)'),
            ('36',  '36 months (3 yrs)'),
            ('480', '480 months (40 yrs)'),
        ],
        widget=forms.Select(attrs={'class': 'form-select'}),
        label='Loan Amount Term'
    )
    credit_history = forms.ChoiceField(
        choices=CREDIT_CHOICES,
        widget=forms.Select(attrs={'class': 'form-select'}),
        label='Credit History'
    )
    property_area = forms.ChoiceField(
        choices=PROPERTY_CHOICES,
        widget=forms.Select(attrs={'class': 'form-select'}),
        label='Property Area'
    )

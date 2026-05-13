"""
bank/forms.py
-------------
All forms: LoanApplicationForm, BulkUploadForm, WhatIfForm, EMIForm
"""

from django import forms


class LoanApplicationForm(forms.Form):
    GENDER_CHOICES       = [('Male','Male'),('Female','Female')]
    MARRIED_CHOICES      = [('Yes','Yes'),('No','No')]
    DEPENDENTS_CHOICES   = [('0','0'),('1','1'),('2','2'),('3+','3+')]
    EDUCATION_CHOICES    = [('Graduate','Graduate'),('Not Graduate','Not Graduate')]
    EMP_CHOICES          = [('Yes','Yes'),('No','No')]
    PROPERTY_CHOICES     = [('Urban','Urban'),('Semiurban','Semiurban'),('Rural','Rural')]
    CREDIT_CHOICES       = [('1','Good (1)'),('0','Poor (0)')]
    LOAN_TYPE_CHOICES    = [('home','Home Loan'),('personal','Personal Loan'),('car','Car Loan')]
    TERM_CHOICES         = [('360','360 months'),('300','300 months'),('240','240 months'),
                             ('180','180 months'),('120','120 months'),('84','84 months'),
                             ('60','60 months'),('36','36 months'),('480','480 months')]

    loan_type          = forms.ChoiceField(choices=LOAN_TYPE_CHOICES,
                            widget=forms.Select(attrs={'class':'form-select'}))
    gender             = forms.ChoiceField(choices=GENDER_CHOICES,
                            widget=forms.Select(attrs={'class':'form-select'}))
    married            = forms.ChoiceField(choices=MARRIED_CHOICES,
                            widget=forms.Select(attrs={'class':'form-select'}))
    dependents         = forms.ChoiceField(choices=DEPENDENTS_CHOICES,
                            widget=forms.Select(attrs={'class':'form-select'}))
    education          = forms.ChoiceField(choices=EDUCATION_CHOICES,
                            widget=forms.Select(attrs={'class':'form-select'}))
    self_employed      = forms.ChoiceField(choices=EMP_CHOICES,
                            widget=forms.Select(attrs={'class':'form-select'}))
    applicant_income   = forms.FloatField(min_value=0,
                            widget=forms.NumberInput(attrs={'class':'form-control','placeholder':'e.g. 5000'}))
    coapplicant_income = forms.FloatField(min_value=0, initial=0,
                            widget=forms.NumberInput(attrs={'class':'form-control','placeholder':'0'}))
    loan_amount        = forms.FloatField(min_value=1,
                            widget=forms.NumberInput(attrs={'class':'form-control','placeholder':'e.g. 150 (thousands)'}))
    loan_amount_term   = forms.ChoiceField(choices=TERM_CHOICES,
                            widget=forms.Select(attrs={'class':'form-select'}))
    credit_history     = forms.ChoiceField(choices=CREDIT_CHOICES,
                            widget=forms.Select(attrs={'class':'form-select'}))
    property_area      = forms.ChoiceField(choices=PROPERTY_CHOICES,
                            widget=forms.Select(attrs={'class':'form-select'}))


class EMIForm(forms.Form):
    LOAN_TYPE_CHOICES = [('home','Home Loan (8.5%)'),
                         ('personal','Personal Loan (13.5%)'),
                         ('car','Car Loan (9.5%)'),
                         ('custom','Custom Rate')]
    loan_type    = forms.ChoiceField(choices=LOAN_TYPE_CHOICES,
                      widget=forms.Select(attrs={'class':'form-select','id':'emiLoanType'}))
    principal    = forms.FloatField(min_value=1000,
                      widget=forms.NumberInput(attrs={'class':'form-control','placeholder':'Loan amount in ₹'}))
    annual_rate  = forms.FloatField(min_value=0.1, max_value=50, required=False,
                      widget=forms.NumberInput(attrs={'class':'form-control','placeholder':'e.g. 10.5','step':'0.1'}))
    months       = forms.IntegerField(min_value=1, max_value=480,
                      widget=forms.NumberInput(attrs={'class':'form-control','placeholder':'e.g. 360'}))


class BulkUploadForm(forms.Form):
    csv_file = forms.FileField(
        widget=forms.FileInput(attrs={'class':'form-control','accept':'.csv'}),
        label='Upload CSV File'
    )


class WhatIfForm(forms.Form):
    applicant_income   = forms.FloatField(min_value=0,
                            widget=forms.NumberInput(attrs={'class':'form-control','id':'wi_income'}))
    loan_amount        = forms.FloatField(min_value=1,
                            widget=forms.NumberInput(attrs={'class':'form-control','id':'wi_loan'}))

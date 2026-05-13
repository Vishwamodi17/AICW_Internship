"""
bank/models.py
--------------
Database models for the Bank Simulation system.
"""

from django.db import models
from django.contrib.auth.models import User


class LoanApplication(models.Model):
    """Stores every loan application submitted by a user."""

    LOAN_TYPE_CHOICES = [
        ('home',     'Home Loan'),
        ('personal', 'Personal Loan'),
        ('car',      'Car Loan'),
    ]
    STATUS_CHOICES = [
        ('Approved', 'Approved'),
        ('Rejected', 'Rejected'),
    ]

    user              = models.ForeignKey(User, on_delete=models.CASCADE,
                                          related_name='loan_applications')
    loan_type         = models.CharField(max_length=20, choices=LOAN_TYPE_CHOICES)

    # Applicant details
    gender            = models.CharField(max_length=10)
    married           = models.CharField(max_length=5)
    dependents        = models.CharField(max_length=5)
    education         = models.CharField(max_length=20)
    self_employed     = models.CharField(max_length=5)
    applicant_income  = models.FloatField()
    coapplicant_income= models.FloatField(default=0)
    loan_amount       = models.FloatField()
    loan_amount_term  = models.FloatField()
    credit_history    = models.CharField(max_length=5)
    property_area     = models.CharField(max_length=15)

    # Prediction output
    result            = models.CharField(max_length=10, choices=STATUS_CHOICES)
    approved_prob     = models.FloatField()
    rejected_prob     = models.FloatField()
    monthly_emi       = models.FloatField(default=0)
    interest_rate     = models.FloatField(default=0)
    model_used        = models.CharField(max_length=50, default='')
    receipt_id        = models.CharField(max_length=30, unique=True)

    created_at        = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ['-created_at']

    def __str__(self):
        return f"{self.user.username} | {self.loan_type} | {self.result} | {self.created_at:%d-%m-%Y}"

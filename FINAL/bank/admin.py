from django.contrib import admin
from .models import LoanApplication

@admin.register(LoanApplication)
class LoanApplicationAdmin(admin.ModelAdmin):
    list_display  = ['user','loan_type','result','approved_prob','monthly_emi','created_at']
    list_filter   = ['result','loan_type','created_at']
    search_fields = ['user__username','receipt_id']
    readonly_fields = ['receipt_id','created_at']

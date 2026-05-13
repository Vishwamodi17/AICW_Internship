"""
bank/views.py
-------------
All views for the LoanAI Bank Simulation system.
"""

import csv, io, json
from datetime import datetime

from django.shortcuts               import render, redirect, get_object_or_404
from django.contrib.auth            import authenticate, login, logout
from django.contrib.auth.forms      import UserCreationForm
from django.contrib.auth.decorators import login_required
from django.contrib                 import messages

from .forms  import LoanApplicationForm, BulkUploadForm
from .models import LoanApplication
from .ml.engine import predict, whatif, load, LOAN_TYPES


# ── Auth ───────────────────────────────────────────────────────
def bank_register(request):
    if request.user.is_authenticated:
        return redirect('bank_dashboard')
    form = UserCreationForm(request.POST or None)
    if request.method == 'POST' and form.is_valid():
        user = form.save()
        login(request, user)
        messages.success(request, 'Account created! Welcome to LoanAI.')
        return redirect('bank_dashboard')
    return render(request, 'bank/register.html', {'form': form})


def bank_login(request):
    if request.user.is_authenticated:
        return redirect('bank_dashboard')
    error = None
    if request.method == 'POST':
        user = authenticate(
            request,
            username=request.POST.get('username', ''),
            password=request.POST.get('password', ''),
        )
        if user:
            login(request, user)
            return redirect(request.GET.get('next', 'bank_dashboard'))
        error = 'Invalid credentials. Please try again.'
    return render(request, 'bank/login.html', {'error': error})


def bank_logout(request):
    logout(request)
    return redirect('bank_home')


# ── Public Home ────────────────────────────────────────────────
def bank_home(request):
    pkg = load()
    return render(request, 'bank/home.html', {
        'all_results': pkg['all_results'],
        'loan_types':  LOAN_TYPES,
    })


# ── Dashboard ──────────────────────────────────────────────────
@login_required(login_url='bank_login')
def bank_dashboard(request):
    apps     = LoanApplication.objects.filter(user=request.user)
    total    = apps.count()
    approved = apps.filter(result='Approved').count()
    rejected = apps.filter(result='Rejected').count()
    recent   = apps[:5]

    monthly_data = {}
    for a in apps:
        key = a.created_at.strftime('%b %Y')
        monthly_data[key] = monthly_data.get(key, 0) + 1

    return render(request, 'bank/dashboard.html', {
        'apps':          apps,
        'total':         total,
        'approved':      approved,
        'rejected':      rejected,
        'recent':        recent,
        'monthly_json':  json.dumps(monthly_data),
        'approval_rate': round(approved / total * 100, 1) if total else 0,
    })


# ── Apply ──────────────────────────────────────────────────────
@login_required(login_url='bank_login')
def bank_apply(request):
    return render(request, 'bank/apply.html', {
        'form':       LoanApplicationForm(),
        'loan_types': LOAN_TYPES,
        'prefill':    '{}',
    })


@login_required(login_url='bank_login')
def bank_predict(request):
    if request.method != 'POST':
        return redirect('bank_apply')

    form = LoanApplicationForm(request.POST)
    if not form.is_valid():
        return render(request, 'bank/apply.html', {
            'form':       form,
            'loan_types': LOAN_TYPES,
            'prefill':    '{}',
        })

    data      = form.cleaned_data
    loan_type = data['loan_type']
    result    = predict(data, loan_type)

    receipt_id = f"LA{request.user.id:04d}{datetime.now().strftime('%d%m%y%H%M%S')}"
    app = LoanApplication.objects.create(
        user               = request.user,
        loan_type          = loan_type,
        gender             = data['gender'],
        married            = data['married'],
        dependents         = data['dependents'],
        education          = data['education'],
        self_employed      = data['self_employed'],
        applicant_income   = data['applicant_income'],
        coapplicant_income = data['coapplicant_income'],
        loan_amount        = data['loan_amount'],
        loan_amount_term   = data['loan_amount_term'],
        credit_history     = data['credit_history'],
        property_area      = data['property_area'],
        result             = result['result'],
        approved_prob      = result['approved_prob'],
        rejected_prob      = result['rejected_prob'],
        monthly_emi        = result['emi_info']['monthly_emi'],
        interest_rate      = result['emi_info']['rate'],
        model_used         = result['model_used'],
        receipt_id         = receipt_id,
    )

    scenarios = whatif(data, loan_type)

    return render(request, 'bank/result.html', {
        'result':         result,
        'form_data':      data,
        'app':            app,
        'receipt_id':     receipt_id,
        'receipt_date':   datetime.now().strftime('%d %B %Y, %I:%M %p'),
        'applicant_name': request.user.get_full_name() or request.user.username,
        'scenarios':      json.dumps(scenarios),
    })


# ── Application Detail ─────────────────────────────────────────
@login_required(login_url='bank_login')
def bank_app_detail(request, pk):
    app = get_object_or_404(LoanApplication, pk=pk, user=request.user)
    return render(request, 'bank/app_detail.html', {'app': app})


# ── Bulk Upload ────────────────────────────────────────────────
@login_required(login_url='bank_login')
def bank_bulk(request):
    results = None
    form    = BulkUploadForm(request.POST or None, request.FILES or None)
    if request.method == 'POST' and form.is_valid():
        decoded = request.FILES['csv_file'].read().decode('utf-8')
        reader  = csv.DictReader(io.StringIO(decoded))
        results = []
        for row in reader:
            try:
                r = predict(row, row.get('loan_type', 'home'))
                results.append({
                    'name':          row.get('name', 'N/A'),
                    'result':        r['result'],
                    'approved_prob': r['approved_prob'],
                    'emi':           r['emi_info']['monthly_emi'],
                })
            except Exception:
                results.append({
                    'name': row.get('name', '?'),
                    'result': 'Error', 'approved_prob': 0, 'emi': 0,
                })
    return render(request, 'bank/bulk.html', {'form': form, 'results': results})


# ── Model Comparison ───────────────────────────────────────────
@login_required(login_url='bank_login')
def bank_models(request):
    pkg = load()
    return render(request, 'bank/models.html', {
        'all_results': pkg['all_results'],
        'best_name':   pkg['best_name'],
        'importances': sorted(
            pkg['importances'].items(), key=lambda x: x[1], reverse=True
        ),
    })


# ── Admin ──────────────────────────────────────────────────────
@login_required(login_url='bank_login')
def bank_admin_apps(request):
    if not request.user.is_staff:
        return redirect('bank_dashboard')
    apps = LoanApplication.objects.all().select_related('user')
    return render(request, 'bank/admin_apps.html', {'apps': apps})

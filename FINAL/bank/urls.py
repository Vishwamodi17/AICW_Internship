from django.urls import path
from . import views

urlpatterns = [
    path('',              views.bank_home,       name='bank_home'),
    path('register/',     views.bank_register,   name='bank_register'),
    path('login/',        views.bank_login,      name='bank_login'),
    path('logout/',       views.bank_logout,     name='bank_logout'),
    path('dashboard/',    views.bank_dashboard,  name='bank_dashboard'),
    path('apply/',        views.bank_apply,      name='bank_apply'),
    path('predict/',      views.bank_predict,    name='bank_predict'),
    path('bulk/',         views.bank_bulk,       name='bank_bulk'),
    path('models/',       views.bank_models,     name='bank_models'),
    path('admin-apps/',   views.bank_admin_apps, name='bank_admin_apps'),
    path('app/<int:pk>/', views.bank_app_detail, name='bank_app_detail'),
]

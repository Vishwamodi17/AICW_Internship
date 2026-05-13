from django.urls import path
from . import views

urlpatterns = [
    path('',           views.landing,      name='landing'),
    path('register/',  views.register_view, name='register'),
    path('login/',     views.login_view,    name='login'),
    path('logout/',    views.logout_view,   name='logout'),
    path('apply/',     views.apply,         name='apply'),
    path('emi-review/',views.emi_review,    name='emi_review'),
    path('predict/',   views.predict,       name='predict'),
]

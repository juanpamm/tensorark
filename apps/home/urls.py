from django.urls import path
from apps.home.views import index

urlpatterns = [
    path('', index, name='index')
]
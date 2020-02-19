from django.urls import path
from apps.textproc.views import upload_text_nn_template

urlpatterns = [
    path('upload_text_nn_page/', upload_text_nn_template, name='upload_text_nn_template'),
]
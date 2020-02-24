from django.urls import path
from apps.textproc.views import upload_text_nn_template, load_text_set

urlpatterns = [
    path('upload_text_nn_page/', upload_text_nn_template, name='upload_text_nn_template'),
    path('upload_txt_set/', load_text_set, name='load_text_set'),
]
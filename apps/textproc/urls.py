from django.urls import path
from apps.textproc.views import upload_text_nn_template, load_text_set, build_textproc_nn_template, \
    execute_model_training

urlpatterns = [
    path('upload_text_nn_page/', upload_text_nn_template, name='upload_text_nn_template'),
    path('upload_txt_set/', load_text_set, name='load_text_set'),
    path('build_nn_page/<str:folder>/', build_textproc_nn_template, name='build_textproc_nn_template'),
    path('train_nn/', execute_model_training, name='execute_model_training')
]

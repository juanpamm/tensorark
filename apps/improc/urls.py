from django.urls import path
from apps.improc.views import build_improc_nn_template, execute_nn_training, load_image_set, download_saved_model

urlpatterns = [
    path('build_nn', build_improc_nn_template, name='build_improc_nn_template'),
    path('train_nn/', execute_nn_training, name='execute_nn_training'),
    path('upload_img_set/', load_image_set, name='load_image_set'),
    path('download_model/', download_saved_model, name='download_saved_model')
]
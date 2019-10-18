from django.urls import path
from apps.improc.views import index, execute_nn_training, load_image_set, download_saved_model

urlpatterns = [
    path('', index, name='index'),
    path('train_nn/', execute_nn_training, name='execute_nn_training'),
    path('upload_img_set/', load_image_set, name='load_image_set'),
    path('download_model/', download_saved_model, name='download_saved_model')
]
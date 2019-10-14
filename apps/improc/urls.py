from django.urls import path
from apps.improc.views import index, execute_nn_training, load_image_set

urlpatterns = [
    path('', index, name='index'),
    path('train_nn/', execute_nn_training, name='execute_nn_training'),
    path('upload_img_set/', load_image_set, name='load_image_set')
]
from django.urls import path
from apps.improc.views import index, execute_nn_training

urlpatterns = [
    path('', index, name='index'),
    path('train_nn/', execute_nn_training, name='execute_nn_training')
]
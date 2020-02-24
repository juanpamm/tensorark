from django.core.files.storage import default_storage
from django.shortcuts import render
from django.http import JsonResponse, HttpResponse, Http404
from tensorark.settings import MEDIA_ROOT
from utils import utils
from tensorflow import keras
import json
import os.path
import numpy as np
import tensorflow as tf
import shutil


def upload_text_nn_template(request):
    return render(request, 'textproc/upload_text_set_nn.html')


def load_text_set(request):
    file = request.FILES['file']
    action = request.POST.get('action')
    app = request.POST.get('app')
    default_storage.save(file.name, file)
    working_dir_name = utils.get_name_for_working_dir(MEDIA_ROOT, action, app)
    dst_path = os.path.join(MEDIA_ROOT, working_dir_name)
    # converted_path = os.path.join(dst_path, 'converted_set')
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)

    # Extraction of the image set loaded by the user
    utils.file_extraction_manager(MEDIA_ROOT, file, dst_path)
    extracted_dir = os.listdir(dst_path)[0]
    path_to_extracted_dir = os.path.join(dst_path, extracted_dir)
    '''
    # Paths to training and testing sets
    path_to_training_set = os.path.join(path_to_extracted_dir, 'training')
    path_to_testing_set = os.path.join(path_to_extracted_dir, 'testing')

    # Set the names for the classes
    utils.set_name_classes(path_to_training_set)

    # Remove image_set folder
    shutil.rmtree(path_to_extracted_dir, ignore_errors=True)
    '''
    result = {
        'upload_val': True,
        'txt_set_name': working_dir_name
    }
    json_data = json.dumps(result)
    return JsonResponse(json_data, safe=False)

from django.core.files.storage import default_storage
from django.shortcuts import render
from django.http import JsonResponse, HttpResponse, Http404
from tensorark.settings import MEDIA_ROOT
from utils import utils
from tensorflow import keras
import json
import os.path
import numpy as np
import random
import tensorflow as tf
import shutil


def upload_text_nn_template(request):
    return render(request, 'textproc/upload_text_set_nn.html')


def build_textproc_nn_template(request, folder):
    contexto = {'folder': folder}
    return render(request, 'textproc/build_textproc_nn.html', contexto)


def load_text_set(request):
    file = request.FILES['file']
    action = request.POST.get('action')
    app = request.POST.get('app')
    default_storage.save(file.name, file)
    working_dir_name = utils.get_name_for_working_dir(MEDIA_ROOT, action, app)
    dst_path = os.path.join(MEDIA_ROOT, working_dir_name)
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)

    # Extraction of the text set loaded by the user
    utils.file_extraction_manager(MEDIA_ROOT, file, dst_path)
    data = preprocess_train_test_data(dst_path)
    determine_model_to_use(data)
    result = {
        'upload_val': True,
        'txt_set_name': working_dir_name
    }
    json_data = json.dumps(result)
    return JsonResponse(json_data, safe=False)


def read_train_text_files(path_to_dataset, data_type):
    # Load the training or test data, according to the value of the data_type variable
    texts = []
    labels = []
    for category in ['pos', 'neg']:
        train_path = os.path.join(path_to_dataset, data_type, category)
        for fname in sorted(os.listdir(train_path)):
            if fname.endswith('.txt'):
                with open(os.path.join(train_path, fname), encoding="utf8") as f:
                    texts.append(f.read())
                labels.append(0 if category == 'neg' else 1)

    return texts, labels


def preprocess_train_test_data(work_dir):
    work_dir_full_path = os.path.join(MEDIA_ROOT, work_dir)
    data_dir = os.listdir(work_dir_full_path)[0]
    train_test_set_path = os.path.join(work_dir_full_path, data_dir)

    train_texts_and_labels = read_train_text_files(train_test_set_path, 'train')
    test_texts_and_labels = read_train_text_files(train_test_set_path, 'test')

    # Shuffle the training data and labels.
    random.seed()
    random.shuffle(train_texts_and_labels[0])
    random.seed()
    random.shuffle(train_texts_and_labels[1])

    return (train_texts_and_labels[0], np.array(train_texts_and_labels[1])), \
           (test_texts_and_labels[0], np.array(test_texts_and_labels[1]))


def determine_model_to_use(preprocessed_data):
    words_per_sample = [len(s.split()) for s in preprocessed_data[0][0]]
    median_num_words_per_sample = np.median(words_per_sample)
    num_samples = len(preprocessed_data[0][0])
    print(num_samples)

    num_samples_num_words_ratio = num_samples / median_num_words_per_sample
    print(num_samples_num_words_ratio)

    if num_samples_num_words_ratio < 1500:
        print('Multi-Layer Perceptron')
    elif num_samples_num_words_ratio >= 1500:
        print('Sequence model')

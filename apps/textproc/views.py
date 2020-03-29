from django.core.files.storage import default_storage
from django.shortcuts import render
from django.http import JsonResponse, HttpResponse, Http404
from tensorark.settings import MEDIA_ROOT
from utils import utils
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
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

    num_samples_num_words_ratio = num_samples / median_num_words_per_sample

    if num_samples_num_words_ratio < 1500:
        x_train, x_val = ngram_vectorize(preprocessed_data[0][0], preprocessed_data[0][1], preprocessed_data[1][0])
    elif num_samples_num_words_ratio >= 1500:
        print('Sequence model')


def ngram_vectorize(train_texts, train_labels, val_texts):
    # Vectorization parameters
    # Range (inclusive) of n-gram sizes for tokenizing text.
    ngram_range = (1, 2)

    # Limit on the number of features. We use the top 20K features.
    top_features = 20000

    # Whether text should be split into word or character n-grams.
    # One of 'word', 'char'.
    token_mode = 'word'

    # Minimum document/corpus frequency below which a token will be discarded.
    min_freq_to_discard_token = 2

    """Vectorizes texts as n-gram vectors.

    1 text = 1 tf-idf vector the length of vocabulary of unigrams + bigrams.

    # Arguments
        train_texts: list, training text strings.
        train_labels: np.ndarray, training labels.
        val_texts: list, validation text strings.

    # Returns
        x_train, x_val: vectorized training and validation texts
    """
    # Create keyword arguments to pass to the 'tf-idf' vectorizer.
    kwargs = {
            'ngram_range': ngram_range,  # Use 1-grams + 2-grams.
            'dtype': 'int32',
            'strip_accents': 'unicode',
            'decode_error': 'replace',
            'analyzer': token_mode,  # Split text into word tokens.
            'min_df': min_freq_to_discard_token,
    }
    vectorizer = TfidfVectorizer(**kwargs)

    # Learn vocabulary from training texts and vectorize training texts.
    x_train = vectorizer.fit_transform(train_texts)

    # Vectorize validation texts.
    x_val = vectorizer.transform(val_texts)

    # Select top 'k' of the vectorized features.
    selector = SelectKBest(f_classif, k=min(top_features, x_train.shape[1]))
    selector.fit(x_train, train_labels)
    x_train = selector.transform(x_train).astype('float32')
    x_val = selector.transform(x_val).astype('float32')
    return x_train, x_val

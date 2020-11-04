from django.core.files.storage import default_storage
from django.shortcuts import render
from django.http import JsonResponse, HttpResponse, Http404
from django.views.decorators.csrf import ensure_csrf_cookie
from tensorark.settings import MEDIA_ROOT
from utils import utils
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import f_classif
import joblib
from tensorflow import keras
import json
import os.path
import numpy as np
import random
import tensorflow as tf

graph = tf.Graph()
loaded_neural_network = keras.Sequential()
loaded_selector = SelectKBest()
loaded_vectorizer = TfidfVectorizer()


def upload_text_nn_template(request):
    return render(request, 'textproc/upload_text_set_nn.html')


def build_textproc_nn_template(request, folder):
    contexto = {'folder': folder}
    return render(request, 'textproc/build_textproc_nn.html', contexto)


def load_textproc_model_template(request):
    return render(request, 'textproc/load_textproc_model.html')


@ensure_csrf_cookie
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
    utils.file_extraction_manager(MEDIA_ROOT, file.name, dst_path)

    result = {
        'upload_val': True,
        'txt_set_name': working_dir_name
    }
    json_data = json.dumps(result)
    return JsonResponse(json_data, safe=False)


def get_classes(path_to_dataset, data_type):
    data_path = os.path.join(path_to_dataset, data_type)
    classes_names = os.listdir(data_path)

    return classes_names


def read_train_text_files(path_to_dataset, data_type):
    class_names = get_classes(path_to_dataset, data_type)
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


def preprocess_train_test_data(path_to_dataset):
    train_texts_and_labels = read_train_text_files(path_to_dataset, 'train')
    test_texts_and_labels = read_train_text_files(path_to_dataset, 'test')

    # Shuffle the training data and labels.
    random.seed(123)
    random.shuffle(train_texts_and_labels[0])
    random.seed(123)
    random.shuffle(train_texts_and_labels[1])

    return ((train_texts_and_labels[0], np.array(train_texts_and_labels[1])),
            (test_texts_and_labels[0], np.array(test_texts_and_labels[1])))


def execute_model_training(request):
    # Variables needed for the training process
    layers = int(request.POST.get('layers'))
    nodes = json.loads(request.POST.get('nodes'))
    activation_functions = json.loads(request.POST.get('act_func'))
    output_act_func = request.POST.get('output_act_func')
    epochs = int(request.POST.get('epochs'))
    fold_name = request.POST.get('folder')
    dst_path = os.path.join(MEDIA_ROOT, fold_name)
    for i in range(len(nodes)):
        nodes[i] = int(nodes[i])

    # Execute function to train the neural network
    results = textproc_train_neural_network(layers, nodes, activation_functions, epochs, output_act_func, dst_path)

    # Setting the information to be sent to the client
    acc_percentage = results.get("accuracy") * 100
    st_percentage = '{number:.{digits}f}'.format(number=acc_percentage, digits=2)
    training_result = {
        'net_accuracy': st_percentage,
        'loss_plot': results.get("loss_plot_img_name"),
        'accuracy_plot': results.get("accuracy_plot_img_name"),
        'conf_matrix': results.get("conf_matrix_name")
    }

    json_data = json.dumps(training_result)
    return JsonResponse(json_data, safe=False)


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
            'dtype': 'float32',
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

    return x_train, x_val, selector, vectorizer


def textproc_add_layers_to_network(model, nodes, activation_func):
    if activation_func == 'relu':
        model.add(keras.layers.Dense(nodes, activation=tf.nn.relu))
    elif activation_func == 'sigmoid':
        model.add(keras.layers.Dense(nodes, activation=tf.nn.sigmoid))
    elif activation_func == 'tanh':
        model.add(keras.layers.Dense(nodes, activation=tf.nn.tanh))
    elif activation_func == 'elu':
        model.add(keras.layers.Dense(nodes, activation=tf.nn.elu))
    elif activation_func == 'softmax':
        model.add(keras.layers.Dense(nodes, activation=tf.nn.softmax))


def textproc_build_neural_network(nlayers, nodes, num_classes, act_functions, output_act_func, input_shape):
    model = keras.Sequential([
        keras.layers.Dropout(rate=0.0, input_shape=input_shape)
    ])

    # Construction of the hidden layers
    for i in range(nlayers):
        textproc_add_layers_to_network(model, nodes[i], act_functions[i])

    # Construction of the output layer
    textproc_add_layers_to_network(model, 1, output_act_func)

    model.compile(optimizer='adam',
                  loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def textproc_train_neural_network(layers, nodes, act_functions, epochs, output_act_func, working_dir_name):
    with graph.as_default():
        sess = tf.compat.v1.Session()
        work_dir_full_path = os.path.join(MEDIA_ROOT, working_dir_name)
        data_dir = os.listdir(work_dir_full_path)[0]
        train_test_set_path = os.path.join(work_dir_full_path, data_dir)
        classes = get_classes(train_test_set_path, 'train')              
        # Checkpoint for network
        checkpoint_path = os.path.join(work_dir_full_path, 'saved_model')
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)
            
        (train_texts, train_labels), (test_texts, test_labels) = preprocess_train_test_data(train_test_set_path)

        x_train, val_texts, selector, vectorizer = ngram_vectorize(train_texts, train_labels, test_texts)
        # Construction, training and saving of the neural network
        model = textproc_build_neural_network(layers, nodes, len(classes), act_functions, output_act_func,
                                              x_train.shape[1:])

        history = model.fit(
                    x_train,
                    train_labels,
                    epochs=epochs,
                    validation_data=(val_texts, test_labels))

        selector_dump_path = os.path.join(checkpoint_path, 'selector.pkl')
        vectorizer_dump_path = os.path.join(checkpoint_path, 'vectorizer.pkl')
        joblib.dump(vectorizer, vectorizer_dump_path)
        joblib.dump(selector, selector_dump_path)
        utils.save_model_to_json(checkpoint_path, model)
        model.save(os.path.join(checkpoint_path, 'neural_network.h5'))
        test_loss, test_acc = model.evaluate(val_texts, test_labels)

        working_dir_folder_name = os.path.split(working_dir_name)[1]

        # Use the test set for prediction
        predicts = (model.predict(val_texts) > 0.5).astype("int32")

        # Build confusion matrix
        con_mat = tf.compat.v1.confusion_matrix(labels=test_labels, predictions=predicts)
        con_mat_val = con_mat.eval(session=sess)

        # Path to confusion matrix image
        con_matrix_img_name = 'textproc_conf_matrix.png'
        con_matrix_img_path = os.path.join(working_dir_name, con_matrix_img_name)
        img_name_to_send = working_dir_folder_name + '/' + con_matrix_img_name

        utils.confusion_matrix_plotter(con_mat_val, classes, con_matrix_img_path, checkpoint_path)

        # Code to create plot images
        history_dict = history.history

        acc = history_dict['accuracy']
        val_acc = history_dict['val_accuracy']
        loss = history_dict['loss']
        val_loss = history_dict['val_loss']
        epochs = range(1, len(acc) + 1)

        loss_plot_img_name = "textproc_train_val_loss.png"
        loss_plot_img_partial_path = working_dir_folder_name + '/' + loss_plot_img_name
        loss_plot_img_path = os.path.join(working_dir_name, loss_plot_img_name)

        accuracy_plot_img_name = "textproc_acc_val_accu.png"
        accuracy_plot_img_partial_path = working_dir_folder_name + '/' + accuracy_plot_img_name
        accuracy_plot_img_path = os.path.join(working_dir_name, accuracy_plot_img_name)

        utils.loss_accuracy_plotter(epochs, loss, val_loss, acc, val_acc, checkpoint_path, loss_plot_img_path,
                                    accuracy_plot_img_path)

        utils.compress_model_folder(working_dir_name)

    return {"accuracy": test_acc, "conf_matrix_name": img_name_to_send,
            "loss_plot_img_name": loss_plot_img_partial_path, "accuracy_plot_img_name": accuracy_plot_img_partial_path}


def textproc_download_saved_model(request, dir_name):
    dst_path = os.path.join(MEDIA_ROOT, dir_name)
    full_file_path = os.path.join(dst_path, 'nn_model.zip')
    if os.path.exists(full_file_path):
        with open(full_file_path, 'rb') as fh:
            response = HttpResponse(fh.read(), content_type="application/zip")
            response['Content-Disposition'] = 'inline; filename=' + os.path.basename(full_file_path)
            return response
    raise Http404


def textproc_load_model(request):
    global loaded_neural_network, loaded_selector, loaded_vectorizer
    keras.backend.clear_session()
    action = request.POST.get('action')
    model_zip = request.FILES['file']
    app = request.POST.get('app')
    activation_funcs = {
        'relu': 'Rectified Linear Unit',
        'sigmoid': 'Sigmoid',
        'tanh': 'Hyperbolic Tangent',
        'elu': 'Exponential Linear Unit',
        'softmax': 'Softmax'
    }

    default_storage.save(model_zip.name, model_zip)
    load_dir_name = utils.get_name_for_working_dir(MEDIA_ROOT, action, app)
    dst_path = os.path.join(MEDIA_ROOT, load_dir_name)
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)

    utils.file_extraction_manager(MEDIA_ROOT, model_zip.name, dst_path)
    path_to_json_file = os.path.join(dst_path, 'json_nn.json')
    path_to_model_file = os.path.join(dst_path, 'neural_network.h5')
    path_to_selector_file = os.path.join(dst_path, 'selector.pkl')
    path_to_vectorizer_file = os.path.join(dst_path, 'vectorizer.pkl')

    f = open(path_to_json_file, "r")
    f_content = json.loads(f.read())
    f.close()
    f_layers = f_content['config']['layers']
    len_f_layers = len(f_layers)

    result = {
        'num_layers': len_f_layers,
        'load_dir_name': load_dir_name,
        'hidden_layers': []
    }

    for i in range(len_f_layers):
        if i == 0:
            result['input_layer'] = f_layers[i]['config']['batch_input_shape'][1]
        elif i == (len_f_layers - 1):
            result['output_layer'] = [f_layers[i]['config']['units'],
                                      activation_funcs.get(f_layers[i]['config']['activation'])]
        elif i != 1:
            result['hidden_layers'].append([f_layers[i]['config']['units'],
                                            activation_funcs.get(f_layers[i]['config']['activation'])])

    loaded_neural_network = keras.models.load_model(path_to_model_file)
    loaded_selector = joblib.load(path_to_selector_file)
    loaded_vectorizer = joblib.load(path_to_vectorizer_file)
    json_data = json.dumps(result)
    return JsonResponse(json_data, safe=False)


def vectorize_texts_for_test(test_texts):
    # Learn vocabulary from training texts and vectorize training texts.
    vect_test_texts = loaded_vectorizer.transform(test_texts).toarray()

    vect_test_texts = loaded_selector.transform(vect_test_texts).astype('float32')

    return vect_test_texts


def predict_with_loaded_model(request):
    global loaded_neural_network
    dir_name = request.POST.get('dir_name')
    test_model_zip = request.FILES['file']
    load_model_directory = os.path.join(MEDIA_ROOT, dir_name)
    texts_from_files = []
    predictions = []
    classes = ['Negative', 'Positive']

    default_storage.save(test_model_zip.name, test_model_zip)
    utils.file_extraction_manager(MEDIA_ROOT, test_model_zip.name, load_model_directory)
    path_to_extracted_files = os.path.join(load_model_directory, test_model_zip.name.split('.')[0])

    list_of_files = os.listdir(path_to_extracted_files)

    for file in list_of_files:
        file_full_path = os.path.join(path_to_extracted_files, file)
        f = open(file_full_path, "r")
        file_contents = f.read()
        texts_from_files.append(file_contents)

    test_texts = vectorize_texts_for_test(texts_from_files)

    predicts = (loaded_neural_network.predict(test_texts) > 0.5).astype("int32")

    for pred in predicts:
        if pred[0] == 1:
            predictions.append('Positive')
        else:
            predictions.append('Negative')

    file_dir_name = test_model_zip.name.split('.')[0]
    result = {'predictions': predictions, 'files_names': list_of_files, 'dir_files': file_dir_name}

    json_data = json.dumps(result)

    return JsonResponse(json_data, safe=False)


def download_test_texts(request, dir_name, texts_dir, text_name):
    dst_path = os.path.join(MEDIA_ROOT, dir_name)
    dir_file_path = os.path.join(dst_path, texts_dir)
    full_file_path = os.path.join(dir_file_path, text_name)
    if os.path.exists(full_file_path):
        with open(full_file_path, 'rb') as fh:
            response = HttpResponse(fh.read(), content_type="text/plain")
            response['Content-Disposition'] = 'inline; filename=' + os.path.basename(full_file_path)
            return response
    raise Http404

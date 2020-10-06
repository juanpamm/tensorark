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
import pandas
import seaborn
import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('Agg')

graph = tf.Graph()


def build_improc_nn_template(request, folder):
    contexto = {'folder': folder}
    return render(request, 'improc/build_improc_nn.html', contexto)


def upload_image_nn_template(request):
    return render(request, 'improc/upload_image_set_nn.html')


def load_model_template(request):
    return render(request, 'improc/load_model.html')


def improc_add_layers_to_network(model, nodes, activation_func):
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


def improc_build_neural_network(nlayers, nodes, act_functions, output_act_func):
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(utils.width, utils.height))
    ])

    # Construction of the hidden layers
    for i in range(nlayers):
        improc_add_layers_to_network(model, nodes[i], act_functions[i])

    # Construction of the output layer
    improc_add_layers_to_network(model, len(utils.class_names), output_act_func)

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def improc_train_neural_network_v2(layers, nodes, act_functions, epochs, output_act_func, dst_path):
    with graph.as_default():
        sess = tf.compat.v1.Session()
        path_for_converted_set = os.path.join(dst_path, 'converted_set')
        # Checkpoint for network
        checkpoint_path = os.path.join(dst_path, 'saved_model')
        if not os.path.exists(checkpoint_path):
            os.makedirs(checkpoint_path)

        # Loading of the train and testing images and labels
        (train_images, train_labels), (test_images, test_labels) = utils.load_data(path_for_converted_set)
        classes = utils.class_names

        train_images = train_images / 255.0
        test_images = test_images / 255.0

        # Construction, training and saving of the neural network
        model = improc_build_neural_network(layers, nodes, act_functions, output_act_func)
        model.fit(train_images, train_labels, epochs=epochs)
        utils.save_model_to_json(checkpoint_path, model)
        model.save(os.path.join(checkpoint_path, 'neural_network.h5'))
        test_loss, test_acc = model.evaluate(test_images, test_labels)
        print('Test accuracy:', test_acc)

        # Use the test set for prediction
        predictions = model.predict(test_images)
        predicts = model.predict_classes(test_images)

        # Build confusion matrix
        con_mat = tf.compat.v1.confusion_matrix(labels=test_labels, predictions=predicts)
        con_mat_val = con_mat.eval(session=sess)

        # Path to confusion matrix image
        con_matrix_img_name = 'conf_matrix.png'
        con_matrix_img_path = os.path.join(dst_path, con_matrix_img_name)
        working_dir = os.path.split(dst_path)[1]
        img_name_to_send = working_dir + '/' + con_matrix_img_name

        con_mat_df = pandas.DataFrame(con_mat_val, index=classes, columns=classes)
        plt.figure(figsize=(8, 8))
        seaborn.heatmap(con_mat_df, annot=True, cmap=plt.cm.Blues, fmt="d")
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.savefig(con_matrix_img_path, format='png', bbox_inches='tight')
        shutil.copy(con_matrix_img_path, checkpoint_path)
        utils.compress_model_folder(dst_path)

        '''
        plt.figure()
        plt.imshow(test_images[0])
        plt.colorbar()
        plt.grid(False)
        plt.show()

        plt.figure()
        plt.imshow(test_images[1])
        plt.colorbar()
        plt.grid(False)
        plt.show()

        plt.figure()
        plt.imshow(test_images[2])
        plt.colorbar()
        plt.grid(False)
        plt.show()
        '''
        print(predictions[0])
        print(np.argmax(predictions[0]))
        print(classes[int(np.argmax(predictions[0]))])
        print(predictions[1])
        print(np.argmax(predictions[1]))
        print(classes[int(np.argmax(predictions[1]))])
        print(predictions[2])
        print(np.argmax(predictions[2]))
        print(classes[int(np.argmax(predictions[2]))])

    return {"accuracy": test_acc, "predictions": predictions, "img_name": img_name_to_send,
            "first_predict": classes[int(np.argmax(predictions[0]))]}


def upload_image_set(request):
    file = request.FILES['file']
    action = request.POST.get('action')
    app = request.POST.get('app')
    default_storage.save(file.name, file)
    working_dir_name = utils.get_name_for_working_dir(MEDIA_ROOT, action, app)
    dst_path = os.path.join(MEDIA_ROOT, working_dir_name)
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)

    result = {
        'working_dir': working_dir_name,
        'file_name': file.name
    }
    json_data = json.dumps(result)

    return JsonResponse(json_data, safe=False)


def decompress_image_set(request):
    file_name = request.POST.get('file_name')
    working_dir = request.POST.get('working_dir')
    dst_path = os.path.join(MEDIA_ROOT, working_dir)
    # Extraction of the image set loaded by the user
    utils.file_extraction_manager(MEDIA_ROOT, file_name, dst_path)
    extracted_dir = os.listdir(dst_path)[0]
    result = {
        'extracted_dir': extracted_dir
    }
    json_data = json.dumps(result)

    return JsonResponse(json_data, safe=False)


def convert_image_set(request):
    extracted_dir = request.POST.get('extracted_dir')
    working_dir = request.POST.get('working_dir')
    dst_path = os.path.join(MEDIA_ROOT, working_dir)
    path_to_extracted_dir = os.path.join(dst_path, extracted_dir)
    converted_path = os.path.join(dst_path, 'converted_set')

    # Paths to training and testing sets
    path_to_training_set = os.path.join(path_to_extracted_dir, 'training')
    path_to_testing_set = os.path.join(path_to_extracted_dir, 'testing')

    # Image set conversion into MNIST format
    utils.convert_image_set([path_to_training_set, 'train'], converted_path)
    utils.convert_image_set([path_to_testing_set, 'test'], converted_path)

    # Gzip compress the files obtained in the conversion
    utils.gzip_all_files_in_dir(converted_path)

    # Set the names for the classes
    utils.set_name_classes(path_to_training_set)

    # Remove image_set folder
    shutil.rmtree(path_to_extracted_dir, ignore_errors=True)

    result = {'success_val': True}
    json_data = json.dumps(result)

    return JsonResponse(json_data, safe=False)


def download_saved_model(request, dir_name):
    dst_path = os.path.join(MEDIA_ROOT, dir_name)
    full_file_path = os.path.join(dst_path, 'nn_model.zip')
    if os.path.exists(full_file_path):
        with open(full_file_path, 'rb') as fh:
            response = HttpResponse(fh.read(), content_type="application/zip")
            response['Content-Disposition'] = 'inline; filename=' + os.path.basename(full_file_path)
            return response
    raise Http404


def execute_nn_training(request):
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
    results = improc_train_neural_network_v2(layers, nodes, activation_functions, epochs, output_act_func, dst_path)

    # Setting the information to be sent to the client
    acc_percentage = results.get("accuracy") * 100
    st_percentage = '{number:.{digits}f}'.format(number=acc_percentage, digits=2)
    training_result = {
        'net_accuracy': st_percentage,
        'prediction': results.get("first_predict"),
        'img_name': results.get("img_name")
    }

    json_data = json.dumps(training_result)
    return JsonResponse(json_data, safe=False)


# -------------------------------------- MODEL LOADING ------------------------------------

def load_model(request):
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
    path_to_conf_matrix = os.path.split(dst_path)[1] + '/conf_matrix.png'
    path_to_json_file = os.path.join(dst_path, 'json_nn.json')
    path_to_model_file = os.path.join(dst_path, 'neural_network.h5')

    f = open(path_to_json_file, "r")
    f_content = json.loads(f.read())['config']
    f.close()
    len_f_content = len(f_content)

    result = {
        'num_layers': len_f_content,
        'img_set_name': load_dir_name,
        'hidden_layers': [],
        'img_name': path_to_conf_matrix
    }

    for i in range(len_f_content):
        if i == 0:
            result['input_layer'] = [f_content[i]['config']['batch_input_shape'][1],
                                     f_content[i]['config']['batch_input_shape'][2]]
        elif i == (len_f_content - 1):
            result['output_layer'] = [f_content[i]['config']['units'],
                                      activation_funcs.get(f_content[i]['config']['activation'])]
        else:
            result['hidden_layers'].append([f_content[i]['config']['units'],
                                            activation_funcs.get(f_content[i]['config']['activation'])])

    loaded_neural_network = keras.models.load_model(path_to_model_file)
    # loaded_neural_network.summary()

    json_data = json.dumps(result)
    return JsonResponse(json_data, safe=False)

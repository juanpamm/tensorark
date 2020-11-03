from django.core.files.storage import default_storage
from django.shortcuts import render
from django.http import JsonResponse, HttpResponse, Http404
from tensorark.settings import MEDIA_ROOT
from utils import utils
from tensorflow import keras
from PIL import Image
import json
import os.path
import numpy as np
import tensorflow as tf
import shutil
import matplotlib
matplotlib.use('Agg')

graph = tf.Graph()
loaded_neural_network = keras.Sequential()
loaded_classes = []


def build_improc_nn_template(request, folder):
    context = {'folder': folder}
    return render(request, 'improc/build_improc_nn.html', context)


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
        working_dir = os.path.split(dst_path)[1]
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
        history = model.fit(train_images, train_labels, epochs=epochs, validation_data=(test_images, test_labels))
        utils.save_model_to_json(checkpoint_path, model)
        model.save(os.path.join(checkpoint_path, 'neural_network.h5'))
        test_loss, test_acc = model.evaluate(test_images, test_labels)

        # Use the test set for prediction
        predicts = np.argmax(model.predict(test_images), axis=-1)

        # Build confusion matrix
        con_mat = tf.compat.v1.confusion_matrix(labels=test_labels, predictions=predicts)
        con_mat_val = con_mat.eval(session=sess)

        # Path to confusion matrix image
        con_matrix_img_name = 'improc_conf_matrix.png'
        con_matrix_img_path = os.path.join(dst_path, con_matrix_img_name)
        img_name_to_send = working_dir + '/' + con_matrix_img_name

        utils.confusion_matrix_plotter(con_mat_val, classes, con_matrix_img_path, checkpoint_path)

        # Code to create plot images
        history_dict = history.history

        acc = history_dict['accuracy']
        val_acc = history_dict['val_accuracy']
        loss = history_dict['loss']
        val_loss = history_dict['val_loss']
        epochs = range(1, len(acc) + 1)

        # Path to loss plot
        loss_plot_img_name = 'improc_loss_plot.png'
        loss_plot_img_path = os.path.join(dst_path, loss_plot_img_name)
        loss_plot_path_to_send = working_dir + '/' + loss_plot_img_name

        # Path to accuracy plot
        acc_val_accu_img = "improc_accuracy_plot.png"
        accuracy_plot_img_path = os.path.join(dst_path, acc_val_accu_img)
        accuracy_plot_path_to_send = working_dir + '/' + acc_val_accu_img

        utils.loss_accuracy_plotter(epochs, loss, val_loss, acc, val_acc, checkpoint_path, loss_plot_img_path,
                                    accuracy_plot_img_path)

        utils.compress_model_folder(dst_path)

    return {"accuracy": test_acc, "img_name": img_name_to_send, "loss_plot_name": loss_plot_path_to_send,
            "accuracy_plot_name": accuracy_plot_path_to_send}


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
        'img_name': results.get("img_name"),
        'loss_plot': results.get("loss_plot_name"),
        'accuracy_plot': results.get("accuracy_plot_name")
    }

    json_data = json.dumps(training_result)
    return JsonResponse(json_data, safe=False)


# -------------------------------------- MODEL LOADING ------------------------------------

def load_model(request):
    global loaded_neural_network
    global loaded_classes
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
    path_to_conf_matrix = os.path.split(dst_path)[1] + '/improc_conf_matrix.png'
    path_to_loss_plot = os.path.split(dst_path)[1] + '/improc_loss_plot.png'
    path_to_acc_plot = os.path.split(dst_path)[1] + '/improc_accuracy_plot.png'
    path_to_json_file = os.path.join(dst_path, 'json_nn.json')
    path_to_model_file = os.path.join(dst_path, 'neural_network.h5')

    f = open(path_to_json_file, "r")
    f_content = json.loads(f.read())
    f.close()
    loaded_classes = f_content['classes']
    f_layers = f_content['config']['layers']
    len_f_layers = len(f_layers)

    result = {
        'num_layers': len_f_layers,
        'load_dir_name': load_dir_name,
        'hidden_layers': [],
        'conf_matrix': path_to_conf_matrix,
        'loss_plot': path_to_loss_plot,
        'accuracy_plot': path_to_acc_plot
    }

    for i in range(len_f_layers):
        if i == 0:
            result['input_layer'] = [f_layers[i]['config']['batch_input_shape'][1],
                                     f_layers[i]['config']['batch_input_shape'][2]]
        elif i == (len_f_layers - 1):
            result['output_layer'] = [f_layers[i]['config']['units'],
                                      activation_funcs.get(f_layers[i]['config']['activation'])]
        elif i != 1:
            result['hidden_layers'].append([f_layers[i]['config']['units'],
                                            activation_funcs.get(f_layers[i]['config']['activation'])])

    result['classes'] = loaded_classes
    loaded_neural_network = keras.models.load_model(path_to_model_file)

    json_data = json.dumps(result)
    return JsonResponse(json_data, safe=False)


def get_images_to_predict(path_to_images):
    # load the image
    list_of_images = os.listdir(path_to_images)
    list_of_converted_images = []
    for img in list_of_images:
        image = Image.open(os.path.join(path_to_images, img))
        image = image.convert("L")
        arr_image = np.array(image)
        new_arr_image = (np.expand_dims(arr_image, 0))
        new_arr_image = new_arr_image / 255.0
        list_of_converted_images.append(new_arr_image)

    return list_of_converted_images, list_of_images


def predict_with_loaded_model(request):
    dir_name = request.POST.get('dir_name')
    test_model_zip = request.FILES['file']
    load_model_directory = os.path.join(MEDIA_ROOT, dir_name)
    predictions = []

    default_storage.save(test_model_zip.name, test_model_zip)
    utils.file_extraction_manager(MEDIA_ROOT, test_model_zip.name, load_model_directory)
    path_to_extracted_files = os.path.join(load_model_directory, test_model_zip.name.split('.')[0])

    images_array, list_of_images = get_images_to_predict(path_to_extracted_files)

    global loaded_neural_network
    predicts = loaded_neural_network.predict(np.vstack(images_array))

    for i in range(0, len(predicts)):
        prediction = int(np.argmax(predicts[i]))
        predictions.append(loaded_classes[prediction])

    image_dir_name = test_model_zip.name.split('.')[0]
    result = {'predictions': predictions, 'images_names': list_of_images, 'dir_image': image_dir_name}

    json_data = json.dumps(result)

    return JsonResponse(json_data, safe=False)


def download_test_images(request, dir_name, image_dir, image_name):
    dst_path = os.path.join(MEDIA_ROOT, dir_name)
    dir_file_path = os.path.join(dst_path, image_dir)
    full_file_path = os.path.join(dir_file_path, image_name)
    print('Full file path: ', full_file_path)
    if os.path.exists(full_file_path):
        with open(full_file_path, 'rb') as fh:
            response = HttpResponse(fh.read(), content_type="image/*")
            response['Content-Disposition'] = 'inline; filename=' + os.path.basename(full_file_path)
            return response
    raise Http404

from django.core.files.storage import default_storage
from django.shortcuts import render
from django.http import JsonResponse, HttpResponse, Http404
from tensorark.settings import MEDIA_ROOT
from utils import utils
import json
import os.path
import numpy as np
import tensorflow as tf
import shutil
# import matplotlib.pyplot as plt
from tensorflow import keras

graph = tf.Graph()
dst_path = ""


def build_improc_nn_template(request):
    return render(request, 'improc/build_improc_nn.html')


def add_layers_to_network(model, nodes, activation_func):
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


def build_neural_network(nlayers, nodes, act_functions, output_act_func):
    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(utils.width, utils.height))
    ])

    # Construction of the hidden layers
    for i in range(nlayers):
        add_layers_to_network(model, nodes[i], act_functions[i])

    # Construction of the output layer
    add_layers_to_network(model, len(utils.class_names), output_act_func)

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    return model


def train_neural_network_v2(layers, nodes, act_functions, epochs, output_act_func):
    global dst_path
    with graph.as_default():
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
        model = build_neural_network(layers, nodes, act_functions, output_act_func)
        model.fit(train_images, train_labels, epochs=epochs)
        utils.save_model_to_json(checkpoint_path, model)
        model.save(os.path.join(checkpoint_path, 'neural_network.h5'))
        utils.compress_model_folder(dst_path)
        test_loss, test_acc = model.evaluate(test_images, test_labels)
        print('Test accuracy:', test_acc)

        # Use the test set for prediction
        predictions = model.predict(test_images)

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

    return {"accuracy": test_acc, "predictions": predictions, "first_predict": classes[int(np.argmax(predictions[0]))]}


def load_image_set(request):
    global dst_path
    file = request.FILES['file']
    default_storage.save(file.name, file)
    dst_path = os.path.join(MEDIA_ROOT, os.path.splitext(file.name)[0])
    converted_path = os.path.join(dst_path, 'converted_set')
    if not os.path.exists(dst_path):
        os.mkdir(dst_path)

    # Extraction of the image set loaded by the user
    utils.file_extraction_manager(MEDIA_ROOT, file, dst_path)
    extracted_dir = os.listdir(dst_path)[0]
    path_to_extracted_dir = os.path.join(dst_path, extracted_dir)

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

    result = {
        'upload_val': True
    }
    json_data = json.dumps(result)
    return JsonResponse(json_data, safe=False)


def download_saved_model(request):
    file_path = os.path.join('saved_model', 'neural_network.h5')
    full_file_path = os.path.join(dst_path, file_path)
    if os.path.exists(full_file_path):
        with open(full_file_path, 'rb') as fh:
            response = HttpResponse(fh.read(), content_type="application/x-hdf5")
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
    for i in range(len(nodes)):
        nodes[i] = int(nodes[i])

    # Execute function to train the neural network
    results = train_neural_network_v2(layers, nodes, activation_functions, epochs, output_act_func)

    # Setting the information to be sent to the client
    acc_percentage = results.get("accuracy") * 100
    st_percentage = '{number:.{digits}f}'.format(number=acc_percentage, digits=2)
    training_result = {
        'net_accuracy': st_percentage,
        'prediction': results.get("first_predict")
    }

    json_data = json.dumps(training_result)
    return JsonResponse(json_data, safe=False)

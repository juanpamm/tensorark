from django.core.files.storage import default_storage
from django.shortcuts import render
from django.http import JsonResponse
from tensorark.settings import MEDIA_ROOT
from utils import utils
import json
import os.path
import numpy as np
import tensorflow as tf
# import matplotlib.pyplot as plt
from tensorflow import keras
# from keras.datasets import fashion_mnist

graph = tf.Graph()


def index(request):
    return render(request, 'improc/index.html')


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


def build_neural_network(nlayers, nodes, act_functions):
    print('Height: ', utils.height)
    print('Width', utils.width)

    model = keras.Sequential([
        keras.layers.Flatten(input_shape=(utils.width, utils.height))
    ])

    for i in range(nlayers):
        if i == (nlayers - 1):
            add_layers_to_network(model, len(utils.class_names), 'softmax')
        else:
            add_layers_to_network(model, nodes[i], act_functions[i])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])

    return model


def train_neural_network_v2(layers, nodes, act_functions, epochs, dst_path):
    with graph.as_default():
        # fashion_mnist_set = fashion_mnist

        # (train_images, train_labels), (test_images, test_labels) = fashion_mnist_set.load_data()

        (train_images, train_labels), (test_images, test_labels) = utils.load_data(dst_path)

        # classes = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
        #               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

        classes = utils.class_names
        print(classes)

        train_images = train_images / 255.0
        test_images = test_images / 255.0

        model = build_neural_network(layers, nodes, act_functions)
        model.fit(train_images, train_labels, epochs=epochs)
        test_loss, test_acc = model.evaluate(test_images, test_labels)

        print('Test accuracy:', test_acc)

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


'''
def apply_sigmoid_to_net(data, hidden_lay_list, lay_list):
    num_layers = len(hidden_lay_list)

    for j in range(num_layers):
        if j == 0:
            l1 = tf.matmul(data, hidden_lay_list[0]['weights']) + hidden_lay_list[0]['biases']
            l1 = tf.nn.sigmoid(l1)
            lay_list.append(l1)
        elif j == (num_layers - 1):
            output = tf.matmul(lay_list[num_layers - 2], hidden_lay_list[num_layers - 1]['weights']) \
                + hidden_lay_list[num_layers - 1]['biases']
            lay_list.append(output)
        else:
            li = tf.matmul(lay_list[j - 1], hidden_lay_list[j]['weights']) + hidden_lay_list[j]['biases']
            li = tf.nn.sigmoid(li)
            lay_list.append(li)


def apply_relu_to_net(data, hidden_lay_list, lay_list):
    num_layers = len(hidden_lay_list)

    for j in range(num_layers):
        if j == 0:
            l1 = tf.matmul(data, hidden_lay_list[0]['weights']) + hidden_lay_list[0]['biases']
            l1 = tf.nn.relu(l1)
            lay_list.append(l1)
        elif j == (num_layers - 1):
            output = tf.matmul(lay_list[num_layers - 2], hidden_lay_list[num_layers - 1]['weights']) \
                + hidden_lay_list[num_layers - 1]['biases']
            lay_list.append(output)
        else:
            li = tf.matmul(lay_list[j - 1], hidden_lay_list[j]['weights']) + hidden_lay_list[j]['biases']
            li = tf.nn.relu(li)
            lay_list.append(li)


def apply_elu_to_net(data, hidden_lay_list, lay_list):
    num_layers = len(hidden_lay_list)

    for j in range(num_layers):
        if j == 0:
            l1 = tf.matmul(data, hidden_lay_list[0]['weights']) + hidden_lay_list[0]['biases']
            l1 = tf.nn.elu(l1)
            lay_list.append(l1)
        elif j == (num_layers - 1):
            output = tf.matmul(lay_list[num_layers - 2], hidden_lay_list[num_layers - 1]['weights']) + \
                     hidden_lay_list[num_layers - 1]['biases']
            lay_list.append(output)
        else:
            li = tf.matmul(lay_list[j - 1], hidden_lay_list[j]['weights']) + hidden_lay_list[j]['biases']
            li = tf.nn.elu(li)
            lay_list.append(li)


def apply_tanh_to_net(data, hidden_lay_list, lay_list):
    num_layers = len(hidden_lay_list)

    for j in range(num_layers):
        if j == 0:
            l1 = tf.matmul(data, hidden_lay_list[0]['weights']) + hidden_lay_list[0]['biases']
            l1 = tf.nn.tanh(l1)
            lay_list.append(l1)
        elif j == (num_layers - 1):
            output = tf.matmul(lay_list[num_layers - 2], hidden_lay_list[num_layers - 1]['weights']) + \
                     hidden_lay_list[num_layers - 1]['biases']
            lay_list.append(output)
        else:
            li = tf.matmul(lay_list[j - 1], hidden_lay_list[j]['weights']) + hidden_lay_list[j]['biases']
            li = tf.nn.tanh(li)
            lay_list.append(li)


def apply_act_func(act_func, lay_list, data, hidden_lay_list):
    switch = {
        'relu': apply_relu_to_net(data, hidden_lay_list, lay_list),
        'sigmoid': apply_sigmoid_to_net(data, hidden_lay_list, lay_list),
        'tanh': apply_tanh_to_net(data, hidden_lay_list, lay_list),
        'elu': apply_elu_to_net(data, hidden_lay_list, lay_list)
    }

    switch.get(act_func)


def neural_network_model(data, nodes_hl, num_layers, act_function):
    # (input_data * weights) + biases
    n_classes = 10
    hidden_layer_list = []
    layers_list = []

    for i in range(num_layers):
        if i == 0:
            first_layer = {'weights': tf.Variable(tf.random_normal([784, nodes_hl[0]])),
                           'biases': tf.Variable(tf.random_normal([nodes_hl[0]]))}
            hidden_layer_list.append(first_layer)
        else:
            hidden_layer = {'weights': tf.Variable(tf.random_normal([nodes_hl[i - 1], nodes_hl[i]])),
                            'biases': tf.Variable(tf.random_normal([nodes_hl[i]]))}
            hidden_layer_list.append(hidden_layer)

    output_layer = {'weights': tf.Variable(tf.random_normal([nodes_hl[num_layers - 1], n_classes])),
                    'biases': tf.Variable(tf.random_normal([n_classes]))}
    hidden_layer_list.append(output_layer)

    apply_act_func(act_function, layers_list, data, hidden_layer_list)

    return layers_list[num_layers]


def train_neural_network(nodes_hl, num_layers, num_epochs, act_function):
    mnist = input_data.read_data_sets("/tmp/data", one_hot=True)
    # height x width
    x = tf.placeholder('float', [None, 784])
    y = tf.placeholder('float')
    batch_size = 100

    prediction = neural_network_model(x, nodes_hl, num_layers, act_function)
    cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits_v2(logits=prediction, labels=y))
    optimizer = tf.train.AdamOptimizer().minimize(cost)

    hm_epochs = num_epochs

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples / batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict={x: epoch_x, y: epoch_y})
                epoch_loss += c
            print('Epoch ', (epoch + 1), 'completed out of ', hm_epochs, 'loss: ', epoch_loss)

        correct = tf.equal(tf.argmax(prediction, 1), tf.argmax(y, 1))
        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        return accuracy.eval({x: mnist.test.images, y: mnist.test.labels})
'''


def execute_nn_training(request):
    file = request.FILES['file']
    default_storage.save(file.name, file)
    layers = int(request.POST.get('layers'))
    nodes = json.loads(request.POST.get('nodes'))
    activation_functions = json.loads(request.POST.get('act_func'))
    epochs = int(request.POST.get('epochs'))
    dst_path = os.path.join(MEDIA_ROOT, 'converted_set')

    for i in range(len(nodes)):
        nodes[i] = int(nodes[i])

    utils.file_extraction_manager(MEDIA_ROOT, file)

    path_for_train_set = os.path.join(utils.get_last_modified_dir(MEDIA_ROOT), 'training')
    path_for_test_set = os.path.join(utils.get_last_modified_dir(MEDIA_ROOT), 'testing')

    utils.convert_image_set([path_for_train_set, 'train'], dst_path)
    utils.convert_image_set([path_for_test_set, 'test'], dst_path)

    utils.gzip_all_files_in_dir(dst_path)

    utils.set_name_classes(path_for_train_set)

    results = train_neural_network_v2(layers, nodes, activation_functions, epochs, dst_path)

    acc_percentage = results.get("accuracy") * 100
    st_percentage = '{number:.{digits}f}'.format(number=acc_percentage, digits=2)
    training_result = {
        'net_accuracy': st_percentage,
        'prediction': results.get("first_predict")
    }

    json_data = json.dumps(training_result)
    return JsonResponse(json_data, safe=False)

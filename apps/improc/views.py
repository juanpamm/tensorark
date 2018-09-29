from django.shortcuts import render
from django.http import JsonResponse
import json
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

# Create your views here.


def neural_network_model(data, nodes_hl, num_layers):
    # (input_data * weights) + biases
    n_classes = 10
    hidden_layer_list = []
    layers_list = []

    for i in range(num_layers):
        if i == 0:
            first_layer = {'weights': tf.Variable(tf.random_normal([784, nodes_hl])),
                           'biases': tf.Variable(tf.random_normal([nodes_hl]))}
            hidden_layer_list.append(first_layer)
        elif i == (num_layers - 1):
            output_layer = {'weights': tf.Variable(tf.random_normal([nodes_hl, n_classes])),
                            'biases': tf.Variable(tf.random_normal([n_classes]))}
            hidden_layer_list.append(output_layer)
        else:
            hidden_layer = {'weights': tf.Variable(tf.random_normal([nodes_hl, nodes_hl])),
                            'biases': tf.Variable(tf.random_normal([nodes_hl]))}
            hidden_layer_list.append(hidden_layer)

    for j in range(num_layers):
        if j == 0:
            l1 = tf.matmul(data, hidden_layer_list[0]['weights']) + hidden_layer_list[0]['biases']
            l1 = tf.nn.relu(l1)
            layers_list.append(l1)
        elif j == (num_layers - 1):
            output = tf.matmul(layers_list[num_layers - 2], hidden_layer_list[num_layers - 1]['weights']) + \
                     hidden_layer_list[num_layers - 1]['biases']
            layers_list.append(output)
        else:
            li = tf.matmul(layers_list[j - 1], hidden_layer_list[j]['weights']) + hidden_layer_list[j]['biases']
            li = tf.nn.relu(li)
            layers_list.append(li)

    return layers_list[num_layers - 1]


def train_neural_network(nodes_hl, num_layers, num_epochs):
    mnist = input_data.read_data_sets("/tmp/data", one_hot=True)
    # height x width
    x = tf.placeholder('float', [None, 784])
    y = tf.placeholder('float')
    batch_size = 100

    prediction = neural_network_model(x, nodes_hl, num_layers)
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


def index(request):
    return render(request, 'improc/index.html')


def execute_nn_training(request):
    layers = int(request.GET.get('layers'))
    nodes = int(request.GET.get('nodes'))
    epochs = int(request.GET.get('epochs'))

    accuracy = train_neural_network(nodes, layers, epochs)
    acc_percentage = accuracy * 100

    training_result = {
        'net_accuracy': str(acc_percentage)
    }

    json_data = json.dumps(training_result)
    return JsonResponse(json_data, safe=False)


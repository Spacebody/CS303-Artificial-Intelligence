import os
import pickle
import matplotlib
import numpy as np
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import math
from collections import OrderedDict
import copy
from decimal import *
from matplotlib import style

__author__ = "Yilin Zheng"
__version__ = "5.5.0"

# set error raise for debug
np.seterr(divide='raise', invalid='raise')

# Hyperparameters
epoch = 100
learning_rate = 0.1
n = 100
lambda_value = 0.0005
batch_size = 100
# kernal size
input_size = 784
kernal_1 = 300
kernal_2 = 100
output_size = 10

# Layer class
class Layer:
    
    def __init__(self, W, b):
        self.W = W
        self.b = b
    
    def W(self):
        return self.W

    def b(self):
        return self.b

# Activation class
class Activation:
    
    def __init__(self):
        pass

    def relu(self, x, drelu=False):
        if drelu:
            return np.greater(x, 0.0).astype(float)
        return np.maximum(x, 0.0)

    def softmax(self, x):
        return ((np.exp(x.T-np.max(x.T))) / np.sum(np.exp(x.T-np.max(x.T)), axis=0)).T

# Loss class      
class Loss:

    def __init__(self):
        pass
 
    def cross_entropy_loss(self, h, t): 
        return - np.sum(t * np.log(h) + (1 - t) * np.log(1 - h)) / float(h.shape[0])

# Simple neural network
class Simple_neural_network:

    def __init__(self, learning_rate, batch_size, lambda_value, hidden_activation_type, output_activation_type):
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.lambda_value = lambda_value
        self.layer_num = 0
        self.layers = OrderedDict()
        self.activation = Activation()
        self.hidden_activation_type = hidden_activation_type
        self.output_activation_type = output_activation_type
        self.loss = Loss()
        self.loss_record = []
        self.z = OrderedDict()
        self.a = OrderedDict()
        self.accuracy = []

    def get_learning_rate(self):
        return self.learning_rate

    def set_learning_rate(self, learning_rate):
        self.learning_rate = learning_rate

    def normal_initializer(self, neuron_size):
        return np.random.uniform(-np.sqrt(6. / float(
            np.sum(neuron_size))), np.sqrt(6. / float(np.sum(neuron_size))), neuron_size)

    def bias_initializer(self, neuron_size):
        return np.zeros(neuron_size).astype(float)

    def variables(self, neuron_size, neuron_type):
        if neuron_type == 'W':
            w = self.normal_initializer(neuron_size)
            return w 
        elif neuron_type == 'b':
            b = self.bias_initializer(neuron_size)
            return b           
        else:
            print("Invalid type")

    def add_layer(self, W, b, layer_type):
        if layer_type == 'hidden':
            self.layer_num += 1
            self.layers[self.layer_num] = Layer(W, b)
        elif layer_type == 'output':
            self.layer_num += 1
            self.layers[self.layer_num] = Layer(W, b)
        else: 
            print("Invalid type")

    def activate(self, x, activation_type, d=False):
        if activation_type == 'relu':
            if d:
                return self.activation.relu(x, drelu=True)
            return self.activation.relu(x)
        elif activation_type == 'softmax':
            return self.activation.softmax(x)
        else:
            print("Invalid type")

    def feed_forward(self, x):
        self.z[0] = self.a[0] = a = z = x[:]
        for i in range(1, self.layer_num):
            self.z[i] = z = np.dot(a, self.layers[i].W) + self.layers[i].b
            self.a[i] = a = self.activate(z, self.hidden_activation_type)
        z = np.dot(a, self.layers[self.layer_num].W) + self.layers[self.layer_num].b
        h = self.activate(z, self.output_activation_type)
        return h

    def cross_entropy_loss(self, x, t):
        h = self.feed_forward(x)
        loss = self.loss.cross_entropy_loss(h, t)
        self.loss_record.append(loss)
        return self.loss_record

    def test_accuracy(self, x, t):
        h = self.feed_forward(x)
        correct = np.sum((h.argmax(axis=1) == t.argmax(axis=1)).astype(int)).astype(float)
        accuracy = correct / float(h.shape[0])
        self.accuracy.append(accuracy)
        return accuracy

    def get_accuracy(self):
        return self.accuracy

    def get_loss(self):
        return self.loss_record

    def back_propagation(self, h, t):
        sigma = OrderedDict()
        sigma[self.layer_num] = h - t
        for i in reversed(range(1, self.layer_num)):
            sigma[i] = np.multiply(np.dot(sigma[i + 1], self.layers[i + 1].W.T), self.activate(self.z[i], self.hidden_activation_type, d=True))  
        for i in reversed(range(1, self.layer_num+1)):
            self.layers[i].W = (1 - self.learning_rate * self.lambda_value) * self.layers[i].W - self.learning_rate / float(self.batch_size) * np.dot(self.a[i - 1].T, sigma[i])
            self.layers[i].b -= self.learning_rate / float(self.batch_size) * np.sum(sigma[i])
           
    def train(self, train_x, train_t, test_x, test_t, iteration):
        h = self.feed_forward(train_x)
        self.back_propagation(h, train_t)
        

if __name__ == "__main__":
    (train_images, train_labels, test_images, test_labels) = pickle.load(
        open('./mnist_data/data.pkl', 'rb'))

    train_size = len(train_images)
    test_size = len(test_images)

    # Prompt of dataset size
    print("train images size: {}".format(len(train_images)))
    print("train labels size: {}".format(len(train_labels)))
    print("test images size: {}".format(len(test_images)))
    print("test labels size: {}".format(len(test_labels)))

    ### 1. Data preprocessing: normalize all pixels to [0,1) by dividing 256
    train_images = np.asarray([row / 255. for row in train_images])
    test_images = np.asarray([row / 255. for row in test_images])
    train_labels = np.eye(np.max(train_labels) + 1)[train_labels]
    test_labels = np.eye(np.max(test_labels) + 1)[test_labels]
    # uncomment the following line to set seed 
    np.random.seed(2017)
    snn = Simple_neural_network(learning_rate, batch_size, lambda_value, 'relu', 'softmax')
    
    # hiddne layer 1
    w1 = snn.variables([input_size, kernal_1], 'W')
    b1 = snn.variables([kernal_1, ], 'b')
    snn.add_layer(w1, b1, 'hidden')
    # hidden layer2
    w2 = snn.variables([kernal_1, kernal_2], 'W')
    b2 = snn.variables([kernal_2, ], 'b')
    snn.add_layer(w2, b2, 'hidden')
    # output layer
    w3 = snn.variables([kernal_2, output_size], 'W')
    b3 = snn.variables([output_size, ], 'b')
    snn.add_layer(w3, b3, 'output')

    # trainning
    iteration = n
    for i in range(epoch):
        # mask = np.random.choice(train_images.shape[0], batch_size)
        if i == 50:
            snn.set_learning_rate(learning_rate / 10.)
        for j in range(epoch):
            images_batch = train_images[j*batch_size:(j+1)*batch_size]
            labels_batch = train_labels[j*batch_size:(j+1)*batch_size]
            h = snn.train(images_batch, labels_batch, test_images, test_labels, iteration)
        snn.test_accuracy(test_images, test_labels)
        snn.cross_entropy_loss(test_images, test_labels)
        print('-------------------------------------------------')
        print('Epoch {0}'.format(i + 1))
        print('Accuracy: {0}'.format([float(Decimal("%.4f" % e)) for e in snn.get_accuracy()]))
        print('Loss: {0}'.format([float(Decimal("%.4f" % e)) for e in snn.get_loss()]))
    
    accuracy_record = snn.get_accuracy()
    loss_record = snn.get_loss()
    
    # plot
    style.use('fivethirtyeight')
    plt.figure(figsize=(12,10))
    # plot accuracy vs. epoch
    plt.subplot(211)
    plt.title("Accuracy - Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.plot([x1 for x1 in range(1, epoch + 1)], accuracy_record)
    # plot loss vs. epoch
    plt.subplot(212)
    plt.title("Loss - Epoch")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.plot([x2 for x2 in range(1, epoch + 1)], loss_record)

    plt.tight_layout()
    plt.savefig('result.pdf', dbi=300)
    plt.show()

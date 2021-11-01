"""
Using the Keras API with TF 2.3, we will be buiding a deep neural
network with three hidden layer.

Data-set: Fashion-MNIST
- 70,000 images (grayscale).
- 10 categories of articles of cloting.
- (28, 28, 1) shapes of images.
- 60,000 images for training dataset.
- 10,000 imafes for test dataset.
- Map of label and name classes:
    Label   | Description
    0       | T-shirt/top
    1       | Trouser
    2       | Pullover
    3       | Dress
    4       | Coat
    5       | Sandal
    6       | Shirt
    7       | Sneaker
    8       | Bag
    9       | Ankle boot
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import tensorflow as tf
from tensorflow import keras as ks

# Load dataset
dataset = ks.datasets.fashion_mnist.load_data()
(training_images, training_labels), (test_images, test_labels) = dataset

# Exploring dataset
print('Training images dataset shape: {}'.format(training_images.shape))
print('# Labels for training dataset: {}'.format(len(training_labels)))
print('Test images dataset shape: {}'.format(test_images.shape))
print('# Labels for test dataset: {}'.format(len(test_labels)))

# Normalizing images
training_images = training_images / 255.0
test_images = test_images / 255.0

# Building model
input_data_shape = (28, 28)
hidden_activation_function = 'relu'
output_activation_function = 'softmax'

dnn_model = ks.Sequential()
# Input layer
dnn_model.add(ks.layers.Flatten(input_shape=input_data_shape, name='input_layer'))
# Hidden layers
dnn_model.add(ks.layers.Dense(
    256,
    activation=hidden_activation_function,
    name='hidden_layer_1'
))

dnn_model.add(ks.layers.Dense(
    192,
    activation=hidden_activation_function,
    name='hidden_layer_2'
))

dnn_model.add(ks.layers.Dense(
    128,
    activation=hidden_activation_function,
    name='hidden_layer_3'
))

dnn_model.add(ks.layers.Dense(
    10,
    activation=output_activation_function,
    name='output_layer'
))
dnn_model.summary()

# Use an optimization function with the help of the compile method.
# Configure el model for training
dnn_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

dnn_model.fit(
    x=training_images,
    y=training_labels,
    epochs=10
)

# Training evaluation
training_loss, training_accuracy = dnn_model.evaluate(
    x=training_images,
    y=training_labels
)
print('Training loss {}'.format(round(float(training_loss), 2)))
print('Training data accuracy {}'.format(round(float(training_accuracy), 2)))

# Test evaluation
test_loss, test_accuracy = dnn_model.evaluate(
    x=test_images,
    y=test_labels
)
print('Test loss {}'.format(round(float(test_loss), 2)))
print('Test data accuracy {}'.format(round(float(test_accuracy), 2)))

"""
Training loss 0.23
Training data accuracy 0.91
--------------------------------------------------------------------------------
Test loss 0.35
Test data accuracy 0.88
"""

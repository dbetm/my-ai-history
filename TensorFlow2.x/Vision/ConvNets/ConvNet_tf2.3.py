"""
Using the Keras API with TF 2.3, we will be buiding a convolutional neural
network (conv layers, pooling layers)+ fully connected (deep neural network).

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


mnist_fashion = ks.datasets.fashion_mnist
dataset = mnist_fashion.load_data()
(training_images, training_labels), (test_images, test_labels) = dataset

# Exploring dataset
print('Training images dataset shape: {}'.format(training_images.shape))
print('# Labels for training dataset: {}'.format(len(training_labels)))
print('Test images dataset shape: {}'.format(test_images.shape))
print('# Labels for test dataset: {}'.format(len(test_labels)))

# Normalizing images
training_images = training_images / 255.0
test_images = test_images / 255.0

# Reshape datasets of images
training_images = training_images.reshape((60000, 28, 28, 1))
test_images = test_images.reshape((10000, 28, 28, 1))
print('-'*42)
print('Training Dataset Shape: {}'.format(training_images.shape))
print('No. of Training Dataset Labels: {}'.format(len(training_labels)))
print('Test Dataset Shape: {}'.format(test_images.shape))
print('No. of Test Dataset Labels: {}'.format(len(test_labels)))

# Model
cnn_model = ks.models.Sequential()

# First layer: Convolutional layer with ReLU activation function.
# Input tensor: 2D array (28x28 pixels).
# Kernels: 50 filters of shape 3x3 pixels.
# Activation function: ReLU activation.
cnn_model.add(
    ks.layers.Conv2D(
        filters=50,
        kernel_size=(3, 3),
        activation='relu',
        input_shape=(28, 28, 1),
        name='Conv2D_layer_1'
    )
)

# Second layer: Pooling layer
# Input tensor: Shape (50, 26, 26)
# Output tensor: Shape (50, 13, 13)
cnn_model.add(ks.layers.MaxPooling2D((2, 2), name='Maxpooling_2D'))

# Fully connected layer
# Input tensor: Shape 50x13x13 -> (8450)
# Output tensor: Shape (10) -> Probabilities for each class
cnn_model.add(ks.layers.Flatten(name='Flatten'))
cnn_model.add(ks.layers.Dense(units=50, activation='relu', name='hidden_layer_1'))
cnn_model.add(ks.layers.Dense(units=10, activation='softmax', name='output_layer'))

cnn_model.summary()

# Compile models
cnn_model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
cnn_model.fit(x=training_images, y=training_labels, epochs=10)

# Model evaluation

# Train
training_loss, training_accuracy = cnn_model.evaluate(
    x=training_images,
    y=training_labels,
    verbose=0
)
print('Training Accuracy {}'.format(round(float(training_accuracy), 2)))
print('Training Loss {}'.format(round(float(training_loss), 2)))

# Test
test_loss, test_accuracy = cnn_model.evaluate(
    x=test_images,
    y=test_labels,
    verbose=0
)
print('Test Accuracy {}'.format(round(float(test_accuracy), 2)))
print('Test Loss {}'.format(round(float(test_loss), 2)))

"""
Training Accuracy 0.97
Training Loss 0.09
-------------------------------
Test Accuracy 0.91
Test Loss 0.29
"""

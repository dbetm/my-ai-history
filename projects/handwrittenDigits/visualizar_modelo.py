#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 23:08:54 2019

@author: david
"""

from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras.optimizers import SGD


input_shape = [28, 28, 1]
num_classes = 10

### Constuir el modelo
model = Sequential()
# Agregar capa de CONV+ReLU
model.add(Conv2D(32, kernel_size=(3,3), activation='relu', input_shape=input_shape))
# Agregar otra capa de CONV+ReLU
model.add(Conv2D(64, kernel_size=(3,3), activation='relu'))
# Agregar capa de reducción de características
model.add(MaxPooling2D(pool_size=(2,2)))
# Agregar operador para regularizar (evitar overfitting)
model.add(Dropout(0.25))
# Aplanar la matriz
model.add(Flatten())
# Agregar capa FC con ReLU
model.add(Dense(128, activation='relu'))
# Agregar operador para regularizar (evitar overfitting)
model.add(Dropout(0.5))
# Agregar capa final FC con ReLU
model.add(Dense(num_classes, activation='softmax'))

# crear el objeto del modelo
model.compile(loss='categorical_crossentropy', 
              optimizer=SGD(0.01), metrics=['accuracy'])


ruta = "models/model_plot_mnist.png"

# Genarar la gráfica
plot_model(model, to_file=ruta, show_shapes=True, show_layer_names=True)
# Mostrar la grafica aquí
img = mpimg.imread(ruta)
plt.figure(figsize=(30,15))
imgplot = plt.imshow(img)

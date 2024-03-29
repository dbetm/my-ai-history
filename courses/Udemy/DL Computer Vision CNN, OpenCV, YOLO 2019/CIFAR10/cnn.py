#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
CNN simple para clasificar imágenes en 10 clases posibles:
    1) airplane
    2) auto
    3) bird
    4) cat
    5) deer
    6) dog
    7) frog
    8) horse
    9) ship
    10) truck
"""

from __future__ import print_function
import keras
from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D
from keras.models import load_model
import os

batch_size = 12
num_classes = 10
epochs = 1

# Cargar el dataset
(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# Mostrar las dimensiones de los datos
print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# Formatear los datos de entrenamiento
# Normalizar y cambiar el tipo de dato
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# Codificar las salidas
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

# Construir el modelo

model = Sequential()
# padding 'same' significa que agrega padding a la entrada de tal forma que 
# la salida tenga el mismo tamaño que la entrada original
model.add(Conv2D(32, (3,3), padding='same', input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(32, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(64, (3,3), padding='same', input_shape=x_train.shape[1:]))
model.add(Activation('relu'))
model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

# inicializar optimizador RMSprop y configurar algunos parámetros
opt = keras.optimizers.rmsprop(learning_rate=0.0001, rho=1e-6)

# Creamos el objeto
model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])

print(model.summary())

# Entrenar el modelo
history = model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,
                    validation_data=(x_test, y_test),
                    shuffle=True)
# Guardar el modelo
model.save("models/cifar10_simple_cnn_1epoch.h5")

# Evaluar el rendimiento del modelo entrenado
scores = model.evaluate(x_test, y_test, verbose=1)
print("Test loss:", scores[0])
print("Test accuracy", scores[1])




















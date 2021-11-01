#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 21 20:48:12 2019
Primera CNN con Keras
@author: david
"""

### 1) Cargar el dataset
import keras
from keras.datasets import mnist
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from keras import backend as K
from keras.optimizers import SGD
import numpy as np
import matplotlib.pyplot as plt


### 2A Examinar los datos
def examine(x_train, y_train, x_test, y_test):
    print("Dimensiones de x_train", str(x_train.shape))
    print("Número de instancias de entrenamiento", str(len(x_train)))
    print("Número de etiquetas del entrenamiento", str(len(y_train)))
    print("Número de instancias de prueba", str(len(x_test)))
    print("Número de etiquetas para la prueba", str(len(y_test)))
    print("Dimensiones de una imagen de entrenamiento", str(x_train[0].shape))
    print("Dimensiones de las imágenes de prueba", str(x_test[0].shape))

def take_look(x_train):
    # graficar 6 imágenes, subplot lleva 3 args: nfilas, ncols, index
    # ponemos el mapa de colores como 'grey' dado que las imágenes están a
    sup_lim = len(x_train)
    base = 330
    # escala de grises
    for i in range(1,7):
        plt.subplot(base + i)
        random_num = np.random.randint(0, sup_lim)
        plt.imshow(x_train[random_num], cmap=plt.get_cmap('gray'))
    # display our plots
    plt.show()
     

# cargarlo, se tienen las instancias de entrenamiento
# la otra tiene las instancias de prueba
(x_train, y_train), (x_test, y_test) = mnist.load_data()
# hay 60,000 instancias para entrenamiento
# x_train son las imágenes, y_train los labels correspondientes
# hay 10,000 instancias de prueba
print(x_test.shape)
examine(x_train, y_train, x_test, y_test)
# take_look(x_train)


# Poner en el formato correcto con el que trabaja Keras, nuestros datos
# núm de instancias, filas, columnas y profundidad (número de canales)

# guardar el número de filas y columnas
img_rows = x_train[0].shape[0]
img_cols = x_train[1].shape[0]

# Necesitamos agregar la 4ta dimensión (profundidad) a nuestros datos, cambiando de:
# (60000, 28, 28) a (60000, 28, 28, 1)
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

# guardamos la forma de una simple imagen
input_shape = (img_rows, img_cols, 1)

# cambiar el tipo de imagen a float32
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalizar los datos, al cambiar del rango (0-255) a (0-1)
x_train /= 255
x_test /= 255

print('\nForma de x_train', x_train.shape)
print(x_train.shape[0], "instancias de entrenamiento")
print(x_test.shape[0], "instancias de prueba")

### Codificar nuestra salidas (Y)
# y_test.shape -> (10000)
# Now we one hot our outputs, convertir vector de enteros a una matriz binaria
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
# y_test.shape -> (10000, 10)
num_classes = y_test.shape[1]
print("Número de clases: " + str(num_classes))

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

# Mostrar una tabla resumen de nuestro modelo
print(model.summary())

### Entrenamiento
batch_size = 32
epochs = 10

history = model.fit(x_train, y_train, batch_size = batch_size, epochs=epochs,
                    verbose=1, validation_data=(x_test, y_test))
# verbose = 1, para que muestra una barra animada del progreso de entrenamiento
score = model.evaluate(x_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])

### Graficando la pérdida en ambos datasets (validación y entrenamiento)
history_dict = history.history
loss_values = history_dict['loss'] # entrenamiento
val_loss_values = history_dict['val_loss'] # validación
epochs = range(1, len(loss_values) + 1)

line1 = plt.plot(epochs, val_loss_values, label="Test loss")
line2 = plt.plot(epochs, loss_values, label="Training loss")

plt.setp(line1, linewidth=2.0, marker='+', markersize=10.0)
plt.setp(line2, linewidth=2.0, marker='4', markersize=10.0)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.show()

### Graficando la eficacia en ambos datasets (validación y entrenamiento)
acc_values = history_dict['accuracy'] # entrenamiento
val_acc_values = history_dict['val_accuracy'] # validación
epochs = range(1, len(acc_values) + 1)

line1 = plt.plot(epochs, val_acc_values, label="Test accuracity")
line2 = plt.plot(epochs, acc_values, label="Training accuracity")

plt.setp(line1, linewidth=2.0, marker='+', markersize=10.0)
plt.setp(line2, linewidth=2.0, marker='4', markersize=10.0)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.grid(True)
plt.show()


## Guardando el modelo
model.save("models/mnist_simple_cnn_10epochs.h5")
print("Model saved")













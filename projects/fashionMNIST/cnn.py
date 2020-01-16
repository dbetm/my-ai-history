#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 15 17:54:01 2020

@author: david
"""

# Importamos la bibliotecas necesarias
import struct
import numpy as np
import matplotlib.pyplot as plt
# para el modelo usando keras
import keras
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras import backend as K
# para probar el modelo
import cv2

""" La siguiente función es usada para leer los datos y retornarlo como un
arreglo de numpy """
def read_idx(filename):
    """Credit: https://gist.github.com/tylerneylon"""
    with open(filename, 'rb') as f:
        zero, data_type, dims = struct.unpack('>HBB', f.read(4))
        shape = tuple(struct.unpack('>I', f.read(4))[0] for d in range(dims))
        return np.frombuffer(f.read(), dtype=np.uint8).reshape(shape)

# Usamos la función anterior para extraer el conjunto de entrenamiento
# y el de prueba.
x_train = read_idx("/home/david/datasets/fashion_mnist/train-images-idx3-ubyte")        
y_train = read_idx("/home/david/datasets/fashion_mnist/train-labels-idx1-ubyte")
x_test = read_idx("/home/david/datasets/fashion_mnist/t10k-images-idx3-ubyte")        
y_test = read_idx("/home/david/datasets/fashion_mnist/t10k-labels-idx1-ubyte")

# Vamos a inspeccionar el dataset

# Imprimir el número de instancias en x_train, x_test, y_train, y_test
print("Dimensiones de x_train", str(x_train.shape))
print("Total de instancias en los datos de entrenamiento: ", str(len(x_train)))
print("Total de etiquetas en el conjunto de entrenamiento: ", str(len(y_train)))
print("Total de instancias en los datos de prueba: ", str(len(x_test)))
print("Total de etiquetas en el conjunto de prueba: ", str(len(y_test)))
print()
print("Dimensiones de las imágenes (entrenamiento): ", str(x_train[0].shape))
print("Dimensiones de las imágenes (prueba): ", str(x_test[0].shape))

# Vamos a ver algunas imágenes (solo 6) usando matplotlib
# Los argumentos de subplot son nfila, ncolumna e index
# el mapa de color se define como gris, ya que las imágenes están en escala de gris
index = 1

for i in range(0, 7):    
    plt.subplot(330+index)
    index += 1
    random_num = np.random.randint(0, len(x_train)) # seleccionamos un índice al 'azar'
    plt.imshow(x_train[random_num], cmap=plt.get_cmap('gray'))

# Mostrar las gráficas
plt.show()

# Vamos a crear el modelo usando Keras

# Primero adaptamos la data
# Parámetros de entrenamiento
batch_size = 128
epochs = 3

# Guardamos las filas y columnas de las imágenes
img_rows = x_train[0].shape[0]
img_cols = x_train[1].shape[0]

# Agregamos la cuarta dimensión, que es el número de canales, y es el formato que
# requiere Keras
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, 1)
x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, 1)

# Guardar la forma de una simple imagen
input_shape = (img_rows, img_cols, 1)

# Cambiar el tipo de dato a float32
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

# Normalizar los datos al cambiar el rango de (0 a 255) a (0 a 1).
x_train /= 255
x_test /= 255

# Vemos la nueva forma de los datos de entrenamiento
print("x_train shape: ", x_train.shape)
print(x_train.shape[0], ' train samples')
print(x_test.shape[0], " test samples")

# Ahora tenemos una salida de codificación activa
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)

# Contamos el número de clases de nuestra 'hot encoded matrix'
print("Número de clases: " + str(y_test.shape[1]))
num_classes = y_test.shape[1]
num_pixels = x_train.shape[1] * x_train.shape[2]

# ahora sí se construye la arquitectura del modelo
model = Sequential()

model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(BatchNormalization())

model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(BatchNormalization())

model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(BatchNormalization())

model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss = 'categorical_crossentropy',
              optimizer = keras.optimizers.Adadelta(),
              metrics = ['accuracy'])
print(model.summary())


# Entrenar el modelo
history = model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))

score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

# Guardamos el modelo
model.save("/home/david/datasets/fashion_mnist/fashionMNIST.h5")

# Probamos el modelo

# Creamos un diccionario para asociar un id con la clase
dic = {0:"T-shirt/top", 1:"Trouser", 2:"Pullover", 3:"Dress", 4:"Coat", 5:"Sandal"}
dic[6] = "Shirt"
dic[7] = "Sneaker"
dic[8] = "Bag"
dic[9] = "Ankle boot"

# Función para mostrar la imagen, la predicción y lo real
def draw_test(name, pred, actual, input_im):
    BLACK = [0,0,0]
    res = dic[int(pred)]
    actual = dic[int(actual)]
    expanded_image = cv2.copyMakeBorder(
        input_im, 0, 0, 0, 4*imageL.shape[0], cv2.BORDER_CONSTANT, value=BLACK)
    expanded_image = cv2.cvtColor(expanded_image, cv2.COLOR_GRAY2BGR)
    cv2.putText(expanded_image, "Predicted -" + str(res), (152, 70), 
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 255, 0), 1)
    cv2.putText(expanded_image, " Actual - " + str(actual), (152, 90), 
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)
    cv2.imshow(name, expanded_image)

for i in range(0, 10):
    rand = np.random.randint(0, len(x_test))
    input_im = x_test[rand]
    actual = y_test[rand].argmax(axis=0)
    imageL = cv2.resize(input_im, None,  fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    input_im = input_im.reshape(1, 28, 28, 1)
    # Get prediction
    res = str(model.predict_classes(input_im, 1, verbose=1)[0])
    draw_test("Prediction", res, actual, imageL)
    cv2.waitKey(0)

cv2.destroyAllWindows()



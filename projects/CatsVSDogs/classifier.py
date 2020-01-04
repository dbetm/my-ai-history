#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec 27 19:56:56 2019
Clasificador de perros y gatos
Para demostrar la utilidad de: data augmentation
@author: david
"""

from __future__ import print_function
from os import listdir
from os.path import isfile, join
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
import cv2
import numpy as np
import sys
import os
import shutil

path = "/home/david/datasets/catsvsdogs/images/"
# obtenemos los nombres de los archivos
archivos = [f for f in listdir(path) if isfile(join(path, f))]

print(str(len(archivos)) + ' images loaded')

### Dividiendo el dataset
"""
1) Se necesita guardar sus etiquetas (ejemplo: y_train e y_test)
2) Se necesita redimensionar las imágenes a 150x150
3) Se usarán 1000 imágenes de perros y 1000 de gatos para el set de entrenamiento
4) Para el set de prueba de usarán 500 de cada clase.
5) Perros tendrán label de 1 y gatos de 0
6) Se guardarán nuevas imágenes en los sig. directorios:
    - /home/david/datasets/catsvsdogs/train/dogs/
    - /home/david/datasets/catsvsdogs/train/cats/
    - /home/david/datasets/catsvsdogs/test/dogs/
    - /home/david/datasets/catsvsdogs/test/cats/
"""

dog_count = 0
cat_count = 0
training_size = 1000
test_size = 500
training_images = []
training_labels = []
test_images = []
test_labels = []
size_img = 150
dog_dir_train = "/home/david/datasets/catsvsdogs/train/dogs/"
cat_dir_train = "/home/david/datasets/catsvsdogs/train/cats/"
dog_dir_test = "/home/david/datasets/catsvsdogs/test/dogs/"
cat_dir_test = "/home/david/datasets/catsvsdogs/test/cats/"


# regresa el número de ceros que se le concatenan al nombre
def getZeros(number):
    if(number > 10 and number < 100):
        return "0"
    elif(number < 10):
        return "00"
    else:
        return ""

for i, file in enumerate(archivos):
    image = cv2.imread(path+file)
    print(path+file)
    image = cv2.resize(image, (size_img, size_img), interpolation=cv2.INTER_AREA)
    if(archivos[i][0] == "d"):
        dog_count += 1
        if(dog_count <= training_size):
            training_images.append(image)
            training_labels.append(1)
            zeros = getZeros(dog_count)
            ruta = dog_dir_train +  "dog" + str(zeros) + str(dog_count) + ".jpg"
        elif(dog_count > training_size and dog_count <= training_size + test_size):
            test_images.append(image)
            test_labels.append(1)
            zeros = getZeros(dog_count - 1000)
            ruta = dog_dir_test + "dog" + str(zeros) + str(dog_count-1000) + ".jpg"
        cv2.imwrite(ruta, image)
    elif(archivos[i][0] == "c"):
        cat_count += 1
        if(cat_count <= training_size):
            training_images.append(image)
            training_labels.append(0)
            zeros = getZeros(cat_count)
            ruta = cat_dir_train +  "cat" + str(zeros) + str(cat_count) + ".jpg"
        elif(cat_count > training_size and cat_count <= training_size + test_size):
            test_images.append(image)
            test_labels.append(0)
            zeros = getZeros(cat_count - 1000)
            ruta = cat_dir_test + "cat" + str(zeros) + str(cat_count-1000) + ".jpg"
        cv2.imwrite(ruta, image)
    elif(dog_count == training_size+test_size and cat_count == training_size+test_size):
        break
print("Training and test data extraction complete")


# Let's save our dataset's to NPZ files
# using numpy's savez function to store our loaded data as NPZ files
np.savez('cats_vs_dogs_training_data.npz', np.array(training_images))
np.savez('cats_vs_dogs_training_labels.npz', np.array(training_labels))
np.savez('cats_vs_dogs_test_data.npz', np.array(test_images))
np.savez('cats_vs_dogs_test_labels.npz', np.array(test_labels))
# moví los archivos generados a otra ubicación, ya que están pesados

# Función para cargar los datos
def load_data_training_and_test(ruta, datasetname):
    npzfile = np.load(ruta + datasetname + "_training_data.npz")
    train = npzfile['arr_0']
    
    npzfile = np.load(ruta + datasetname + "_training_labels.npz")
    train_labels = npzfile['arr_0']
    
    npzfile = np.load(ruta + datasetname + "_test_data.npz")
    test = npzfile['arr_0']
    
    npzfile = np.load(ruta + datasetname + "_test_labels.npz")
    test_labels = npzfile['arr_0']
    
    return (train, train_labels), (test, test_labels)

# Cargar los datos ya en el formato que espera Keras
ruta = "/home/david/datasets/catsvsdogs/"
datasetname = "cats_vs_dogs"

(x_train, y_train), (x_test, y_test) = load_data_training_and_test(ruta, datasetname)
print(x_train.shape) # imágenes-entrenamiento
print(y_train.shape) # labels-entrenamiento
print(x_test.shape) # imágenes-prueba
print(y_test.shape) # labels-prueba
# Cambiando la forma de los datos (labels) de (2000) a (2000, 1) y de entrenamiento
# de (1000) a (1000, 1)
y_train = y_train.reshape(y_train.shape[0], 1)
y_test = y_test.reshape(y_test.shape[0], 1)

# Cambiar el tipo de imagen a float32
x_train = x_train.astype("float32")
x_test = x_test.astype("float32")

# Normalizar nuestros datos, al cambiar el rango (0 a 255) a (0 a 1)
x_train /= 255
x_test /= 255

print(x_train.shape) # imágenes-entrenamiento
print(y_train.shape) # labels-entrenamiento
print(x_test.shape) # imágenes-prueba
print(y_test.shape) # labels-prueba

### Creamos el modelo
batch_size = 16
epochs = 25

img_rows = x_train[0].shape[0]
img_cols = x_train[1].shape[0]
input_shape = (img_rows, img_cols, 3)

model = Sequential()
model.add(Conv2D(32, (3,3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(32, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Conv2D(64, (3,3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))


model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])
print(model.summary())

### Entrenamos el modelo
history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    validation_data=(x_test, y_test),
                    shuffle=True)

model.save("/home/david/datasets/catsvsdogs/cats_vs_dogs_v1.h5")

# Evaluar el rendimiento del modelo entrenado
scores = model.evaluate(x_test, y_test, verbose=1)
print("Test loss: ", scores[0])
print("Test accuracy: ", scores[1])


## Probamos con 10 imágenes

# Cargar modelo que no se entrenó con data augmentation
classifier = load_model("/home/david/datasets/catsvsdogs/cats_vs_dogs_v1.h5")

def draw_test(name, pred, input_im):
    BLACK = [0,0,0]
    if(pred == "[0]"):
        pred = "cat"
    elif(pred == "[1]"):
        pred = "dog"
    expanded_image = cv2.copyMakeBorder(input_im, 0, 0, 0, imageL.shape[0],
                                        cv2.BORDER_CONSTANT, value=BLACK)
    cv2.putText(expanded_image, str(pred), (253, 70), 
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 4, (0, 255, 0), 2)
    cv2.imshow(name, expanded_image)

# Seleccionar 10 imágenes al azar para clasificar
for i in range(0, 10):
    rand = np.random.randint(0, len(x_test))
    input_im = x_test[rand]
    
    imageL = cv2.resize(input_im, None, fx=2, fy=2, 
                        interpolation=cv2.INTER_CUBIC)
    cv2.imshow("Test image", imageL)
    
    input_im = input_im.reshape(1, 150, 150, 3)
    
    # Get prediction
    res = str(classifier.predict_classes(input_im, 1, verbose=1)[0])
    draw_test("Prediction", res, imageL)
    cv2.waitKey(0)

cv2.destroyAllWindows()






#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan  3 17:22:20 2020

@author: david
"""

import os
import numpy as np
from keras.models import Sequential, load_model
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.preprocessing.image import ImageDataGenerator
from keras import optimizers
import scipy
import pylab as pl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import cv2

input_shape = (150, 150, 3)
img_width = 150
img_height = 150

nb_train_samples = 2000
nb_validation_samples = 2000
batch_size = 16
epochs = 25

train_data_dir = "/home/david/datasets/catsvsdogs/train/"
validation_data_dir = "/home/david/datasets/catsvsdogs/test/"

# Creando nuestro generador de datos de nuestros datos de prueba
validation_datagen = ImageDataGenerator(
    # llevar el rango de píxeles de [0...255] a [0...1]
    rescale = 1./255
    )

# Creando el generador de datos para los datos de entrenamiento
train_datagen = ImageDataGenerator(
        rescale = 1./255,           # normalize pixel values to [0,1]
        rotation_range = 30,        # randomly applies rotations
        width_shift_range = 0.3,    # randomly applies width shifting
        height_shift_range = 0.3,   # randomly applies height shifting
        horizontal_flip = True,     # randonly flips the image
        # uses the fill mode nearest to fill gaps created by the above
        fill_mode = 'nearest'
    )

# Especificar criterio sobre los datos de entrenamiento, tales como:
# el directorio, tamaño de imagen, tamaño de lote y tipo
# Automáticamente recuperar las imágenes y sus clases para los sets de
# entrenamiento y validación
train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size = (img_width, img_height),
        batch_size = batch_size,
        class_mode = 'binary',
        shuffle = True
    )


validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size = (img_width, img_height),
        batch_size = batch_size,
        class_mode = 'binary',
        shuffle = True
    )


# Entrenamos el modelo
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


# Entrenamiento
history = model.fit_generator(
    train_generator,
    steps_per_epoch = nb_train_samples // batch_size,
    epochs = epochs,
    validation_data = validation_generator,
    validation_steps = nb_validation_samples // batch_size
    )

model.save("/home/david/datasets/catsvsdogs/cats_vs_dogs_v2.h5")


# Cargamos el dataset
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


### Probamos con 10 imágenes
# Cargar modelo que no se entrenó con data augmentation
classifier = load_model("/home/david/datasets/catsvsdogs/cats_vs_dogs_v2.h5")

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

input_im = cv2.imread("/home/david/datasets/catsvsdogs/perro.jpg")


imageL = cv2.resize(input_im, None, fx=2, fy=2, 
                        interpolation=cv2.INTER_CUBIC)
cv2.imshow("Test image", input_im)

input_im = input_im.reshape(1, 150, 150, 3)

# Get prediction
res = str(classifier.predict_classes(input_im, 1, verbose=1)[0])
draw_test("Prediction", res, imageL)
cv2.waitKey(0)

cv2.destroyAllWindows()



## Graficando la pérdida y eficacia
# Pérdida
history_dict = history.history

loss_values = history_dict["loss"]
val_loss_values = history_dict["val_loss"]
epochs = range(1, len(loss_values) + 1)

line1 = plt.plot(epochs, val_loss_values, label="Test loss")
line2 = plt.plot(epochs, loss_values, label="Training loss")
plt.setp(line1, linewidth=2.0, marker="+", markersize=10.0)
plt.setp(line2, linewidth=2.0, marker="4", markersize=10.0)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.grid(True)
plt.legend()
plt.show()

# Eficacia
history_dict = history.history
acc_values = history_dict["accuracy"]
val_acc_values = history_dict["val_accuracy"]
epochs = range(1, len(loss_values) + 1)

line1 = plt.plot(epochs, val_acc_values, label="Test acc")
line2 = plt.plot(epochs, acc_values, label="Training acc")
plt.setp(line1, linewidth=2.0, marker="+", markersize=10.0)
plt.setp(line2, linewidth=2.0, marker="4", markersize=10.0)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.grid(True)
plt.legend()
plt.show()





















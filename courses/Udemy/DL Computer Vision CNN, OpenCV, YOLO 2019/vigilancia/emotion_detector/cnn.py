#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 19 11:49:44 2020
Red neuronal convolucional LittleVGG para detectar emociones
@author: david
"""

from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization
from keras.layers import Conv2D, MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.layers.advanced_activations import ELU
from keras.layers.core import Activation, Flatten, Dropout, Dense
from keras.optimizers import RMSprop, SGD, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import os
import matplotlib.pyplot as plt
import sklearn
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
# Eficacia del 35% :(
num_classes = 7
img_rows, img_cols = 48, 48
batch_size = 16

train_dir = "/home/david/datasets/vigilancia/fer2013/train/"
validation_dir = "/home/david/datasets/vigilancia/fer2013/validation/"

# Aumento de datos

train_datagen = ImageDataGenerator(
        rescale = 1. / 255,
        rotation_range = 30,
        shear_range = 0.3,
        zoom_range = 0.3,
        width_shift_range = 0.4,
        height_shift_range = 0.4,
        horizontal_flip = True,
        fill_mode = 'nearest'
    )

validation_datagen = ImageDataGenerator(rescale = 1. / 255)

train_generator = train_datagen.flow_from_directory(
        train_dir,
        color_mode = 'grayscale',
        target_size = (img_rows, img_cols),
        batch_size = batch_size,
        class_mode = 'categorical',
        shuffle = True
    )

validation_generator = validation_datagen.flow_from_directory(
        validation_dir, 
        color_mode = 'grayscale',
        target_size = (img_rows, img_cols),
        batch_size = batch_size,
        class_mode = 'categorical',
        shuffle = True
    )

### Creando el modelo ###

model = Sequential()
model.add(Conv2D(32, (3,3), padding = 'same', kernel_initializer = 'he_normal',
                 input_shape = (img_rows, img_cols, 1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(32, (3,3), padding = 'same', kernel_initializer = "he_normal",
                 input_shape = (img_rows, img_cols, 1)))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size = (2,2)))
model.add(Dropout(0.2))

# 2do bloque de capas: Conv => ELU => CONV => ELU => POOL
# Notar que la inicialización de los pesos es 'he_normal', la cual es
# una distribución normal centrada en 0
model.add(Conv2D(64, (3, 3), padding="same", kernel_initializer="he_normal"))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), padding="same", kernel_initializer="he_normal"))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

# 3er bloque de capas: Conv => ELU => Conv=> ELU => Pool
model.add(Conv2D(128, (3, 3), padding="same", kernel_initializer="he_normal"))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), padding="same", kernel_initializer="he_normal"))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

# 4to bloque de capas: Conv => ELU = Conv => ELU => Pool
model.add(Conv2D(256, (3, 3), padding="same", kernel_initializer="he_normal"))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Conv2D(256, (3, 3), padding="same", kernel_initializer="he_normal"))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

# 5to bloque de capas: Primer set FC => capas con activación ELU
model.add(Flatten())
model.add(Dense(64, kernel_initializer="he_normal"))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# 6to bloque: Segundo set FC => capas con activación ELU
model.add(Dense(64, kernel_initializer="he_normal"))
model.add(Activation('elu'))
model.add(BatchNormalization())
model.add(Dropout(0.5))

# 7mo bloque: softmax classifier
model.add(Dense(num_classes, kernel_initializer="he_normal"))
model.add(Activation("softmax"))

print(model.summary())

### Entrenando el modelo ###
checkpoint = ModelCheckpoint(
        "/home/david/datasets/vigilancia/models/emotion_little_vgg_3.h5",
        monitor="val_loss",
        mode="min",
        save_best_only = True,
        verbose=1
    )

earlystop = EarlyStopping(
        monitor = 'val_loss', 
        min_delta = 0, 
        patience = 3,
        verbose = 1,
        restore_best_weights = True
    )

reduce_lr = ReduceLROnPlateau(
    monitor = 'val_loss', 
    factor = 0.2,
    patience = 3, 
    verbose = 1, 
    min_delta = 0.0001
)

# we put our call backs into a callback list
callbacks = [earlystop, checkpoint, reduce_lr]

# We use a very small learning rate 
model.compile(loss = 'categorical_crossentropy',
              optimizer = Adam(lr=0.001),
              metrics = ['accuracy'])

nb_train_samples = 28709
nb_validation_samples = 3589
epochs = 5

history = model.fit_generator(
        train_generator,
        steps_per_epoch = nb_train_samples // batch_size,
        epochs = epochs,
        callbacks = callbacks,
        validation_data = validation_generator,
        validation_steps = nb_validation_samples // batch_size
    )

##e VER EL REPORTE DE LA CLASIFICACIÓN ###

# We need to recreate our validation generator with shuffle = false
validation_generator = validation_datagen.flow_from_directory(
        validation_dir,
        color_mode = 'grayscale',
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

class_labels = validation_generator.class_indices
class_labels = {v: k for k, v in class_labels.items()}
classes = list(class_labels.values())

#Confution Matrix and Classification Report
Y_pred = model.predict_generator(
        validation_generator, nb_validation_samples // batch_size+1
    )
y_pred = np.argmax(Y_pred, axis=1)

print('Confusion Matrix')
print(confusion_matrix(validation_generator.classes, y_pred))
print('Classification Report')
target_names = list(class_labels.values())
print(classification_report(validation_generator.classes, y_pred, 
                            target_names=target_names))

plt.figure(figsize=(8,8))
cnf_matrix = confusion_matrix(validation_generator.classes, y_pred)

plt.imshow(cnf_matrix, interpolation='nearest')
plt.colorbar()
tick_marks = np.arange(len(classes))
_ = plt.xticks(tick_marks, classes, rotation=90)
_ = plt.yticks(tick_marks, classes)



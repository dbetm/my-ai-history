# -*- coding: utf-8 -*-
"""
Editor de Spyder

Clasificador:
    Objetivo: Usar los callbacks de Keras
"""

from __future__ import print_function
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, load_model
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.optimizers import SGD, RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import sklearn
from sklearn.metrics import classification_report, confusion_matrix
import numpy as np
import matplotlib.pyplot as plt
import os

num_classes = 81 # hay 81 tipos de frutas
img_rows, img_cols = 32, 32
batch_size = 16

### OBTENER LAS IMÁGENES

train_data_dir = "/home/david/datasets/fruits-360/train"
validation_data_dir = "/home/david/datasets/fruits-360/validation"

# Vamos hacer algo de data augmentation
train_datagen = ImageDataGenerator(
        rescale = 1./255,
        rotation_range = 30,
        width_shift_range = 0.3,
        height_shift_range = 0.3,
        horizontal_flip = True,
        fill_mode = "nearest"
    )

validation_datagen = ImageDataGenerator(rescale = 1./255)

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size = (img_rows, img_cols),
        batch_size = batch_size,
        class_mode = 'categorical',
        shuffle = True
    )

validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size = (img_rows, img_cols),
        batch_size = batch_size,
        class_mode = 'categorical',
        shuffle = True
    )


### CREAR EL MODELO
model = Sequential()

model.add(Conv2D(32, (3, 3), padding='same', input_shape=(img_rows, img_cols, 3)))
model.add(Activation('relu'))
model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Conv2D(32, (3, 3), padding='same'))
model.add(Activation('relu'))
model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes))
model.add(Activation('softmax'))

print(model.summary())

### Agregamos los callbac, guardamos y compilamos el modelo

checkpoint = ModelCheckpoint("/home/david/datasets/fruits-360/models/fruits_v1.h5",
                             monitor = 'val_loss',
                             mode = 'min',
                             save_best_only = True,
                             verbose = 1)

earlystop = EarlyStopping(
        monitor = 'val_loss',
        min_delta = 0,
        patience = 3,
        verbose = 1,
        restore_best_weights = True
    )

reduce_lr = ReduceLROnPlateau(
        monitor = "val_loss",
        factor = 0.2,
        patience = 3,
        verbose = 1,
        min_delta = 0.0001
    )

# Ponemos los callback en una lista
callbacks = [earlystop, checkpoint, reduce_lr]

# Usamos una pequeña tasa de aprendizaje
model.compile(
            loss = 'categorical_crossentropy',
            optimizer = RMSprop(lr = 0.001),
            metrics = ['accuracy']
        )

nb_train_samples = 41322
nb_validation_samples = 13877
epochs = 5

### ENTRENAR EL MODELO
history = model.fit_generator(
        train_generator,
        steps_per_epoch = nb_train_samples // batch_size,
        epochs = epochs,
        callbacks = callbacks,
        validation_data = validation_generator,
        validation_steps = nb_validation_samples // batch_size
    )


### Cargamos el modelo entrenado
model = load_model("/home/david/datasets/fruits-360/models/fruits_v1.h5")

img_row, img_height, img_depth = 32, 32, 3

class_labels = validation_generator.class_indices
class_labels = {v: k for k, v in class_labels.items()}
classes = list(class_labels.values())

nb_train_samples = 41322
nb_validation_samples = 13877

### MOSTRANDO LA MATRIZ DE CONFUSIÓN
"""
Lecciones aprendidas:
    Para generar la matriz de confusión debe hacerse luego de terminar el entrenamiento
    dado que el validation generator no se guarda con el modelo
"""


Y_pred = model.predict_generator(
        validation_generator,
        nb_validation_samples // batch_size + 1
    )
y_pred = np.argmax(Y_pred, axis=1)

print("Confusion Matrix")
print(confusion_matrix(validation_generator.classes, y_pred))
print("Classification report")
target_names = list(class_labels.values())
print(classification_report(validation_generator.classes, y_pred))

plt.figure(figsize = (20,20))
cnf_matrix = confusion_matrix(validation_generator.classes, y_pred)

plt.imshow(cnf_matrix, interpolation='nearest')
plt.colorbar()

tick_marks = np.arange(len(classes))
_ = plt.xticks(tick_marks, classes, rotation=90)
_ = plt.yticks(tick_marks, classes)









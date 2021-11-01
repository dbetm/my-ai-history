#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 16 22:04:04 2020
Red neuronal para clasificar monos en su respectiva raza dada su foto
Usando transferencia de conocimiento, usando MobileNet
@author: david
"""

# val_accuracy: 0.9219

# Congelar todas la capas, excepto top 4, así solo se va a entrenar top 4
from keras.applications import MobileNet
from keras.layers import Dense, Dropout, Activation, Flatten, GlobalAveragePooling2D
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.models import load_model
import os
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

# MobileNet fue diseñada para trabajar en imágenes de 224x224
img_rows, img_cols = 224, 224

# Cargamos el MobileNet sin las capas FC
mobile_net = MobileNet(
        weights='imagenet',
        include_top = False,
        input_shape = (img_rows, img_cols, 3)
    )

# Aquí congelamos las últimas 4 capas
# Por defecto las capas son entrenables
for layer in mobile_net.layers:
    layer.trainable = False

# Vamos a imprimir nuestras capas
for (i, layer) in enumerate(mobile_net.layers):
    print(str(i) + " " + layer.__class__.__name__, layer.trainable)    


# La función que retorne la cabecera FC (Full connected)
def addTopModelMobileNet(bottom_model, num_classes):
    top_model = bottom_model.output
    top_model = GlobalAveragePooling2D()(top_model)
    top_model = Dense(1024, activation='relu')(top_model)
    top_model = Dense(1024, activation='relu')(top_model)
    top_model = Dense(512, activation='relu')(top_model)
    top_model = Dense(num_classes, activation='softmax')(top_model)
    return top_model

# Agregando la cabecera FC al modelo de MobileNet

# Seteamos el número de clases
num_classes = 10

fc_head = addTopModelMobileNet(mobile_net, num_classes)
model = Model(inputs = mobile_net.input, outputs = fc_head)
print(model.summary())


# Cargar el dataset
train_data_dir = "/home/david/datasets/monkey_breed/train"
validation_data_dir = "/home/david/datasets/monkey_breed/validation"    
    
# Hacer algo de aumento de datos
train_datagen = ImageDataGenerator(
        rescale = 1. / 255,
        rotation_range = 45,
        width_shift_range=0.3,
        height_shift_range=0.3,
        horizontal_flip=True,
        fill_mode='nearest'
    )

validation_datagen = ImageDataGenerator(rescale=1./255)

# setear el tamaño del lote
batch_size = 32

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical'
    )

validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical'
    )


# Entrenar el modelo
checkpoint = ModelCheckpoint(
        "/home/david/datasets/monkey_breed/models/monkey_breed_mobileNet.h5",
        monitor = "val_loss",
        mode = "min",
        save_best_only = True,
        verbose = 1
    )

earlystop = EarlyStopping(
        monitor = 'val_loss',
        min_delta = 0,
        patience = 3,
        verbose = 1,
        restore_best_weights = True
    )

# Enlistar los callbacks
callbacks = [earlystop, checkpoint]

# Usamos una tasa de aprendizaje pequeña
model.compile(
        loss="categorical_crossentropy",
        optimizer = RMSprop(lr = 0.001),
        metrics = ['accuracy']
    )

# Número de instancias en el set de entrenamiento y el de prueba
nb_train_samples = 1097
nb_validation_samples = 272

# Solo se definen 5 épocas
epochs = 5
batch_size = 16

history = model.fit_generator(
        train_generator,
        steps_per_epoch = nb_train_samples // batch_size,
        epochs = epochs,
        callbacks = callbacks,
        validation_data = validation_generator,
        validation_steps = nb_validation_samples // batch_size
    )


# Cargamos el modelo
classifier = load_model("/home/david/datasets/monkey_breed/models/monkey_breed_mobileNet.h5")

monkey_breeds_dict = {"[0]": "mantled_howler ", 
                      "[1]": "patas_monkey",
                      "[2]": "bald_uakari",
                      "[3]": "japanese_macaque",
                      "[4]": "pygmy_marmoset ",
                      "[5]": "white_headed_capuchin",
                      "[6]": "silvery_marmoset",
                      "[7]": "common_squirrel_monkey",
                      "[8]": "black_headed_night_monkey",
                      "[9]": "nilgiri_langur"}

monkey_breeds_dict_n = {"n0": "mantled_howler ", 
                      "n1": "patas_monkey",
                      "n2": "bald_uakari",
                      "n3": "japanese_macaque",
                      "n4": "pygmy_marmoset ",
                      "n5": "white_headed_capuchin",
                      "n6": "silvery_marmoset",
                      "n7": "common_squirrel_monkey",
                      "n8": "black_headed_night_monkey",
                      "n9": "nilgiri_langur"}


def draw_test(name, pred, im):
    monkey = monkey_breeds_dict[str(pred)]
    BLACK = [0,0,0]
    expanded_image = cv2.copyMakeBorder(im, 80, 0, 0, 100 ,cv2.BORDER_CONSTANT,value=BLACK)
    cv2.putText(expanded_image, monkey, (20, 60) , cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 2)
    cv2.imshow(name, expanded_image)
    
def getRandomImage(path):
    """function loads a random images from a random folder in our test path """
    folders = list(filter(lambda x: os.path.isdir(os.path.join(path, x)), os.listdir(path)))
    random_directory = np.random.randint(0,len(folders))
    path_class = folders[random_directory]
    print("Class - " + monkey_breeds_dict_n[str(path_class)])
    file_path = path + path_class
    file_names = [f for f in listdir(file_path) if isfile(join(file_path, f))]
    random_file_index = np.random.randint(0,len(file_names))
    image_name = file_names[random_file_index]
    return cv2.imread(file_path+"/"+image_name)  

for i in range(0,10):
    input_im = getRandomImage("/home/david/datasets/monkey_breed/validation/")
    input_original = input_im.copy()
    input_original = cv2.resize(input_original, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_LINEAR)
    
    input_im = cv2.resize(input_im, (224, 224), interpolation = cv2.INTER_LINEAR)
    input_im = input_im / 255.
    input_im = input_im.reshape(1,224,224,3) 
    
    # Get Prediction
    res = np.argmax(classifier.predict(input_im, 1, verbose = 0), axis=1)
    
    # Show image with predicted class
    draw_test("Prediction", res, input_original) 
    cv2.waitKey(0)

cv2.destroyAllWindows()


#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 17 01:32:53 2020
CNN para clasificar 17 clases de flores
El entrenamiento es mediante transfer learning del modelo VGG16
@author: david
"""

# Eficacia baja: val_accuracy = 0.48

from keras.applications import VGG16
from keras.applications import VGG16
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.layers.normalization import BatchNormalization
from keras.models import Model
from keras.optimizers import RMSprop
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import RMSprop
from keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.models import load_model
import os
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join

# Dimensiones de las imágenes de entrada
img_cols = 64
img_rows = 64

# Cargar el modelo VGG16 sin la cabecera FC
vgg16 = VGG16(weights = 'imagenet',
                  include_top = False,
                  input_shape = (img_rows, img_cols, 3)
              )
# Congelamos las capas
for layer in vgg16.layers:
    layer.trainable = False
    
# Imprimimos las capas
for (i, layer) in enumerate(vgg16.layers):
    print(str(i) + " " + layer.__class__.__name__, layer.trainable)

# Ahora se crea el modelo
train_data_dir = "/home/david/datasets/17_flowers/train"
validation_data_dir = "/home/david/datasets/17_flowers/validation"

train_datagen = ImageDataGenerator(
        rescale = 1. / 255,
        rotation_range = 20,
        width_shift_range = 0.2,
        height_shift_range = 0.2,
        horizontal_flip = True,
        fill_mode = 'nearest'
    )


validation_datagen = ImageDataGenerator(rescale = 1./255)

train_batch_size = 16
val_batchsize = 10

train_generator = train_datagen.flow_from_directory(
        train_data_dir,
        target_size = (img_rows, img_cols),
        batch_size = train_batch_size,
        class_mode = 'categorical'
    )

validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        target_size = (img_rows, img_cols),
        batch_size = val_batchsize,
        class_mode = 'categorical',
        shuffle = False
    )

num_classes = 17


def addTopModel(bottom_model, num_classes, D=256):
    """creates the top or head of the model that will be 
    placed ontop of the bottom layers"""
    top_model = bottom_model.output
    top_model = Flatten(name = "flatten")(top_model)
    top_model = Dense(D, activation = "relu")(top_model)
    top_model = Dropout(0.3)(top_model)
    top_model = Dense(num_classes, activation = "softmax")(top_model)
    return top_model


fc_head = addTopModel(vgg16, num_classes)

model = Model(inputs = vgg16.input, outputs = fc_head)

print(model.summary())

# Definimos dos callbacks
checkpoint = ModelCheckpoint(
        "/home/david/datasets/17_flowers/models/flowers_vgg_64.h5",
        monitor = 'val_loss',
        mode = 'min',
        save_best_only = True,
        verbose = 1
    )

earlystop = EarlyStopping(
        monitor = 'val_loss', 
        min_delta = 0, 
        patience = 5,
        verbose = 1,
        restore_best_weights = True
    )

# para que se vaya reduciendo la tasa de aprendizaje
reduce_lr = ReduceLROnPlateau(
        monitor = 'val_loss',
        factor = 0.2,
        patience = 3,
        verbose = 1,
        min_delta = 0.00001
    )

callbacks = [earlystop, checkpoint, reduce_lr]

# Usamoa una tasa de aprendizaje pequeña
model.compile(
        loss = 'categorical_crossentropy',
        optimizer = RMSprop(lr = 0.0001),
        metrics = ['accuracy']
    )

nb_train_samples = 1190
nb_validation_samples = 170
epochs = 25
batch_size = 32

history = model.fit_generator(
        train_generator,
        steps_per_epoch = nb_train_samples // batch_size,
        epochs = epochs,
        callbacks = callbacks,
        validation_data = validation_generator,
        validation_steps = nb_validation_samples // batch_size
    )

# Cargamos el modelo
classifier = load_model("/home/david/datasets/17_flowers/models/flowers_vgg_64.h5")


dic = {"0":"bluebell", "1":"buttercup", "2":"colts_foot", "3":"cowslip"}
dic["4"] = "crocus"
dic["5"] = "daffodil"
dic["6"] = "daisy"
dic["7"] = "dandelion"
dic["8"] = "fritillary"
dic["9"] = "iris"
dic["10"] = "lily_valley"
dic["11"] = "pansy"
dic["12"] = "snowdrop"
dic["13"] = "sunflower"
dic["14"] = "tigerlily"
dic["15"] = "tulip"
dic["16"] = "windflower"


# Función para mostrar la imagen, la predicción y lo real
def draw_test(name, pred, im):
    monkey = dic[str(pred).replace("[", "").replace("]", "")]
    BLACK = [0,0,0]
    expanded_image = cv2.copyMakeBorder(im, 80, 0, 0, 100 ,cv2.BORDER_CONSTANT,value=BLACK)
    cv2.putText(expanded_image, monkey, (20, 60) , cv2.FONT_HERSHEY_SIMPLEX,1, (0,0,255), 2)
    cv2.imshow(name, expanded_image)
    
def getRandomImage(path):
    """function loads a random images from a random folder in our test path """
    folders = list(filter(lambda x: os.path.isdir(os.path.join(path, x)), os.listdir(path)))
    random_directory = np.random.randint(0,len(folders))
    path_class = folders[random_directory]
    print("Class - " + path_class)
    file_path = path + path_class
    file_names = [f for f in listdir(file_path) if isfile(join(file_path, f))]
    random_file_index = np.random.randint(0,len(file_names))
    image_name = file_names[random_file_index]
    return cv2.imread(file_path+"/"+image_name)  

for i in range(0,10):
    input_im = getRandomImage("/home/david/datasets/17_flowers/validation/")
    input_original = input_im.copy()
    input_original = cv2.resize(input_original, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_LINEAR)
    
    input_im = cv2.resize(input_im, (64, 64), interpolation = cv2.INTER_LINEAR)
    input_im = input_im / 255.
    input_im = input_im.reshape(1,64,64,3) 
    
    # Get Prediction
    res = np.argmax(classifier.predict(input_im, 1, verbose = 0), axis=1)
    
    # Show image with predicted class
    draw_test("Prediction", res, input_original) 
    cv2.waitKey(0)

cv2.destroyAllWindows()




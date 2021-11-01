#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 22:05:00 2019

@author: david
"""


from keras.models import load_model
from keras.datasets import mnist
import cv2
import numpy as np


clasificador = load_model("models/mnist_simple_cnn_10epochs.h5")

# Se hacen algunas pruebas, este modelo tiene:
# Test loss: 0.04367569178150734
# Test accuracy: 0.9854999780654907


def draw_test(name, pred, img):
    BLACK = [0,0,0]
    expanded_img = cv2.copyMakeBorder(img, 0, 0, 0, 
                  imageL.shape[0], cv2.BORDER_CONSTANT, value=BLACK)
    expanded_img = cv2.cvtColor(expanded_img, cv2.COLOR_GRAY2BGR)
    cv2.putText(expanded_img, str(pred), (152, 70), 
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 4, (255, 0, 0))
    cv2.imshow(name, expanded_img)
    
# Cargamos los datos
(_, _), (x_test, _) = mnist.load_data()    
    
num_instances = 10000 
for i in range(0, 10):
    rand = np.random.randint(0, len(x_test))
    image = x_test[rand]
    
    imageL = cv2.resize(image, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)
    image = image.reshape(1, 28, 28, 1)
    # Obtener la predicción
    res = str(clasificador.predict_classes(image, 1, verbose=1)[0])
    draw_test("Prediction", res, imageL)
    cv2.waitKey(0)
cv2.destroyAllWindows()
    
    
    
    
    
    
    

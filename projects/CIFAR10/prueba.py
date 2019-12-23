#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Dec 23 00:38:34 2019

@author: david
"""
from keras.models import load_model
from keras.datasets import cifar10
import cv2
import numpy as np

(_, _), (x_test, _) = cifar10.load_data()


img_row, img_height, img_depth = 32, 32, 3
clasificador = load_model("models/cifar10_simple_cnn_1epoch.h5")
color = True
scale = 8

def draw_test(name, res, input_img, scale, img_row, img_height):
    BLACK = [0,0,0]
    res = int(res)
    
    if(res == 0):
        pred = "avion"
    elif(res == 1):
        pred = "auto"
    elif(res == 2):
        pred = "pajaro"
    elif(res == 3):
        pred = "gato"
    elif(res == 4):
        pred = "oveja"
    elif(res == 5):
        pred = "perro"
    elif(res == 6):
        pred = "rana"
    elif(res == 7):
        pred = "caballo"
    elif(res == 8):
        pred = "barco"
    else:
        pred = "troca"
        
    expanded_image = cv2.copyMakeBorder(input_img, 0, 0, 0, imageL.shape[0]*2,
                                        cv2.BORDER_CONSTANT, value=BLACK)
    if(color == False):
        expanded_image = cv2.cvtColor(expanded_image, cv2.COLOR_GRAY2BGR)
    cv2.putText(expanded_image, str(pred), (300, 80), 
                cv2.FONT_HERSHEY_COMPLEX_SMALL, 3, (255, 0, 0), 2)
    cv2.imshow(name, expanded_image)


for i in range(0, 10):
    rand = np.random.randint(0, len(x_test))
    img = x_test[rand]
    imageL = cv2.resize(img, None, fx=scale, fy=scale, 
                        interpolation=cv2.INTER_CUBIC)
    img = img.reshape(1, img_row, img_height, img_depth)
    
    # Obtener predicción
    res = str(clasificador.predict_classes(img, 1, verbose=1)[0])
    draw_test("Prediction", res, imageL, scale, img_row, img_height)
    cv2.waitKey(0)
    
cv2.destroyAllWindows()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jan 20 00:27:30 2020

@author: david
"""

from keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.preprocessing.image import img_to_array
import numpy as np
import os
import cv2
import numpy as np
from os import listdir
from os.path import isfile, join
import re

classifier = load_model(
    '/home/david/datasets/vigilancia/models/emotion_little_vgg_3.h5')

validation_data_dir = "/home/david/datasets/vigilancia/fer2013/validation/"
img_rows, img_cols = 48, 48
batch_size = 16

validation_datagen = ImageDataGenerator(rescale = 1. / 255)

validation_generator = validation_datagen.flow_from_directory(
        validation_data_dir,
        color_mode = 'grayscale',
        target_size=(img_rows, img_cols),
        batch_size=batch_size,
        class_mode='categorical',
        shuffle=False)

class_labels = validation_generator.class_indices
class_labels = {v: k for k, v in class_labels.items()}
classes = list(class_labels.values())
print(class_labels)

# Usamos un filtro en cascada de CV2
face_classifier = cv2.CascadeClassifier(
        '/home/david/datasets/vigilancia/haarcascade_frontalface_default.xml'
    )

# Probamos con una simple imagen
def face_detector(img):
    # Convert image to grayscale
    gray = cv2.cvtColor(img.copy(),cv2.COLOR_BGR2GRAY)
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    if faces is ():
        return (0,0,0,0), np.zeros((48,48), np.uint8), img
    
    allfaces = []   
    rects = []
    for (x,y,w,h) in faces:
        cv2.rectangle(img,(x,y),(x+w,y+h),(255,0,0),2)
        roi_gray = gray[y:y+h, x:x+w]
        roi_gray = cv2.resize(
                roi_gray, (48, 48), 
                interpolation = cv2.INTER_AREA
            )
        allfaces.append(roi_gray)
        rects.append((x,w,y,h))
    return rects, allfaces, img


img = cv2.imread("elon.jpeg")
rects, faces, image = face_detector(img)

i = 0
for face in faces:
    roi = face.astype("float") / 255.0
    roi = img_to_array(roi)
    roi = np.expand_dims(roi, axis=0)

    # make a prediction on the ROI, then lookup the class
    preds = classifier.predict(roi)[0]
    label = class_labels[preds.argmax()]   

    #Overlay our detected emotion on our pic
    label_position = (rects[i][0] + int((rects[i][1]/2)), abs(rects[i][2] - 10))
    i =+ 1
    cv2.putText(
        image, label, label_position , cv2.FONT_HERSHEY_SIMPLEX,1, (0,255,0), 2)
    
cv2.imshow("Emotion Detector", image)
cv2.waitKey(0)

cv2.destroyAllWindows()


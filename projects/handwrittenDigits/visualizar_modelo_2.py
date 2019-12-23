#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 22 23:08:54 2019

@author: david
"""

from keras.utils.vis_utils import plot_model
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from keras.models import load_model


model  = load_model("models/mnist_simple_cnn_10epochs.h5")


ruta = "models/model_plot_mnist2.png"

# Genarar la gráfica
plot_model(model, to_file=ruta, show_shapes=True, show_layer_names=True)
# Mostrar la grafica aquí
img = mpimg.imread(ruta)
plt.figure(figsize=(30,15))
imgplot = plt.imshow(img)

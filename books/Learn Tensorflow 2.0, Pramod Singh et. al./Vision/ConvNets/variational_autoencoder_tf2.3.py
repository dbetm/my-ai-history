""" Build a VAE model that will help us to generate new handwritten digits,
using the MNIST data-set.

This data set comprises 70,000 images (black-and-white) of handwritten digits,
ranging from 0 to 9, out of which 60,000 are for training,
and the remaining 10,000 are for testing.
"""
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

import os
import time
import numpy as np
import glob
import matplotlib.pyplot as plt
import PIL
import imageio

from IPython import display

(train_data, _), (test_data, _) = tf.keras.datasets.mnist.load_data()

(train_data, _), (test_data, _) = tf.keras.datasets.mnist.load_data()
train_data = train_data.reshape(train_data.shape[0], 28, 28, 1).astype('float32')
test_data = test_data.reshape(test_data.shape[0], 28, 28, 1).astype('float32')
# Input Image Normalization to the range of [0,1]
train_data /= 255.
test_data /= 255.
# Binarization of the Normalized Output
train_data[train_data >= .5] = 1.
train_data[train_data < .5] = 0.
test_data[test_data >= .5] = 1.
test_data[test_data < .5] = 0.

TRAIN_SIZE = 60000
BATCH_SIZE = 50
TEST_SIZE = 10000
train_batch = tf.data.Dataset.from_tensor_slices(train_data).shuffle(TRAIN_SIZE).batch(BATCH_SIZE)
test_batch = tf.data.Dataset.from_tensor_slices(test_data).shuffle(TEST_SIZE).batch(BATCH_SIZE)

class CONV_VAE(tf.keras.Model):
    def __init__(self, latent_dim):
        super(CONV_VAE, self).__init__()
        self.latent_vec = latent_vec
        self.encoder_model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(28, 28, 1)),
            tf.keras.layers.Conv2D(
                filters=25, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Conv2D(
                filters=50, kernel_size=3, strides=(2, 2), activation='relu'),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(latent_vec + latent_vec),
         ])

        self.decoder_model = tf.keras.Sequential([
            tf.keras.layers.InputLayer(input_shape=(latent_vec,)),
            tf.keras.layers.Dense(units=7*7*25, activation=tf.nn.relu),
            tf.keras.layers.Reshape(target_shape=(7, 7, 25)),
            tf.keras.layers.Conv2DTranspose(
                filters=50,
                kernel_size=3,
                strides=(2, 2),
                padding="SAME",
                activation='relu'),
            tf.keras.layers.Conv2DTranspose(
                filters=25,
                kernel_size=3,
                strides=(2, 2),
                padding="SAME",
                activation='relu'),
            tf.keras.layers.Conv2DTranspose(
                filters=1, kernel_size=3, strides=(1, 1), padding="SAME"),
        ])

    @tf.function
    def sampling(self, sam=None):
        if sam is None:
            sam = tf.random.normal(shape=(50, self.latent_vec))
        return self.decoder(sam, apply_sigmoid=True)

    def encoder(self, inp):
        mean, logd = tf.split(self.encoder_model(inp), num_or_size_splits=2, axis=1)
        return mean, logd

    def reparameterization(self, mean, logd):
        sam = tf.random.normal(shape=mean.shape)
        return sam * tf.exp(logd * .5) + mean

    def decoder(self, out, apply_sigmoid=False):
        logout = self.decoder_model(out)
        if apply_sigmoid:
          probabs = tf.sigmoid(logout)
          return probabs

        return logout


optimizer_func = tf.keras.optimizers.Adam(1e-4)

def log_normal_prob_dist_func(sample, mean, logd, raxis=1):
    log2pi = tf.math.log(2. * np.pi)
    return tf.reduce_sum(-.5 * ((sample - mean) ** 2. * tf.exp(-logd) + logd + log2pi), axis=raxis)

@tf.function
def loss_func(model, inp):
    mean, logd = model.encoder(inp)
    out = model.reparameterization(mean, logd)
    log_inp = model.decoder(out)
    cross_entropy = tf.nn.sigmoid_cross_entropy_with_logits(logits=log_inp, labels=inp)
    logp_inp_out = -tf.reduce_sum(cross_entropy, axis=[1, 2, 3])
    logp_out = log_normal_prob_dist_func(out, 0., 0.)
    logq_out_inp = log_normal_prob_dist_func(out, mean, logd)
    return -tf.reduce_mean(logp_inp_out + logp_out - logq_out_inp)

@tf.function
def gradient_func(model, inp, optimizer_func):
    with tf.GradientTape() as tape:
        loss = loss_func(model, inp)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer_func.apply_gradients(zip(gradients, model.trainable_variables))


epochs = 100
latent_vec = 8
examples = 8

rand_vec = tf.random.normal(
    shape=[examples, latent_vec])
model = CONV_VAE(latent_vec)

def generate_and_save_images(model, epochs, input_data):
    preds = model.sampling(input_data)
    fig = plt.figure(figsize=(4,4))

    for i in range(preds.shape[0]):
        plt.subplot(4, 4, i+1)
        plt.imshow(preds[i, :, :, 0], cmap='gray')
        plt.axis('off')

    plt.savefig('generated_digits/img_at_epoch{:04d}.png'.format(epochs))
    plt.show()

generate_and_save_images(model, 0, rand_vec)

for epoch in range(1, epochs + 1):
    start_time = time.time()
    for x in train_batch:
        gradient_func(model, x, optimizer_func)
    end_time = time.time()

    loss = tf.keras.metrics.Mean()
    for y in test_batch:
      loss(loss_func(model, y))
    elbo = -loss.result()
    display.clear_output(wait=False)
    print('Epoch no.: {}, Test batch ELBO: {}, '
          'elapsed time for current epoch {}'.format(epoch, elbo, end_time - start_time))
    generate_and_save_images(model, epochs, rand_vec)

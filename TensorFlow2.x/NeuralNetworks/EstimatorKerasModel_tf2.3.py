# Custom estimator - TensorFlow 2.3
# Neural Network for iris dataset

from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras as ks
import tensorflow_datasets as tf_ds


def data_input():
    train_test_split = tf_ds.Split.TRAIN
    iris_dataset = tf_ds.load('iris', split=train_test_split, as_supervised=True)
    iris_dataset = iris_dataset.map(
        lambda features, labels : ({'dense_input': features}, labels)
    )
    iris_dataset = iris_dataset.batch(32).repeat()

    return iris_dataset

# Build a simple Keras model
keras_model = ks.models.Sequential([
    ks.layers.Dense(units=16, activation='relu', input_shape=(4,)),
    ks.layers.Dropout(0.2),
    ks.layers.Dense(1, activation='sigmoid')
])

# Compile model
keras_model.compile(
    loss='categorical_crossentropy',
    optimizer='adam'
)

keras_model.summary()
"""
Total params: 97
Trainable params: 97
Non-trainable params: 0
"""

# Build the estimator
model_path = 'keras_estimator/'
estimator_keras_model = ks.estimator.model_to_estimator(
    keras_model=keras_model,
    model_dir=model_path
)

# Train and evaluate the model
estimator_keras_model.train(input_fn=data_input, steps=25)
evaluation_result = estimator_keras_model.evaluate(
    input_fn=data_input,
    steps=10
)
print('Final evaluation result: {}'.format(evaluation_result))

"""
Final evaluation result: {'loss': 100.93359, 'global_step': 25}
"""

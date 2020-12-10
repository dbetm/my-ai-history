"""
Gradient boosting:
The main difference between gradient boosting compared to other boosting methods
is that instead of incrementing the weights of misclassified outcomes from
one previous learner to the next, we optimize the loss function  of the
previous learner.

Implement the gradient boosting method (ensemble technique) with TensorFlow 2.3,
using the iris data set and the BoostedTreesClassifier.
"""

from __future__ import absolute_import, division, print_function, unicode_literals

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
import tensorflow as tf
from tensorflow import keras
from tensorflow.estimator import BoostedTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score

# Load and configure the iris dataset.
col_names = ['SepalLength', 'SepalWidth', 'PetalLength', 'PetalWidth', 'Species']
target_dimensions = ['Setosa', 'Versicolor', 'Virginica']

training_data_path = tf.keras.utils.get_file(
    "iris_training.csv",
    "https://storage.googleapis.com/download.tensorflow.org/data/iris_training.csv"
)

test_data_path = tf.keras.utils.get_file(
    "iris_test.csv",
    "https://storage.googleapis.com/download.tensorflow.org/data/iris_test.csv"
)

training = pd.read_csv(training_data_path, names=col_names, header=0)
print(training)
training = training[training['Species'] >= 1]
training['Species'] = training['Species'].replace([1,2], [0,1])

test = pd.read_csv(test_data_path, names=col_names, header=0)
test = test[test['Species'] >= 1]
test['Species'] = test['Species'].replace([1,2], [0,1])
# drop: Boolean value, Adds the replaced index column to the data if False.
# inplace: Boolean value, make changes in the original data frame itself if True.
training.reset_index(drop=True, inplace=True)
test.reset_index(drop=True, inplace=True)

iris_dataset = pd.concat([training, test], axis=0)
print(iris_dataset.describe())


# Check the relation between the variables, using pairplot and a correlation graph
# sb.pairplot(iris_dataset, diag_kind='kde')
# plt.show()

correlation_data = iris_dataset.corr()
correlation_data.style.background_gradient(cmap='coolwarm', axis=None)
sb.heatmap(
    correlation_data,
    xticklabels=correlation_data.columns.values,
    yticklabels=correlation_data.columns.values
)
# plt.show()

# Descriptive statistics
stats = iris_dataset.describe()
iris_stats = stats.transpose()
print(iris_stats)

# Select the required columns
X_data = iris_dataset[[i for i in iris_dataset.columns if i not in ['Species']]]
Y_data = iris_dataset[['Species']]

# Print shapes of datasets (test and train)
features = train_test_split(X_data , Y_data , test_size=0.2)
training_features, test_features, training_labels, test_labels = features

print('#rows in Training Features: ', training_features.shape[0])
print('#rows in Test Features: ', test_features.shape[0])

# Normalizar los datos
def norm(x):
    stats = x.describe()
    stats = stats.transpose()

    return (x - stats['mean']) / stats['std']

normed_train_features = norm(training_features)
normed_test_features = norm(test_features)

# Build the input pipeline for the TensorFlow model
def feed_input(features_dataframe, target_dataframe, num_of_epochs=10,
    shuffle=True, batch_size=32):

    def input_feed_function():
        dataset = tf.data.Dataset.from_tensor_slices(
            (dict(features_dataframe), target_dataframe)
        )

        if shuffle:
            dataset = dataset.shuffle(2000)
        dataset = dataset.batch(batch_size).repeat(num_of_epochs)

        return dataset

    return input_feed_function

# Train
train_feed_input = feed_input(normed_train_features, training_labels)
train_feed_input_testing = feed_input(
    normed_train_features,
    training_labels,
    num_of_epochs=1,
    shuffle=False
)
# Test
test_feed_input = feed_input(
    normed_test_features,
    test_labels,
    num_of_epochs=1,
    shuffle=False
)

# Model training
feature_columns_numeric = [tf.feature_column.numeric_column(m) for m in training_features.columns]
btree_model = BoostedTreesClassifier(
    feature_columns=feature_columns_numeric,
    n_batches_per_layer=1
)
btree_model.train(train_feed_input)

# Predictions
train_predictions = btree_model.predict(train_feed_input_testing)

test_predictions = btree_model.predict(test_feed_input)

train_predictions_series = pd.Series(
    [p['classes'][0].decode("utf-8") for p in train_predictions]
)

test_predictions_series = pd.Series(
    [p['classes'][0].decode("utf-8") for p in test_predictions]
)

train_predictions_df = pd.DataFrame(
    train_predictions_series,
    columns=['predictions']
)

test_predictions_df = pd.DataFrame(
    test_predictions_series,
    columns=['predictions']
)

training_labels.reset_index(drop=True, inplace=True)
train_predictions_df.reset_index(drop=True, inplace=True)

test_labels.reset_index(drop=True, inplace=True)
test_predictions_df.reset_index(drop=True, inplace=True)

train_labels_with_predictions_df = pd.concat(
    [training_labels, train_predictions_df],
    axis=1
)
print('TRAIN PREDICTIONS: \n', train_labels_with_predictions_df)

test_labels_with_predictions_df = pd.concat(
    [test_labels, test_predictions_df],
    axis=1
)
print('TEST PREDICTIONS: \n', test_labels_with_predictions_df)


# Validations
def calculate_binary_class_scores(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred.astype('int64'))
    precision = precision_score(y_true, y_pred.astype('int64'))
    recall = recall_score(y_true, y_pred.astype('int64'))

    return accuracy, precision, recall

train_accuracy_score, train_precision_score, train_recall_score = calculate_binary_class_scores(training_labels, train_predictions_series)

test_accuracy_score, test_precision_score, test_recall_score = calculate_binary_class_scores(test_labels, test_predictions_series)
# Print performance
print('Training Data Accuracy (%) = ', round(train_accuracy_score*100, 2))
print('Training Data Precision (%) = ', round(train_precision_score*100, 2))
print('Training Data Recall (%) = ', round(train_recall_score*100, 2))

print('-'*50)

print('Test Data Accuracy (%) = ', round(test_accuracy_score*100, 2))
print('Test Data Precision (%) = ', round(test_precision_score*100, 2))
print('Test Data Recall (%) = ', round(test_recall_score*100, 2))

"""
Training Data Accuracy (%) =  97.5
Training Data Precision (%) =  95.24
Training Data Recall (%) =  100.0
--------------------------------------------------
Test Data Accuracy (%) =  100.0
Test Data Precision (%) =  100.0
"""

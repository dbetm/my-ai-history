""" Implement the linear regression method in TF2.3, using the
Boston housing data set and the LinearRegressor estimator.

Samples total: 506
Dimensionality: 13
Features: #real, positive
Targets: #real [5. - 50.]

Linear Regression:
We are trying to map the inputs and the output, such that we are able to
predict the numeric output, such that we are able to predict the numeric output:

y = m1x1 + m2x2 + ..., + mnxn + b

Where x1, x2, ..., xn are different input features, m1, m2, ..., mn are the
different slopes for different features, and b is the intercept.
"""

# Implementation testing with TF2.3

from __future__ import absolute_import, division, print_function
from __future__ import unicode_literals

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sb
from sklearn import datasets
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras as ks
from tensorflow.estimator import LinearRegressor

print("-"*10, tf.__version__, "-"*10)

# Load and configure the Boston housing data set
boston_load = datasets.load_boston()
feature_columns = boston_load.feature_names
target_column = boston_load.target
boston_data = pd.DataFrame(
    boston_load.data,
    columns=feature_columns
).astype(np.float32)
boston_data['MEDV'] = target_column.astype(np.float32)
print(boston_data.head())

# Check the relation between the variables, using pairplot
# sb.pairplot(boston_data, diag_kind='kde')
# plt.show()

# Check the relation between the variables, using a correlation graph
correlation_data = boston_data.corr()
# correlation_data.style.background_gradient(cmap='coolwarm', axis=None)
sb.heatmap(
    correlation_data,
    xticklabels=correlation_data.columns.values,
    yticklabels=correlation_data.columns.values
)
""" Some features with strongest correlation (near to 1):
- Tax and Rad
- Tax and Indus
"""
# plt.show()

# Descriptive stats - central tendency and dispersion
stats = boston_data.describe()
boston_stats = stats.transpose()
print(boston_stats)

# Select the required columns
X_data = boston_data[[i for i in boston_data.columns if i not in ['MEDV']]]
Y_data = boston_data[['MEDV']]

# Train the test split
features = train_test_split(X_data, Y_data, test_size=0.2)
training_features, test_features, training_labels, test_labels = features
# Display num of instances for each dataset
print('#rows in Training Features: ', training_features.shape[0])
print('#rows in Test Features: ', test_features.shape[0])
# More ...
print('#columns in Training Features: ', training_features.shape[1])
print('#columns in Test Features: ', test_features.shape[1])

# Normalize the data

def norm(X):
    stats = X.describe()
    stats = stats.transpose()
    return (X - stats['mean']) / stats['std']

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

train_feed_input = feed_input(normed_train_features, training_labels)
train_feed_input_testing = feed_input(
    normed_train_features,
    training_labels,
    num_of_epochs=1,
    shuffle=False
)

test_feed_input = feed_input(
    normed_test_features,
    test_labels,
    num_of_epochs=1,
    shuffle=False
)

# Model training
feature_columns_numeric = [tf.feature_column.numeric_column(m) for m in training_features.columns]
linear_model = LinearRegressor(
    feature_columns=feature_columns_numeric,
    optimizer='RMSProp'
)
linear_model.train(train_feed_input)

# Predictions
train_predictions = linear_model.predict(train_feed_input_testing)
test_predictions = linear_model.predict(test_feed_input)

train_predictions_series = pd.Series(
    [p['predictions'][0] for p in train_predictions]
)

test_predictions_series = pd.Series(
    [p['predictions'][0] for p in test_predictions]
)

train_predictions_df = pd.DataFrame(
    train_predictions_series, columns=['predictions']
)

test_predictions_df = pd.DataFrame(
    test_predictions_series, columns=['predictions']
)

training_labels.reset_index(drop=True, inplace=True)
train_predictions_df.reset_index(drop=True, inplace=True)

test_labels.reset_index(drop=True, inplace=True)
test_predictions_df.reset_index(drop=True, inplace=True)

train_labels_with_predictions_df = pd.concat(
    [training_labels, train_predictions_df],
    axis=1
)
print('Train labels with predictions')
print(train_labels_with_predictions_df)

test_labels_with_predictions_df = pd.concat(
    [test_labels, test_predictions_df],
    axis=1
)
print('Test labels with predictions')
print(test_labels_with_predictions_df)

# Validación
def calculate_errors_and_r2(y_true, y_pred):
    mean_squared_err = (mean_squared_error(y_true, y_pred))
    root_mean_squared_err = np.sqrt(mean_squared_err)
    r2 = round(r2_score(y_true, y_pred)*100,0)

    return mean_squared_err, root_mean_squared_err, r2

# Print errors and r2 score
    # Train
errors_and_r2 = calculate_errors_and_r2(training_labels, train_predictions_series)
train_MSE, train_root_MSE, train_r2_score_percentage = errors_and_r2
print('Training Data Mean Squared Error = ', train_MSE)
print('Training Data Root Mean Squared Error = ', train_root_MSE)
print('Training Data R2 = ', train_r2_score_percentage)
    # Test
errors_and_r2 = calculate_errors_and_r2(test_labels, test_predictions_series)
test_MSE, test_root_MSE, test_r2_score_percentage = errors_and_r2
print('Test Data Mean Squared Error = ', test_MSE)
print('Test Data Root Mean Squared Error = ', test_root_MSE)
print('Test Data R2 = ', test_r2_score_percentage)

import math
from math import log as ln

import numpy as np


class Neuron:
    # Don't change anything in the `__init__` function.
    def __init__(self, examples):
        np.random.seed(42)
        # Three weights: one for each feature and one more for the bias.
        self.weights = np.random.normal(0, 1, 3 + 1)
        self.examples = examples
        self.train()

    # Don't use regularization.
    # Use mini-batch gradient descent (get only <batch_size> elements from dataset at each iteration)
    # Use the sigmoid activation function.
    # Use the defaults for the function arguments.
    def train(self, learning_rate=0.01, batch_size=2, epochs=200):
        for epoch in range(epochs):
            # sample = np.random.choice(self.examples, batch_size, replace=False)
            acc_loss = 0

            for batch_window in range(len(self.examples) // batch_size):
                offset = (batch_size * batch_window)
                sample = self.examples[0 + offset : batch_size + offset]

                acc_gradients = [0] * len(self.weights)

                for example in sample:
                    y_expected = example["label"]
                    x = example["features"]

                    # forward
                    r = 0
                    for idx in range(len(x)):
                        r += self.weights[idx] * x[idx]
                    z = r + self.weights[-1]
                    a = self.sigmoid(z)

                    # backward
                    loss = self.log_loss(y_expected, a)
                    acc_loss += loss

                    # how does change loss fn when change w_0?
                    dlogloss_da = (-y_expected / a) + ((1 - y_expected) / (1 - a))
                    da_dz = a * (1 - a)
                    dz_r0 = 1.0
                    dr0_dw0 = x[0]
                    w0_grad = dr0_dw0 * dz_r0 * da_dz * dlogloss_da

                    # how does change loss fn when change w_1?
                    dz_r1 = 1.0
                    dr1_dw1 = x[1]
                    w1_grad = dr1_dw1 * dz_r1 * da_dz * dlogloss_da

                    # how does change loss fn when change w_2?
                    dz_r2 = 1.0
                    dr2_dw2 = x[2]
                    w2_grad = dr2_dw2 * dz_r2 * da_dz * dlogloss_da

                    # How does change loss function when change b?
                    dz_db = 1.0
                    bias_grad = dz_db * da_dz * dlogloss_da

                    # accumulated gradients for parameters
                    acc_gradients[0] += w0_grad
                    acc_gradients[1] += w1_grad
                    acc_gradients[2] += w2_grad
                    acc_gradients[3] += bias_grad

                for idx in range(len(acc_gradients)):
                    acc_gradients[idx] /= batch_size

                # update weight and bias (model's parameters)
                for idx in range(len(self.weights)):
                    self.weights[idx] = self.weights[idx] - learning_rate*acc_gradients[idx]

            print(f"Epoch {epoch}", f"Loss {acc_loss}")

    # Return the probability—not the corresponding 0 or 1 label.
    def predict(self, features):
        r = 0
        for idx in range(len(features)):
            r += self.weights[idx] * features[idx]

        bias = self.weights[-1]
        z = r + bias

        return self.sigmoid(z)

    def log_loss(self, y_expected: float, y_pred: float) -> float:
        return -(y_expected*ln(y_pred) + (1-y_expected)*ln(1-y_pred))
    
    def sigmoid(self, x):
        return 1 / (1 + math.exp(-x))




if __name__ == "__main__":
    examples = [
        {"features": [0.7737498370415932, 0.893981580520576, 0.7776116731845149], "label": 0},
        {"features": [0.8356527294792708, 0.7535044575176968, 0.7940884252881397], "label": 0},
        # More examples.
        {"features": [0.25835793676162827, 0.2166447564607853, 0.5066866046843734], "label": 1},
        {"features": [0.34848185391755987, 0.15010261370695727, 0.3466287718524547], "label": 1},
        # More examples.
    ]

    model = Neuron(examples=examples)

    
    input_ = [0.79, 0.89, 0.777]
    prediction = model.predict(input_)

    print(f"{prediction=}")
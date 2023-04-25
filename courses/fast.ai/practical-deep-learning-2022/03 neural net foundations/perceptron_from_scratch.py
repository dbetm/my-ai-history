"""This practical exercise intends to build a simple one-layer neural network 
(logistic regression) using only Python, using backprop to update the 
weights and bias value. Explicit code to understand to low-level the 
flow of information.

Problem:
Binary classification - determine the class of an iris flower (setosa 0 or versicolor 1).
Using the sepal_lenght and sepal_width features. All data is used to train.

The problem is linear-solvable.
"""


"""ELEMENTS

ACTIVATION FUNCTION: Sigmoid -> 1 / (1 + e^(-x))
LOSS FUNCTION: MSE -> sum(y_pred - y)^2 // n
WEIGHTS: w_0, w_1
B: b
LEARNING_RATE = 0.1

"""
import math
import random
from typing import List


THRESHOLD = 0.5 # threshold to define when it's a iris-vericolor with label 1
CLASSES = {
    0: "Iris-setosa",
    1: "Iris-versicolor",
}


def load_data(dataset_path: str) -> tuple:
    x = list()
    y = list()
    
    with open(dataset_path, "r") as f:
        lines = f.read().split("\n")
        number_lines = len(lines)

        for idx, line in enumerate(lines):
            if idx == 0 or idx == (number_lines - 1):
                continue

            sepal_lenght, sepal_width, class_name = line.split(",")
            x.append((float(sepal_lenght), float(sepal_width)))

            y.append(0 if class_name == "Iris-setosa" else 1)
    
    return (x, y)


def initialize_weights(n: int) -> List[float]:
    return [random.random() for _ in range(n)]


def sigmoid_fn(z: float) -> float:
    """Sigmoid activation function."""
    return 1 / (1 + math.exp(-z))


def mse(y_predicted: float, y_expected: int) -> float:
    """Mean square error."""
    return 0.5 * (y_predicted - y_expected)**2


def train(x: list, y: list, epochs: int = 5) -> List[float]:
    """Return vector of weights [w1, w2, b]"""
    w = initialize_weights(2)
    b = 0.0
    learning_rate = 0.01
    n = len(x)

    assert len(x) == len(y)

    for epoch in range(1, epochs + 1):
        accumulated_loss = 0.0
        correct = 0

        for x_i, y_i in zip(x, y):
            # ------ Forward ------
            r0 = w[0] * x_i[0]
            r1 = w[1] * x_i[1]
            z = r0 + r1 + b
            a = sigmoid_fn(z)

            label = int(a > THRESHOLD)
            correct = correct + 1 if label == y_i else correct

            # ------ Backward ------
            loss = mse(y_predicted=a, y_expected=y_i)
            accumulated_loss += loss
            # Partial derivatives
    
            # How does change loss function when change w_0?
            dmse_da = a - y_i
            da_dz = a * (1.0 - a)
            dz_dr0 = 1.0
            dr0_dw0 = x_i[0]
            w0_gradient = dr0_dw0 * dz_dr0 * da_dz * dmse_da

            # How does change loss function when change w_1?
            dz_dr1 = 1.0
            dr1_dw0 = x_i[1]
            # the rest of derivativaes are already computed
            w1_gradient = dr1_dw0 * dz_dr1 * da_dz * dmse_da

            # How does change loss function when change b?
            dz_db = 1.0
            # the rest of derivativaes are already computed
            b_gradient = dz_db * da_dz * dmse_da

            # Update weights and bias parameters
            w[0] = w[0] - learning_rate * w0_gradient
            w[1] = w[1] - learning_rate * w1_gradient
            b = b - learning_rate * b_gradient

        print(
            f"Epoch {epoch}",
            f"Loss {accumulated_loss / n}",
            f"Accuracy {correct / n}"
        )

    return [w[0], w[1], b]


def predict(model: List[float], x: List[float], expected: int):
    w_0, w_1, b = model

    r0 = w_0 * x[0]
    r1 = w_1 * x[1]
    z = r0 + r1 + b
    prob = sigmoid_fn(z)

    # z = (w_0 * x[0]) + (w_1 * x[1]) + b
    # prob = sigmoid_fn(z)

    label = int(prob > THRESHOLD)

    if label == 0:
        prob = 1 - prob

    print(
        f"Predicted {CLASSES[label]} - Confidence: {prob} - Expected: {CLASSES[expected]}"
    )


if __name__ == "__main__":
    DATASET_PATH = "../../../../datasets/mini_iris.csv"
    x, y = load_data(DATASET_PATH)

    model = train(x, y, epochs=100)

    print("\nMODEL")
    print(model)

    print("Inference")

    for (x0, x1), label_expected in zip(x, y):
        predict(model, (x0, x1), label_expected)
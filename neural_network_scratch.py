"""
We want to create a neural network from scratch
with the following architecture:
    784 input nodes
    10 hidden nodes
    10 output nodes
"""


import numpy as np


def softmax(x):
    return np.exp(x) / np.sum(np.exp(x), axis=0)


def relu(x):
    return np.maximum(0, x)


def relu_derivative(x):
    return x > 0


def get_predictions(A2):
    return np.argmax(A2, 0)


def get_accuracy(predictions, y):
    print(predictions, y)
    print(predictions == y)
    return np.sum(predictions == y) / y.size


def one_hot(y):
    one_hot_Y: np.zeros(
        (y.size, y.max() + 1)
    )  # classes are 0 to 9, so we need 10 columns
    one_hot_Y[np.arange(y.size), y] = 1

    return one_hot_Y.T


def init_params():
    w1 = np.random.rand(10, 784) - 0.5
    b1 = np.random.rand(10, 1) - 0.5

    w2 = np.random.rand(10, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5

    return w1, b1, w2, b2


def forward_prop(w1, b1, w2, b2, x):
    z1 = np.dot(w1, x) + b1
    A1 = relu(z1)
    z2 = np.dot(w2, A1) + b2
    A2 = softmax(z2)
    return z1, A1, z2, A2


def back_prop(z1, A1, z2, A2, W2, x, y):
    m = y.size
    one_hot_Y = one_hot(y)
    dz2 = A2 - one_hot_Y  # prediction minus target
    dw2 = 1 / m * np.dot(dz2, A1.T)
    db2 = 1 / m * np.sum(dz2, axis=1, keepdims=True)
    dz1 = np.dot(W2.T, dz2) * relu_derivative(z1)
    dw1 = 1 / m * np.dot(dz1, x.T)
    db1 = 1 / m * np.sum(dz1, axis=1, keepdims=True)

    return dw1, db1, dw2, db2


def update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha):
    w1 -= alpha * dw1
    b1 -= alpha * db1
    w2 -= alpha * dw2
    b2 -= alpha * db2

    return w1, b1, w2, b2


def gradient_descent(x, y, epochs, alpha):
    w1, b1, w2, b2 = init_params()
    for _ in range(epochs):
        z1, A1, z2, A2 = forward_prop(w1, b1, w2, b2, x)
        dw1, db1, dw2, db2 = back_prop(z1, A1, z2, A2, w2, x, y)
        w1, b1, w2, b2 = update_params(w1, b1, w2, b2, dw1, db1, dw2, db2, alpha)
        if _ % 10 == 0:
            print("Epoch", _, "cost: ", get_accuracy(get_predictions(A2), y))
    return w1, b1, w2, b2

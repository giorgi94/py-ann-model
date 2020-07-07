import pickle
import random
from typing import List, Callable

import numpy as np
from numpy import ndarray

from .activations import sigmoid


class Layer:

    learning_rate = 0.5

    def __init__(self, inp, out, activation):
        self.inp: int = inp
        self.out: int = out
        self.activation: Callable = activation

        self.Z: ndarray = None
        self.weight: ndarray = None
        self.bias: ndarray = None

    def activation_prime(self):
        return np.diagflat(self.activation(self.Z, prime=True))

    def load_random(self):
        n0, n1 = self.inp, self.out

        self.weight = 2 * np.random.random((n1, n0)) - 1
        self.bias = 2 * np.random.random((n1, 1)) - 1

    def output(self):
        if self.Z is None:
            return None
        return self.activation(self.Z)

    def forward(self, X):
        w = self.weight
        b = self.bias

        self.Z = w.dot(X) + b
        return self.activation(self.Z)

    def backward(self, Y, X):
        delta = self.output() - Y
        act = self.activation_prime()

        d = self.learning_rate * act.dot(delta)

        dX = self.weight.T.dot(d)
        self.bias -= d
        self.weight -= d.dot(X.T)

        return X - dX


class Model:
    def __init__(self):
        self.biases: list = []
        self.weights: list = []
        self.layers: List[Layer] = []

    def add_layer(self, inp, out, activation):
        self.layers.append(Layer(inp, out, activation))

    def load_random(self):
        for l in self.layers:
            l.load_random()

    def forward(self, X):
        x = X.copy()

        for i, l in enumerate(self.layers):
            x = l.forward(x)

        return x

    def output(self):
        return self.layers[-1].output()

    def backward(self, X, Y):
        layers_len = len(self.layers)
        y = Y.copy()

        for i, layer in enumerate(self.layers[::-1]):
            j = layers_len - i - 1

            if j == 0:
                x = X.copy()
            else:
                x = self.layers[j - 1].output()

            y = layer.backward(y, x)


class ANNetworkBase:

    learning_rate = 0.5

    def activation_prime(self, x, act):
        return np.diagflat(act(x, prime=True))

    def load_layers(self, layers):
        self.layers = layers
        self.last = len(layers) - 1

    def load_random_weights(self):
        self.biases = []
        self.weights = []

        n0 = self.layers[0]

        for n1 in self.layers[1:]:
            self.weights.append(2 * np.random.random((n1, n0)) - 1)
            self.biases.append(2 * np.random.random((n1, 1)) - 1)
            n0 = n1

    def forward(self, X):
        pass

    def backward(self, Y):
        pass

    def output(self):
        pass

    def check_error_norm(self, Y):
        return np.linalg.norm(Y - self.output())

    def train(self, X, Y):
        self.forward(X)
        self.backward(Y)

    def training(self, table, max_steps, each=None):

        for step in range(max_steps):
            for row in table:
                X, Y = row
                self.train(X, Y)

            if each and step % each == 0:
                print(self.check_error_norm(Y))

            random.shuffle(table)

    def dump(self, path="dump.pkl"):
        path = os.path.abspath(path)
        assure_path_exists(path)

        with open(path, "wb") as f:
            pickle.dump((self.weights, self.biases), f)

    def load(self, path="dump.pkl"):
        path = os.path.abspath(path)

        with open(path, "rb") as f:
            self.weights, self.biases = pickle.load(f)


class ANNetwork(ANNetworkBase):
    def forward(self, X):
        self.z = [None]
        self.a = [X]

        layer = 0
        for w, b in zip(self.weights, self.biases):
            layer += 1
            self.z.append(w.dot(self.a[layer - 1]) + b)
            self.a.append(self.activation(self.z[layer]))

    def backward(self, Y):
        delta = self.output() - Y

        for layer in range(self.last, 0, -1):
            b_delta = self.learning_rate * self.activation_prime(self.z[layer]).dot(
                delta
            )

            self.biases[layer - 1] -= b_delta
            self.weights[layer - 1] -= b_delta.dot(self.a[layer - 1].T)

            delta = None

            if layer != 1:
                delta = self.weights[layer - 1].T.dot(b_delta)

    def output(self):
        return self.a[-1]

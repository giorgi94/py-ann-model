import pickle
import random
from typing import List, Callable

import numpy as np
from numpy import ndarray

from .activations import sigmoid


class Layer:

    learning_rate = 0.5

    def __init__(self, inp, out, activation=sigmoid):
        self.inp: int = inp
        self.out: int = out
        self.activation = activation

        self.Z: ndarray = None
        self.weight: ndarray = None
        self.bias: ndarray = None

    def activation_prime(self):
        return np.diagflat(self.activation(self.Z, prime=True))

    def load_random(self):
        n0, n1 = self.inp, self.out

        self.weight = 2 * np.random.random((n1, n0)) - 1
        self.bias = 2 * np.random.random((n1, 1)) - 1

    def serialize(self):
        return {
            "layer": self.__class__.__name__,
            "inp": self.inp,
            "out": self.out,
            "activation": self.activation.__name__,
            "weight": weight,
            "bias": bias,
        }

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
        self.layers: list = []

    def add_layer(self, layout):
        self.layers.append(layout)

    def load_random(self):
        for l in self.layers:
            l.load_random()

    def save(self):
        data = {""}

    def forward(self, X):
        x = X.copy()

        for i, l in enumerate(self.layers):
            x = l.forward(x)

        return x

    def train(self, X, Y):
        self.forward(X)
        self.backward(X, Y)

    def output(self):
        return self.layers[-1].output()

    def check_error_norm(self, Y, y):
        return np.linalg.norm(y - Y)

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

        return self.check_error_norm(Y, self.layers[-1].output())

    def save(self, path):
        with open(path, "wb") as fp:
            pickle.dump(self, fp)

    @staticmethod
    def load(path):
        with open(path, "rb") as fp:
            return pickle.load(fp)

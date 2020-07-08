import os
import sys
import pickle
import numpy as np

from ann.models import Model, Layer
from ann.activations import sigmoid

from word import encode_word, merge_words


np.set_printoptions(precision=16)


def main():
    N = Model()

    N.add_layer(Layer(inp=3, out=15))
    N.add_layer(Layer(inp=15, out=25))
    N.add_layer(Layer(inp=25, out=2))

    N.load_random()

    data = [
        [[0, 1, 0], [0, 1]],
        [[0, 0, 1], [1, 1]],
        [[1, 0, 0], [1, 0]],
        [[1, 0.5, 0], [1, 0.5]],
    ]

    train = [
        [np.array(x).reshape((3, 1)), np.array(y).reshape((2, 1))] for x, y in data
    ]

    # X = np.array([[0.3], [0.7], [0.5]])
    # Y = np.array([[0.247], [0.847], [0.5], [0.3]])

    for _ in range(700):
        for X, Y in train:
            N.forward(X)
            N.backward(X, Y)

    print(N.forward(train[0][0]).flatten())
    print(N.forward(train[1][0]).flatten())
    print(N.forward(train[2][0]).flatten())
    print(N.forward(train[3][0]).flatten())


def word_main():
    n = 50

    a = "კატლეტი"
    b = "კოტლეტი"

    X = np.array(merge_words(a, b, n)).reshape((n, 1))

    # N = Model.load("dist/model.pkl")

    N = Model()

    N.add_layer(Layer(inp=n, out=15))
    N.add_layer(Layer(inp=15, out=25))
    N.add_layer(Layer(inp=25, out=2))

    N.load_random()

    train = [(X, np.array([[1.0], [0.0]]))]

    for _ in range(100):
        for X, Y in train:
            N.forward(X)
            N.backward(X, Y)

    print(N.forward(X))


if __name__ == "__main__":
    # main()

    word_main()

import os
import pickle
import random
import sys

import numpy as np

from ann.activations import sigmoid
from ann.models import Layer, Model
from ann.word import distance, encode_word, merge_words

np.set_printoptions(precision=16)

encN = 50


def create_model(load=False):

    n = encN

    if load:
        return Model.load("dist/model.pkl")

    N = Model()

    N.add_layer(Layer(inp=n, out=25))
    N.add_layer(Layer(inp=25, out=25))
    N.add_layer(Layer(inp=25, out=2))

    N.load_random()

    return N


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


def get_X(a, b):
    n = encN

    return np.array(merge_words(a, b, n)).reshape((n, 1))


def generate_training_data(word_cluster):
    n = encN

    words = []

    for c in word_cluster:
        words.extend(c)

    samples = [random.sample(words, 2) for _ in range(50)]

    data = []

    for a, b in samples:
        same = False
        for cluster in word_cluster:
            if a in cluster and b in cluster:
                same = True
                break

        X = np.array(merge_words(a, b, n)).reshape((n, 1))
        Y = np.array([[1.0], [0.0]]) if same else np.array([[0.0], [1.0]])
        data.append((X, Y))

    return data


def word_main():

    word_cluster = [
        ("ყიდვა", "ყიდულობს", "ყუდილუბდა", "საყიდელი", "გასაყიდი"),
        ("სწავლა", "სწავლობს", "სასწავლო", "სწავლობდა"),
    ]

    data = generate_training_data(word_cluster)

    N = create_model(load=True)

    p, q = N.forward(get_X("მოსწავლე", "მოსწავლემ")).flatten().tolist()

    print(p)
    print(q)
    print(p > q)

    # for _ in range(800):
    #     for X, Y in data:
    #         N.forward(X)
    #         N.backward(X, Y)

    # N.save("dist/model.pkl")


if __name__ == "__main__":

    # main()

    word_main()

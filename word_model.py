import os
import pickle
import random
import sys

import numpy as np

from ann.activations import sigmoid
from ann.models import Layer, Model
from ann.word2vec import distance, encode_word, merge_words

np.set_printoptions(precision=16)


class WordSimilarity(Model):
    word_vec_len = 50

    @classmethod
    def create_model(cls, load=None):
        n = cls.word_vec_len

        if load:
            return cls.load(load)

        N = cls()

        N.add_layer(Layer(inp=n, out=25))
        N.add_layer(Layer(inp=25, out=25))
        N.add_layer(Layer(inp=25, out=2))

        N.load_random()

        return N

    @classmethod
    def get_X(cls, a: str, b: str):
        n = cls.word_vec_len
        return np.array(merge_words(a, b, n)).reshape((n, 1))

    @staticmethod
    def get_Y(c: bool):
        return np.array([[1.0], [0.0]]) if c else np.array([[0.0], [1.0]])

    @classmethod
    def generate_training_data(cls, clusters, samples=50):
        n = cls.word_vec_len

        words = []

        for c in clusters:
            words.extend(c)

        samples = [random.sample(words, 2) for _ in range(samples)]

        data = []

        for a, b in samples:
            same = False
            for cluster in clusters:
                if a in cluster and b in cluster:
                    same = True
                    break

            X = cls.get_X(a, b)
            Y = cls.get_Y(same)
            data.append((X, Y))

        return data

    def validate(self, X):
        return self.forward(X).flatten().tolist()


def main():

    clusters = [
        ("ყიდვა", "ყიდულობს", "ყუდილუბდა", "საყიდელი", "გასაყიდი"),
        ("სწავლა", "სწავლობს", "სასწავლო", "სწავლობდა"),
    ]

    N: WordSimilarity = WordSimilarity.create_model(load="dist/model.pkl")

    print(N)

    data = N.generate_training_data(clusters)

    X = N.get_X("ყიდვა", "გაყიდვა")

    p, q = N.validate(X)

    print(p)
    print(q)
    print(p > q)

    # N.backward(X, get_Y(True))

    # print(distance("მოსწავლე", "მოსწავლემ"))

    # for _ in range(800):
    #     for X, Y in data:
    #         N.forward(X)
    #         N.backward(X, Y)


if __name__ == "__main__":

    main()

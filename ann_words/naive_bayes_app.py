import re
from math import prod

import numpy as np
import pandas as pd

from .word2vec import get_word_list


class NaiveBayesClassification:

    labels = [(0, "negative"), (1, "positive")]

    vocabulary: list = []

    df: pd.DataFrame
    wdf: pd.DataFrame

    def load_raw_data(self, data):
        words = []

        for text, _ in data:
            words.extend(text)
        words = list(set(words))
        words.sort()

        self.vocabulary = words[:]
        self.vlen = len(words)

        parse = lambda word, wlist: wlist.count(word)

        parse_item = lambda wlist: [wlist.count(w) for w in words]

        wlist, c = data[0]

        self.df = pd.DataFrame(
            np.array([[c, *parse_item(wlist)] for wlist, c in data], dtype=np.uint64),
            columns=["C", *words],
        )

    def label_P(self, key):
        total = self.df["C"].size
        count = self.df["C"][self.df["C"] == key].size

        return count / total

    def P(self, w: list):
        r = {}

        for key, name in self.labels:
            df = self.df[self.df["C"] == key]
            n = df[w].sum()

            total = df.drop("C", axis=1).sum().sum()

            r[name] = (1 + n) / (total + self.vlen)
        return r

    def train(self):
        weights = {w: self.P(w) for w in self.vocabulary}
        weights["C"] = {name: self.label_P(key) for key, name in self.labels}

        self.wdf = pd.DataFrame.from_dict(weights)

    def classify(self, text):
        cases = []

        word_list = get_word_list(text)

        word_list_v = [w for w in word_list if w in self.vocabulary]
        word_list_n = [w for w in word_list if w not in word_list_v]

        for _, name in self.labels:
            p = self.wdf.T[name][["C", *word_list_v]].product()
            for _ in word_list_n:
                p *= 1 / self.vlen
            cases.append((name, p))

        cases.sort(key=lambda c: -c[1])

        return cases[0][0]


def test():

    data = [
        ("I loved the movie", 1),
        ("I hated the movie", 0),
        ("a great movie. good movie", 1),
        ("poor acting", 0),
        ("great acting. a good movie", 1),
    ]

    data = [(re.findall(r"[a-zA-Z]+", text), c) for text, c in data]

    nbc = NaiveBayesClassification()
    nbc.load_raw_data(data)
    nbc.train()

    print(nbc.classify("I hated the movie"))

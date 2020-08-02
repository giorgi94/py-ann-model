import json
import re
from math import prod

from .word2vec import get_word_list, clean_stopwords


class NaiveBayesClassification:

    data = []
    outcomes = []
    vocabulary = []
    table = []

    words = {}

    def dump(self, filename):
        with open(filename, "w") as fp:
            json.dump(
                {
                    "positive": self.positive,
                    "negative": self.negative,
                    "words": self.words,
                },
                fp,
                ensure_ascii=False,
            )

    def load(self, filename):
        with open(filename, "r") as fp:
            data = json.load(fp)

        self.positive = data["positive"]
        self.negative = data["negative"]
        self.words = data["words"]

        self.vocabulary = list(self.words.keys())

    def train(self, data):
        self.outcomes = [c for _, c in data]

        self.positive = sum(self.outcomes) / len(self.outcomes)
        self.negative = 1 - self.positive

        self.data = [t for t, _ in data]
        self.create_vocabulary()
        self.create_table()

        self.words = {v: {} for v in self.vocabulary}

        self.select_outcomes(0)
        self.select_outcomes(1)

    def create_vocabulary(self):
        t = len(self.data)
        i = 0
        for s in self.data:
            i += 1
            print(i, "/", t)
            for w in s:
                if w not in self.vocabulary:
                    self.vocabulary.append(w)

    def create_table(self):
        vlen = len(self.vocabulary)
        i = 0
        print("Create Table:")
        for v in self.vocabulary:
            i += 1
            print(i, "/", vlen)
            self.table.append([w.count(v) for w in self.data])

    def select_outcomes(self, outcome):
        def clean(col):
            return [c for c, i in zip(col, self.outcomes) if i == outcome]

        name = "+" if outcome == 1 else "-"

        filtered_tabel = [clean(col) for col in self.table]

        n = sum(sum(t) for t in filtered_tabel)

        for i, v in enumerate(self.vocabulary):
            self.words[v][name] = (sum(filtered_tabel[i]) + 1) / (
                n + len(self.vocabulary)
            )

    def classify(self, text):
        def get_pw(w, o):
            if w in self.words:
                return self.words[w][o]
            return 1 / len(self.vocabulary)

        data = clean_stopwords(get_word_list(text))

        p1 = self.positive * prod(get_pw(w, "+") for w in data)
        p2 = self.negative * prod(get_pw(w, "-") for w in data)

        return p1, p2


def test():

    data = [
        ("I loved the movie", True),
        ("I hated the movie", False),
        ("a great movie. good movie", True),
        ("poor acting", False),
        ("great acting. a good movie", True),
    ]

    data = [(get_word_list(t), c) for t, c in data]

    nbc = NaiveBayesClassification()
    nbc.train(data)

    text = "great acting"

    print(nbc.classify(text))

import json
import re
from math import prod

from .word2vec import get_word_list


class NaiveBayesClassification:

    labels = ["+", "-"]

    cases = {}
    words = {}

    trained = {}

    def __init__(self):
        self.cases = {l: 0 for l in self.labels}

    def parse_text(self, words, label):
        self.cases[label] += 1

        for w in words:
            info = self.words.get(w, {label: 0})

            c = info.get(label, 0)
            info[label] = c + 1

            self.words[w] = info

    def _probabily(self, label):
        total = sum((c for _, c in self.cases.items()))
        return self.cases[label] / total

    def _total_cases(self, label):
        return sum((info.get(label, 0) for _, info in self.words.items()))

    def _word_probability(self, word, label, total=None, vlen=None):
        info = self.words.get(word, {label: 0})

        _a = info.get(label, 0)

        _b = total if total is not None else self._total_cases(label)
        _c = vlen if vlen is not None else len(self.words)

        return (_a + 1) / (_b + _c)

    def train(self):
        trained_words = {}
        trained_cases = {l: self._probabily(l) for l in self.labels}
        total_cases = {l: self._total_cases(l) for l in self.labels}

        vocabulary_len = len(self.words)

        for w, _ in self.words.items():
            trained_words[w] = {
                l: self._word_probability(w, l, total_cases[l], vocabulary_len)
                for l in self.labels
            }

        self.trained = {"words": trained_words, "cases": trained_cases}

    def classify(self, text):
        words = get_word_list(text)

        trained_cases = self.trained["cases"]
        trained_words = self.trained["words"]

        filtered = [(w, info) for w, info in trained_words.items() if w in words]

        c = []

        for l in self.labels:
            _p = trained_cases[l] * prod((info[l] for _, info in filtered))
            c.append((l, _p))

        return max(c, key=lambda itm: itm[1])[0]

    def load(self, path):
        with open(path, "r") as fp:
            data = json.load(fp)

            self.labels = data["labels"]
            self.cases = data["cases"]
            self.words = data["words"]
            self.trained = data["trained"]

    def dump(self, path):
        with open(path, "w") as fp:
            json.dump(
                {
                    "labels": self.labels,
                    "cases": self.cases,
                    "words": self.words,
                    "trained": self.trained,
                },
                fp,
                ensure_ascii=False,
            )


def test():

    data = [
        ("I loved the movie", "+"),
        ("I hated the movie", "-"),
        ("a great movie. good movie", "+"),
        ("poor acting", "-"),
        ("great acting. a good movie", "+"),
    ]

    nbc = NaiveBayesClassification()

    for (t, l) in data:
        nbc.parse_text(get_word_list(t), l)

    nbc.train()

    text = "poor movie"

    print(nbc.classify(text))

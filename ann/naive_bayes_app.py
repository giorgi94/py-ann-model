import re
from math import prod


class NaiveBayesClassification:

    data = []
    outcomes = []
    vocabulary = []
    table = []

    words = {}

    def load(self, data):
        self.positive = data["positive"]
        self.negative = data["negative"]
        self.words = data["words"]

        self.vocabulary = list(self.words.keys())

    def train(self, data):
        self.outcomes = [c for _, c in data]

        self.positive = sum(self.outcomes) / len(self.outcomes)
        self.negative = 1 - self.positive

        self.data = [self.get_word_list(t) for t, _ in data]
        self.create_vocabulary()
        self.create_table()

        self.words = {v: {} for v in self.vocabulary}

        self.select_outcomes(0)
        self.select_outcomes(1)

    @staticmethod
    def get_word_list(sentence):
        return [t.lower() for t in re.findall(r"\w+", sentence)]

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

        data = self.get_word_list(text)

        p1 = self.positive * prod(get_pw(w, "+") for w in data)
        p2 = self.negative * prod(get_pw(w, "-") for w in data)

        # print(p1)
        # print(p2)

        # return "+" if p1 > p2 else "-"
        if p2 == 0:
            return p1
        return p1 / p2


if __name__ == "__main__":

    data = [
        ("I loved the movie", 1),
        ("I hated the movie", 0),
        ("a great movie. good movie", 1),
        ("poor acting", 0),
        ("great acting. a good movie", 1),
    ]

    nbc = NaiveBayesClassification(data)

    text = "I loved acting"

    print(nbc.classify(text))

import json
import re

import numpy as np


def clean_stopwords(wlist):
    wl = []

    for w in wlist:
        v = remove_dublicates(w)
        if len(v) > 3 and v not in wl:
            wl.append(v)
    return wl


def striptags(text):
    regex = r"(<([^>]+)>)"
    return re.sub(regex, "", text)


def remove_dublicates(word):
    return re.sub(r"(\w)\1{1,}", r"\1", word)


def get_word_list(text):
    return [remove_dublicates(t.lower()) for t in re.findall(r"\w+", striptags(text))]


def tovec(w: str):
    wlen = len(w)
    wvec = [
        (i / (wlen - 1), (ord(l) - ord("ა")) / (ord("ჰ") - ord("ა")))
        for i, l in enumerate(w)
    ]

    return wvec


def encode_word(w, n=50):
    V = tovec(w)

    X = [x for x, _ in V]
    Y = [y for _, y in V]
    P = [i / n for i in range(n + 1)]
    E = []

    for p in P:
        for i, x in enumerate(X[1:]):
            if X[i] <= p and p <= x:
                a, b = X[i], x
                ya, yb = Y[i], Y[i + 1]
                E.append((yb - ya) / (b - a) * (p - a) + ya)
                break

    return E


def merge_words(a, b, n=50):

    X = [encode_word(a, n), encode_word(b, n)]

    # print([round(x, 3) for x in X[0]])
    # print([round(x, 3) for x in X[1]])
    X = np.array(X).T.flatten()

    H = [0.5, 0.5]

    Y = [
        (X[2 * i] + X[2 * i + 2]) * H[0] + (X[2 * i + 1] + X[2 * i + 3]) * H[1]
        for i in range(n)
    ]

    return Y


def distance(a, b, n=50):
    if len(a) == 1 or len(b) == 1:
        return 0 if a == b else 10

    A = np.array(encode_word(a, n))
    B = np.array(encode_word(b, n))

    return ((A - B) ** 2).sum() ** 0.5


if __name__ == "__main__":
    pass
    # cluster()

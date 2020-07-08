import json

import numpy as np


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
    A = np.array(encode_word(a, n))
    B = np.array(encode_word(b, n))

    return ((A - B) ** 2).sum() ** 0.5


def cluster():
    with open("dist/words.json", "r") as fp:
        words = json.load(fp)
        words = [w for w in words if len(w) > 2]

    Clu = []

    for a in words:
        col = [a]
        words.remove(a)
        for b in words:
            if distance(a, b) < 1:
                col.append(b)
                words.remove(b)
        # print(a)
        Clu.append(col)

    # with open("dist/cluster.json", "w") as fp:
    #     json.dump(Clu, fp, ensure_ascii=False)


if __name__ == "__main__":
    pass
    # cluster()

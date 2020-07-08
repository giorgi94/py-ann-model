import matplotlib.pyplot as plt
import numpy as np


def tovec(w: str):
    wlen = len(w)
    wvec = [
        (i / (wlen - 1), (ord(l) - ord("ა")) / (ord("ჰ") - ord("ა")))
        for i, l in enumerate(w)
    ]

    return wvec


def plot_word(word):

    P = tovec(word)

    X = [x for x, y in P]
    Y = [y for x, y in P]

    plt.plot(X, Y)


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

    print([round(x, 3) for x in X[0]])
    print([round(x, 3) for x in X[1]])
    X = np.array(X).T.flatten()

    H = [0.5, 0.5]

    Y = [
        (X[2 * i] + X[2 * i + 2]) * H[0] + (X[2 * i + 1] + X[2 * i + 3]) * H[1]
        for i in range(n)
    ]

    return Y


if __name__ == "__main__":
    words = ["კატლეტები", "საკატლეტე", "სანოვაგე"]

    for w in words:
        e = encode_word(w, 50)
        print(len(e))
    # plot_word(words[0])

    # plt.show()

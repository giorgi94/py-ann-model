import numpy as np


def sigmoid(x, prime=False):
    if prime:
        return sigmoid(x) * (1 - sigmoid(x))
    return 1 / (1 + np.exp(-x))

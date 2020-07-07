import os
import sys
import numpy as np

from ann.models import Model
from ann.activations import sigmoid


np.set_printoptions(precision=16)

N = Model()


N.add_layer(inp=3, out=20, activation=sigmoid)
N.add_layer(inp=20, out=10, activation=sigmoid)
N.add_layer(inp=10, out=2, activation=sigmoid)

N.load_random()

X = np.array([[0.3], [0.7], [0.5]])
Y = np.array([[0.247], [0.847]])


for _ in range(700):
    N.forward(X)
    N.backward(X, Y)


print(N.forward(X))

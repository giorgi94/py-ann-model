import os
import sys
import numpy as np

from ann.models import Model
from ann.activations import sigmoid


np.set_printoptions(precision=16)

N = Model()


N.add_layer(inp=1, out=20, activation=sigmoid)
N.add_layer(inp=20, out=10, activation=sigmoid)
N.add_layer(inp=10, out=3, activation=sigmoid)

N.load_random()

X = np.array([[1]])
Y = np.array([[0.247], [0.523], [0.546]])

N.backward(X, Y)

print(N.forward(X))

# y = N.layers[-1].backward(Y)

# print(y)

# print(X)

"""
N.activation = sigmoid

N.learning_rate = 0.05

tests = [
    (1, [0, 0, 1]),
    (2, [0, 1, 0]),
    (3, [0, 1, 0]),
    (4, [1, 0, 0]),
    (5, [0, 1, 0]),
    (7, [0, 0, 1]),
]

table = [(np.array([[x]]), np.array(y).reshape(3, 1)) for x, y in tests]

layers = [1, 20, 10, 3]


N.load_layers(layers)
N.load_random_weights()

N.load()

N.training(table[:], max_steps=200, each=1)

N.dump()


print("\ncheck:")

for test in table:
    X, Y = test
    N.forward(X)
    print(N.output().T.round(2), Y.T)
"""

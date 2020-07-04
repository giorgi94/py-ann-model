import os
import sys
import numpy as np

from ann_model import ANNetwork, sigmoid


np.set_printoptions(precision=16)

N = ANNetwork()


N.activation = sigmoid

N.learning_rate = 0.25

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
# N.load_random_weights()

N.load()

N.training(table[:], max_steps=20000, each=500)

# N.dump()


print("\ncheck:")

for test in table:
    X, Y = test
    N.forward(X)
    print(N.output().T.round(2), Y.T)

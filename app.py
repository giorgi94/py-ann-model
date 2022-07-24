from dataclasses import dataclass

import numpy as np


def sigmoid(x: float, der: bool = False) -> float:
    if der:
        return sigmoid(x) * (1 - sigmoid(x))
    return 1 / (1 + np.exp(-x))


@dataclass
class LinearLayer:

    __slots__ = ("weight", "bias")

    weight: np.ndarray
    bias: np.ndarray

    def calc(self, x: np.ndarray) -> np.ndarray:
        return self.weight.dot(x) + self.bias


x = np.array([[0.3, 0.1, 0.4]], dtype=np.float32).T
y = np.array([[0.8341, 0.9421]], dtype=np.float32).T

x.flags.writeable = False
y.flags.writeable = False


def train():
    weight = np.random.rand(2, 3)
    bias = np.random.rand(2, 1)

    learning_rate = 0.05

    epoch_size = 5000

    for epoch in range(epoch_size):

        z = weight.dot(x) + bias

        y_predicted = sigmoid(z)

        delta = y_predicted - y

        s = np.diag(sigmoid(z, True).flatten())

        error = (delta**2).sum() / 2

        # if (epoch + 1) % 10 == 0:
        #     print("step", epoch, "error:", error)

        if epoch != epoch_size - 1:

            der_b = s.dot(delta)
            der_w = der_b.dot(x.T)

            weight -= learning_rate * der_w
            bias -= learning_rate * der_b

    return weight, bias


def calc(weight, bias):

    print(sigmoid(weight.dot(x) + bias).tolist())
    print(y.tolist())


weight, bias = train()

print("answer", weight.tolist(), bias.tolist())

print("result:")

calc(weight, bias)

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


x = np.array([[0.3], [0.1], [0.4]], dtype=np.float32)
y = np.array([[0.8], [0.9]], dtype=np.float32)

weight = np.random.rand(2, 3)
bias = np.random.rand(2, 1)

layer = LinearLayer(weight, bias)

z = layer.calc(x)
a = sigmoid(x)


print(z)

import numpy as np
import random


def sigmoid(x: float, der: bool = False) -> float:
    if der:
        return sigmoid(x) * (1 - sigmoid(x))
    return 1 / (1 + np.exp(-x))


class SigmoidActivation:
    def __call__(self, x):
        return sigmoid(x)

    def derivative(self, x):
        return np.diag(sigmoid(x, True).flatten())


class LinearLayer:
    def __init__(self, weight, bias, learning_rate) -> None:
        self.weight: np.ndarray = weight
        self.bias: np.ndarray = bias

        self.learning_rate: float = learning_rate
        self.activation = SigmoidActivation()

    def calc(self, x: np.ndarray) -> np.ndarray:
        return self.weight.dot(x) + self.bias

    def forward(self, x: np.ndarray):
        return self.activation(self.calc(x))

    def error(self, y_predicted: np.ndarray, y: np.ndarray) -> float:

        return ((y_predicted - y) ** 2).sum() / 2

    def backward(self, x: np.ndarray, y: np.ndarray, x_correction: bool = False):

        z = self.calc(x)

        y_predicted = self.activation(z)

        delta = y_predicted - y

        der_act = self.activation.derivative(z)

        der_b = der_act.dot(delta)
        der_w = der_b.dot(x.T)

        if x_correction:
            der_x = self.weight.T.dot(der_b)

            x -= self.learning_rate * der_x

        self.weight -= self.learning_rate * der_w
        self.bias -= self.learning_rate * der_b


class BaseModel:
    def __init__(self, params: list, train: list) -> None:

        self.train_X = [self.__to_vector(x) for x, _ in train]
        self.train_Y = [self.__to_vector(y) for _, y in train]

        self.params = params

        self.layers = []

    @staticmethod
    def __to_vector(x: list) -> np.ndarray:
        a = np.array([x], dtype=np.float32).T
        a.flags.writeable = False
        return a

    @staticmethod
    def generate_random_params(*args):
        dims = zip(args[1:], args[:-1])
        return [(np.random.rand(i, j), np.random.rand(i, 1)) for i, j in dims]

    def predict(self, x):
        pred = x

        for layer in self.layers:
            pred = layer.forward(pred)

        return pred

    def train(self, epoch_size=50_000):

        for epoch in range(epoch_size):

            ind = random.randint(0, len(self.train_X) - 1)

            x = self.train_X[ind]
            y = self.train_Y[ind]

            predictions = [x]

            for layer in self.layers:
                predictions.append(layer.forward(predictions[-1]))

            predictions.reverse()

            pred_len = len(self.layers)

            need = y

            for i, layer in enumerate(self.layers[::-1]):
                layer.backward(predictions[i + 1], need, i != pred_len - 1)
                need = predictions[i + 1]


class Model(BaseModel):
    pass

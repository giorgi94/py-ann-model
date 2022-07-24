import numpy as np


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

    __slots__ = ("weight", "bias", "activation", "learning_rate")

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


x1 = np.array([[0.389, 0.134, 0.134]], dtype=np.float32).T
y1 = np.array([[0.8341, 0.9421]], dtype=np.float32).T

x1.flags.writeable = False
y1.flags.writeable = False

x2 = np.array([[0.51, 0.823, 0.6]], dtype=np.float32).T
y2 = np.array([[0.134, 0.5756]], dtype=np.float32).T

x2.flags.writeable = False
y2.flags.writeable = False


def train():
    weight_1 = np.random.rand(10, 3)
    bias_1 = np.random.rand(10, 1)

    weight_2 = np.random.rand(2, 10)
    bias_2 = np.random.rand(2, 1)

    learning_rate = 0.5

    epoch_size = 50_000

    layer_1 = LinearLayer(weight_1, bias_1, learning_rate)
    layer_2 = LinearLayer(weight_2, bias_2, learning_rate)

    for epoch in range(epoch_size):
        if epoch % 2 == 0:
            p1 = layer_1.forward(x1)
            layer_2.backward(p1, y1, True)
            layer_1.backward(x1, p1)
        else:
            p1 = layer_1.forward(x2)
            layer_2.backward(p1, y2, True)
            layer_1.backward(x2, p1)

    print(layer_2.forward(layer_1.forward(x1)).tolist())
    print(y1.tolist())

    print()

    print(layer_2.forward(layer_1.forward(x2)).tolist())
    print(y2.tolist())

    # y_1_predicted = layer.forward(x1)
    # y_2_predicted = layer.forward(x2)

    # print("error:")
    # print(layer.error(y_1_predicted, y1))
    # print(layer.error(y_2_predicted, y2))

    # print("answer:")
    # print(weight.tolist())
    # print(bias.tolist())


train()

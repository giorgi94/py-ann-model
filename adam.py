from numpy import sin, cos, pi, array, zeros, inf


class AdamOptimizer:
    beta_1: float = 0.9
    beta_2: float = 0.999

    alpha = 0.005

    eps: float = 10 ** (-8)

    def __init__(self, args):

        if isinstance(args, list):
            self.N = len(args)
            self.args = array(args, dtype=float)
        else:
            self.N = 0
            self.args = args

    def loss(self):
        pass

    def grad(self):
        pass

    def correction(self, steps=300):
        if self.N == 0:
            m = 0
            v = 0
        else:
            m = zeros((self.N,), dtype=float)
            v = zeros((self.N,), dtype=float)

        error = inf

        for t in range(1, steps + 1):
            err = self.loss()

            print("%03d" % t, err, err > error)

            if err > error:
                self.alpha = self.alpha / 10

            g = self.grad()

            m = self.beta_1 * m + (1 - self.beta_1) * g
            v = self.beta_2 * v + (1 - self.beta_2) * g ** 2

            m_hat = m / (1 - self.beta_1 ** t)
            v_hat = v / (1 - self.beta_2 ** t)

            self.args -= self.alpha * m_hat / (v_hat ** 0.5 + self.eps)


class ExampleOptimizer(AdamOptimizer):
    def loss(self, args=None):

        if args is None:
            args = self.args

        return (sin(args) - 0.5) ** 2 / 2

    def grad(self):
        h = 0.00005
        return (self.loss(self.args + h) - self.loss(self.args - h)) / (2 * h)


if __name__ == "__main__":

    optimizer = ExampleOptimizer(0.3)

    optimizer.correction(300)

    print(optimizer.args * 180 / pi)

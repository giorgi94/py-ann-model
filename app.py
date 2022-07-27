import numpy as np

from nn import LinearLayer, Model

train = [
    ([0.389, 0.134, 0.134], [0.8341, 0.9421]),
    ([0.51, 0.823, 0.6], [0.134, 0.5756]),
    ([0.81, 0.323, 0.76], [0.934, 0.0756]),
]


params = Model.generate_random_params(3, 8, 2)

model = Model(params, train)


learning_rate = 0.5

model.layers.extend([LinearLayer(w, b, learning_rate) for w, b in params])


model.train(20_000)

for X, Y in train:
    print(model.predict(np.array([X]).T).flatten())
    print(Y)
    print()

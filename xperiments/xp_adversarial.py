import os
import numpy as np

from brainforge import BackpropNetwork
from brainforge.layers import DenseLayer

from csxdata.utilities.vectorop import ravel_to_matrix, standardize, shuffle
from csxdata.utilities.loader import pull_mnist_data


EPOCHS = 30
ZDIM = 10
M = 4
K = 100


def sample_z(batch_size):
    return np.random.randn(batch_size, ZDIM)


X, Y = pull_mnist_data(split=0., fold=False)

generator = BackpropNetwork(input_shape=ZDIM, layerstack=[
    DenseLayer(60, activation="tanh"),
    DenseLayer(X.shape[-1], activation="linear")
], cost="mse", optimizer="adam")

discriminator = BackpropNetwork(input_shape=X.shape[-1], layerstack=[
    DenseLayer(60, activation="tanh"),
    DenseLayer(2, activation="softmax")
], cost="cxent", optimizer="adam")


for epoch in range(EPOCHS):
    X, = shuffle(X)
    for i, batch in enumerate(X[start:start+M] for start in range(0, len(X), M)):
        m = len(batch)
        z = sample_z(m)
        generated = generator.predict(z)
        x = np.concatenate((batch, generated))
        y = np.concatenate((np.ones(m), np.zeros(m)))[:, None]
        cost = discriminator.learn_batch(x, y)
        print(f"\rTraining discriminator: cost: {cost:.4f}", end="")
        if i >= K:
            break
    # print("\nTraining generator: cost: ", end="")
    # generated = generator.predict(sample_z(M))
    # dpred = discriminator.predict(generated)
    # discriminator_y = np.zeros(M)[:, None]
    # dcost = discriminator.cost(dpred, discriminator_y)
    # print(-dcost)
    # ddelta = discriminator.cost.derivative(dpred, discriminator_y)  # = D'(G(z))
    # gdelta = discriminator.backpropagate(-ddelta)
    # generator.backpropagate(gdelta)
    # generator.update(M)

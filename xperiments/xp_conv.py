from csxdata.utilities.loader import pull_mnist_data

from brainforge import BackpropNetwork
from brainforge.layers import ConvLayer, PoolLayer, Flatten, DenseLayer, Activation
from brainforge.optimization import RMSprop

lX, lY, tX, tY = pull_mnist_data(split=0.2)

ins, ous = lX.shape[1:], lY.shape[1:]
net = BackpropNetwork(input_shape=ins, layerstack=[
    ConvLayer(3, 8, 8, compiled=True),
    PoolLayer(3, compiled=True), Activation("tanh"),
    Flatten(), DenseLayer(60, activation="tanh"),
    DenseLayer(ous, activation="softmax")
], cost="xent", optimizer=RMSprop(eta=0.01))

net.fit(lX, lY, batch_size=32, epochs=100, validation=(tX, tY))

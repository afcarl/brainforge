from brainforge.util import etalon
from brainforge import LayerStack, BackpropNetwork
from brainforge.layers import DenseLayer


ls = LayerStack((4,), layers=[
    DenseLayer(120, activation="tanh"),
    DenseLayer(3, activation="softmax")
])

net = BackpropNetwork(ls, cost="xent", optimizer="momentum")
costs = net.fit(*etalon, epochs=300, validation=etalon, verbose=1)

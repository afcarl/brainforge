import numpy as np
from collections import deque

from csxdata.utilities.loader import pull_mnist_data

from brainforge import BackpropNetwork
from brainforge.layers.abstract_layer import LayerBase, NoParamMixin
from brainforge.layers import DenseLayer, Reshape, Flatten


class DNI(NoParamMixin, LayerBase):

    def __init__(self, synth: BackpropNetwork=None, **kw):
        super().__init__(**kw)
        self.synth = synth
        self.memory = deque()
        self._predictor = None
        self._previous = None

    def _default_synth(self):
        inshape = np.prod(self.inshape),
        synth = BackpropNetwork(input_shape=self.inshape, layerstack=[
            Flatten(),
            DenseLayer(int(np.prod(self.inshape[0])*1.5), activation="tanh"),
            DenseLayer(inshape, activation="linear"),
            Reshape(self.inshape)
        ], cost="mse", optimizer="adam")
        return synth

    def connect(self, brain):
        super().connect(brain)
        self._previous = brain.layers[-1]
        if self.synth is None:
            self.synth = self._default_synth()

    def feedforward(self, X):
        delta = self.synth.predict(X)
        self._previous.backpropagate(delta)
        if self.brain.learning:
            self.memory.append(delta)
        return X

    def backpropagate(self, delta):
        m = self.memory.popleft()
        print(f"Synth cost: {self.synth.cost(m, delta).sum():.4f}")
        synth_delta = self.synth.cost.derivative(m, delta)
        self.synth.backpropagate(synth_delta)
        self.synth.update(len(synth_delta))

    @property
    def outshape(self):
        return self.inshape


def build_decoupled_net(inshape, outshape):
    net = BackpropNetwork(input_shape=inshape, layerstack=[
        DenseLayer(60, activation="tanh"), DNI(),
        DenseLayer(outshape, activation="softmax")
    ], cost="xent", optimizer="adam")
    return net


def build_normal_net(inshape, outshape):
    net = BackpropNetwork(input_shape=inshape, layerstack=[
        DenseLayer(60, activation="tanh"),
        DenseLayer(outshape, activation="softmax")
    ], cost="xent", optimizer="adam")
    return net


def xperiment():
    lX, lY, tX, tY = pull_mnist_data()
    net = build_normal_net(lX.shape[1:], lY.shape[1:])
    net.fit(lX, lY, batch_size=128, validation=(tX, tY), verbose=1)


if __name__ == '__main__':
    xperiment()

import numpy as np

from .abstract_layer import LayerBase, NoParamMixin, FFBase
from ..atomic import Sigmoid
from ..util import rtm, scalX, zX, white
from ..config import floatX

sigmoid = Sigmoid()
X2, X3, X05 = scalX(2.), scalX(3.), scalX(0.5)


class HighwayLayer(FFBase):
    """
    Neural Highway Layer based on Srivastava et al., 2015
    """

    def __init__(self, activation="tanh", **kw):
        super().__init__(1, activation, **kw)
        self.gates = None

    def connect(self, brain):
        self.neurons = np.prod(brain.outshape)
        self.weights = white(self.neurons, self.neurons*3)
        self.biases = zX(self.neurons*3)
        super().connect(brain)

    def feedforward(self, X) -> np.ndarray:
        self.inputs = rtm(X)
        self.gates = self.inputs.dot(self.weights) + self.biases
        self.gates[:, :self.neurons] = self.activation.forward(self.gates[:, :self.neurons])
        self.gates[:, self.neurons:] = sigmoid.forward(self.gates[:, self.neurons:])
        h, t, c = np.split(self.gates, 3, axis=1)
        self.output = h * t + self.inputs * c
        return self.output.reshape(X.shape)

    def backpropagate(self, delta) -> np.ndarray:
        shape = delta.shape
        delta = rtm(delta)

        h, t, c = np.split(self.gates, 3, axis=1)

        dh = self.activation.backward(h) * t * delta
        dt = sigmoid.backward(t) * h * delta
        dc = sigmoid.backward(c) * self.inputs * delta
        dx = c * delta

        dgates = np.concatenate((dh, dt, dc), axis=1)
        self.nabla_w = self.inputs.T.dot(dgates)
        self.nabla_b = dgates.sum(axis=0)

        return (dgates.dot(self.weights.T) + dx).reshape(shape)

    @property
    def outshape(self):
        return self.inshape

    def __str__(self):
        return "Highway-{}".format(str(self.activation))


class DropOut(NoParamMixin, LayerBase):

    def __init__(self, dropchance):
        super().__init__()
        self.dropchance = scalX(1. - dropchance)
        self.mask = None
        self.inshape = None
        self.training = True

    def feedforward(self, X: np.ndarray) -> np.ndarray:
        self.inputs = X
        self.mask = np.random.uniform(0, 1, self.inshape) < self.dropchance  # type: np.ndarray
        self.mask.astype(floatX)
        self.output = X * (self.mask if self.brain.learning else self.dropchance)
        return self.output

    def backpropagate(self, delta: np.ndarray) -> np.ndarray:
        output = delta * self.mask
        self.mask = np.ones_like(self.mask) * self.dropchance
        return output

    @property
    def outshape(self):
        return self.inshape

    def __str__(self):
        return "DropOut({})".format(self.dropchance)


class BatchNormalization(LayerBase):

    def __init__(self, **kw):
        super().__init__(**kw)
        self.cache = []
        self.weights = np.array([1.], dtype=floatX)
        self.biases = np.array([0.], dtype=floatX)
        self.epsilon = scalX(1e-5)
        self.grad_w = np.array([0.], dtype=floatX)
        self.grad_b = np.array([0.], dtype=floatX)

    def feedforward(self, X: np.ndarray):
        mu = X.mean()
        var = X.var()
        vare = var + self.epsilon
        X_cent = (X - mu)
        X_scal = np.sqrt(vare)
        X_norm = X_cent / X_scal
        y = self.weights * X_norm + self.biases
        self.cache = [mu, var, vare, X_cent, X_scal, X_norm, y, len(X)]
        return y

    def backpropagate(self, delta):
        mu, var, vare, X_cent, X_scal, X_norm, y, m = self.cache
        sqvare = np.sqrt(vare)
        self.grad_w[0] = (delta * X_norm).sum()
        self.grad_b[0] = delta.sum()
        dX_norm = delta * self.biases[0]
        dvar = (dX_norm * X_cent).sum() * np.power(vare, -X3/X2) * -X05
        dmu = -dX_norm.sum() / sqvare + ((dvar * -X2) / m) * X_cent.sum()
        dX = dX_norm / sqvare + ((dvar * X2) / m) * X_cent + dmu / m
        return dX

    @property
    def outshape(self):
        return self.inshape

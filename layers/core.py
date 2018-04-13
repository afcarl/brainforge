import numpy as np

from .. import atomic
from ..util import zX, white

from .abstract_layer import LayerBase, NoParamMixin, FFBase


class DenseLayer(FFBase):

    def connect(self, brain):
        inshape = brain.outshape
        if len(inshape) != 1:
            err = "Dense only accepts input shapes with 1 dimension!\n"
            err += "Maybe you should consider placing <Flatten> before <Dense>?"
            raise RuntimeError(err)
        self.weights = white(inshape[0], self.neurons)
        self.biases = zX(self.neurons)
        super().connect(brain)
        if self.compiled:
            from .. import llatomic
            print("Compiling DenseLayer...")
            self.op = llatomic.DenseOp
        else:
            self.op = atomic.DenseOp

    def feedforward(self, X):
        self.inputs = X.copy()
        self.output = self.activation.forward(self.op.forward(
            X, self.weights, self.biases
        ))
        return self.output

    def backpropagate(self, delta):
        delta *= self.activation.backward(self.output)
        self.nabla_w, self.nabla_b, nX = self.op.backward(X=self.inputs, E=delta, W=self.weights)
        return nX

    def __str__(self):
        return "Dense-{}-{}".format(self.neurons, str(self.activation)[:5])


class Activation(NoParamMixin, LayerBase):

    def feedforward(self, X: np.ndarray) -> np.ndarray:
        self.output = self.activation.forward(X)
        return self.output

    def backpropagate(self, delta) -> np.ndarray:
        return delta * self.activation.backward(self.output)

    @property
    def outshape(self):
        return self.inshape

    def __str__(self):
        return "Activation-{}".format(str(self.activation))


class InputLayer(Activation):

    def __init__(self, input_shape):
        super().__init__(activation="linear")
        if not isinstance(input_shape, tuple):
            input_shape = (input_shape,)
        self.inshape = input_shape

    def connect(self, brain):
        pass


class Reshape(NoParamMixin, LayerBase):

    def __init__(self, shape=None):
        super().__init__(activation="linear")
        self.shape = shape

    def connect(self, brain):
        if self.shape is None:
            self.shape = np.prod(brain.outshape),
        super().connect(brain)

    def feedforward(self, X: np.ndarray) -> np.ndarray:
        return atomic.ReshapeOp.forward(X, self.shape)

    def backpropagate(self, delta) -> np.ndarray:
        return atomic.ReshapeOp.forward(delta, self.inshape)

    def capsule(self):
        return [self.inshape]

    @property
    def outshape(self):
        return self.shape

    def __str__(self):
        return self.__class__.__name__


class Flatten(Reshape):

    def __init__(self):
        super().__init__(None)

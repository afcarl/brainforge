import numpy as np

from .abstract_model import Model


class LayerStack(Model):

    def __init__(self, input_shape, layers=()):
        super().__init__(input_shape)
        self.layers = []
        self.architecture = []
        self.learning = False

        self._add_input_layer(input_shape)
        for layer in layers:
            self.add(layer)

    def _add_input_layer(self, input_shape):
        from ..layers.core import InputLayer
        inl = InputLayer(input_shape)
        inl.connect(brain=self)
        self.layers.append(inl)
        self.architecture.append(str(inl))

    def add(self, layer):
        layer.connect(self)
        self.layers.append(layer)
        self.architecture.append(str(layer))

    def pop(self):
        self.layers.pop()
        self.architecture.pop()

    def feedforward(self, X):
        for layer in self.layers:
            X = layer.feedforward(X)
        return X

    def get_states(self, unfold=True):
        hs = [layer.output for layer in self.layers]
        return np.concatenate(hs) if unfold else hs

    def get_weights(self, unfold=True):
        ws = [layer.get_weights(unfold=unfold) for
              layer in self.layers if layer.trainable]
        return np.concatenate(ws) if unfold else ws

    def set_weights(self, ws, fold=True):
        trl = (l for l in self.layers if l.trainable)
        if fold:
            start = 0
            for layer in trl:
                end = start + layer.nparams
                layer.set_weights(ws[start:end], fold=True)
                start = end
        else:
            for w, layer in zip(ws, trl):
                layer.set_weights(w, fold=False)

    def describe(self):
        return "Architecture: " + "->".join(self.architecture),

    def reset(self):
        for layer in (l for l in self.layers if l.trainable):
            layer.reset()

    @property
    def outshape(self):
        return self.layers[-1].outshape

    @property
    def nparams(self):
        return sum(layer.nparams for layer in self.layers if layer.trainable)

    def __iter__(self):
        return iter(self.layers)

    def __getitem__(self, item):
        return self.layers.__getitem__(item)

    predict = feedforward

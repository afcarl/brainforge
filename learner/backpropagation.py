import numpy as np

from .abstract_learner import Learner
from ..optimization import optimizers, GradientDescent


class BackpropNetwork(Learner):

    def __init__(self, layerstack, cost="mse", optimizer="sgd", name="", **kw):
        super().__init__(layerstack, cost, name, **kw)
        self.optimizer = (
            optimizer if isinstance(optimizer, GradientDescent) else optimizers[optimizer]()
        )
        self.optimizer.initialize(nparams=self.layers.nparams)

    def learn_batch(self, X, Y, w=None, **kw):
        m = len(X)
        preds = self.predict(X)
        delta = self.cost.derivative(preds, Y)
        if w is not None:
            delta *= w[:, None]
        self.backpropagate(delta)
        self.update(m)
        return self.cost(self.output, Y) / m

    def backpropagate(self, error):
        # TODO: optimize this, skip untrainable layers at the beginning
        for layer in self.layers[-1:0:-1]:
            error = layer.backpropagate(error)
            if error is None:
                break
        return error

    def update(self, m):
        W = self.layers.get_weights(unfold=True)
        gW = self.get_gradients(unfold=True)
        oW = self.optimizer.optimize(W, gW, m)
        self.layers.set_weights(oW, fold=True)

    def get_weights(self, unfold=True):
        return self.layers.get_weights(unfold=unfold)

    def set_weights(self, ws, fold=True):
        self.layers.set_weights(ws=ws, fold=fold)

    def get_gradients(self, unfold=True):
        nabla = [layer.get_gradients(unfold=unfold) for
                 layer in self.layers if layer.trainable]
        return np.concatenate(nabla) if unfold else nabla

    def set_gradients(self, nabla, fold=True):
        trl = (l for l in self.layers if l.trainable)
        if fold:
            start = 0
            for layer in trl:
                end = start + layer.nparams
                layer.set_gradients(nabla[start:end], fold=True)
                start = end
        else:
            for grad, layer in zip(nabla, trl):
                layer.set_gradients(grad, fold=False)

    @property
    def nparams(self):
        return self.layers.nparams

    train_on_batch = learn_batch
